"""
Single-command full pipeline runner for local Ollama models.

Preserves the same pipeline stages as run_pipeline_v2:
  1) generation
  2) failure analysis
  3) iterative repair loop
  4) examples + report
"""

import argparse
import json
import os
import time
from datetime import datetime

from source_constructs import ALL_CONSTRUCTS, CONSTRUCT_MAP
from ir_generator import IRGenerationPipeline, ModelConfig, GenerationResult
from ir_validator import validate_ir, detect_llvm_tools, validate_ir_with_llvm_tools
from failure_analyzer import FailureModeAnalyzer
from repair_loop import RepairLoop, compute_repair_statistics
from report_generator import generate_report


def parse_levels(levels_arg: str) -> set:
    levels = set()
    for token in levels_arg.split(","):
        t = token.strip().upper()
        if not t:
            continue
        if t.startswith("L"):
            t = t[1:]
        if not t.isdigit():
            raise ValueError(f"Invalid level token: {token}")
        value = int(t)
        if value < 1 or value > 6:
            raise ValueError(f"Level out of range (1-6): {value}")
        levels.add(value)
    return levels


def parse_csv(values: str) -> list:
    return [v.strip() for v in values.split(",") if v.strip()]


def select_constructs(levels_arg: str = "", ids_arg: str = "") -> list:
    constructs = list(ALL_CONSTRUCTS)

    if levels_arg:
        levels = parse_levels(levels_arg)
        constructs = [c for c in constructs if c.level in levels]

    if ids_arg:
        requested_ids = parse_csv(ids_arg)
        missing = [cid for cid in requested_ids if cid not in CONSTRUCT_MAP]
        if missing:
            raise ValueError(f"Unknown construct ids: {', '.join(missing)}")
        id_set = set(requested_ids)
        constructs = [c for c in constructs if c.id in id_set]

    if not constructs:
        raise ValueError("Construct selection is empty. Adjust --levels/--construct-ids.")

    return constructs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full LLM->LLVM pipeline against local Ollama."
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:3b",
        help="Ollama model tag for generation (default: qwen2.5-coder:3b)",
    )
    parser.add_argument(
        "--repair-model",
        default=None,
        help="Ollama model tag for repair loop (default: same as --model)",
    )
    parser.add_argument(
        "--model-display-name",
        default=None,
        help="Friendly model name in result files (default: same as --model)",
    )
    parser.add_argument(
        "--strategies",
        default="basic,cot",
        help="Comma-separated prompt strategies: basic,cot,with_hints",
    )
    parser.add_argument(
        "--levels",
        default="",
        help="Construct levels to include, e.g. '1,2,3' or 'L1,L2'",
    )
    parser.add_argument(
        "--construct-ids",
        default="",
        help="Specific construct ids, e.g. 'L2_01,L4_02'",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max repair iterations per construct (default: 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for generation/repair (default: 0.1)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Token budget per generation call (default: 4096)",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="Retry attempts for generation API errors (default: 3)",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.2,
        help="Delay seconds between generation requests (default: 0.2)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama server base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "ollama_results"),
        help="Directory to write all outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    strategies = parse_csv(args.strategies)
    allowed = {"basic", "cot", "with_hints"}
    invalid = [s for s in strategies if s not in allowed]
    if invalid:
        raise ValueError(f"Unsupported strategies: {', '.join(invalid)}")
    if not strategies:
        raise ValueError("At least one strategy is required.")

    constructs = select_constructs(args.levels, args.construct_ids)
    constructs_map = {c.id: c for c in constructs}
    os.makedirs(args.output_dir, exist_ok=True)
    llvm_tools = detect_llvm_tools()
    if llvm_tools["missing"]:
        raise RuntimeError(
            f"Missing LLVM tools required by this pipeline: {', '.join(llvm_tools['missing'])}. "
            "Please install LLVM and ensure llvm-as and opt are in PATH."
        )

    model_display_name = args.model_display_name or args.model
    generation_model = ModelConfig(
        model_id=args.model,
        display_name=model_display_name,
        provider="ollama",
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        base_url=args.base_url,
    )

    print("=" * 70)
    print("AI-ASSISTED COMPILER LOWERING: OLLAMA -> LLVM IR")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model} | Repair model: {args.repair_model or args.model}")
    print(f"Constructs: {len(constructs)} | Strategies: {', '.join(strategies)}")
    print(f"LLVM tools: llvm-as={llvm_tools['llvm_as']} | opt={llvm_tools['opt']}")
    print("=" * 70)

    # ================================================================
    # PHASE 1: GENERATION
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: LLM IR GENERATION")
    print("=" * 70)

    pipeline = IRGenerationPipeline(models=[generation_model], ollama_base_url=args.base_url)
    total = len(constructs) * len(strategies)
    count = 0
    retries_total = 0
    llvm_validation_results = []

    for construct in constructs:
        for strategy in strategies:
            count += 1
            print(f"\n[{count}/{total}] {construct.id} | {model_display_name} | {strategy}")

            for attempt in range(args.retry_attempts):
                try:
                    result = pipeline.generate_single(construct, generation_model, strategy)
                    if result.error:
                        if attempt < args.retry_attempts - 1:
                            wait = 2 ** attempt
                            print(f"  ERROR: {result.error[:120]}")
                            print(f"  Retrying in {wait}s... ({attempt + 1}/{args.retry_attempts})")
                            time.sleep(wait)
                            pipeline.results.pop()
                            retries_total += 1
                            continue
                        print(f"  ERROR: {result.error[:120]}")
                    else:
                        r = validate_ir(result.generated_ir)
                        llvm_check = validate_ir_with_llvm_tools(
                            result.generated_ir,
                            llvm_as_path=llvm_tools["llvm_as"],
                            opt_path=llvm_tools["opt"],
                        )
                        llvm_validation_results.append(
                            {
                                "construct_id": construct.id,
                                "model": model_display_name,
                                "strategy": strategy,
                                "llvm_tool_validation": llvm_check,
                            }
                        )
                        v_status = "VALID" if r.is_valid else f"{r.error_count} errors"
                        llvm_status = "LLVM_OK" if llvm_check["is_valid"] else "LLVM_FAIL"
                        print(f"  OK ({result.generation_time_s}s) -> {v_status} | {llvm_status}")
                        if not llvm_check["is_valid"] and llvm_check.get("stderr"):
                            llvm_reason = llvm_check["stderr"][-1].splitlines()[0][:180]
                            print(f"  LLVM reason: {llvm_reason}")
                    break
                except Exception as e:
                    if attempt < args.retry_attempts - 1:
                        wait = 2 ** attempt
                        print(f"  Exception: {str(e)[:100]}")
                        print(f"  Retrying in {wait}s... ({attempt + 1}/{args.retry_attempts})")
                        time.sleep(wait)
                        retries_total += 1
                    else:
                        print(f"  FAILED after {args.retry_attempts} attempts: {str(e)[:120]}")
                        result = GenerationResult(
                            construct_id=construct.id,
                            model_name=model_display_name,
                            model_id=args.model,
                            prompt_strategy=strategy,
                            source_code=construct.source_code,
                            generated_ir="",
                            raw_response="",
                            generation_time_s=0,
                            error=str(e),
                        )
                        pipeline.results.append(result)

            time.sleep(args.request_delay)

    print(f"\nGeneration complete. Total retries: {retries_total}")
    gen_path = os.path.join(args.output_dir, "generation_results.json")
    pipeline.save_results(gen_path)
    with open(os.path.join(args.output_dir, "llvm_tool_validation.json"), "w", encoding="utf-8") as f:
        json.dump(llvm_validation_results, f, indent=2)

    # ================================================================
    # PHASE 2: FAILURE ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: FAILURE MODE ANALYSIS")
    print("=" * 70)

    analyzer = FailureModeAnalyzer(pipeline.results, constructs_map)
    analyzer.analyze_all(verbose=True)
    analysis_path = os.path.join(args.output_dir, "failure_analysis.json")
    analyzer.save_analysis(analysis_path)
    analyzer.print_summary()

    # ================================================================
    # PHASE 3: REPAIR LOOP
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: REPAIR LOOP")
    print("=" * 70)

    seen_constructs = set()
    to_repair = []
    for a in analyzer.analyses:
        if not a.is_valid and a.generated_ir.strip() and a.construct_id not in seen_constructs:
            to_repair.append((constructs_map[a.construct_id], a.generated_ir))
            seen_constructs.add(a.construct_id)

    if to_repair:
        print(f"Attempting repair on {len(to_repair)} unique failed constructs...")
        repair_model = args.repair_model or args.model
        repairer = RepairLoop(
            repair_model_id=repair_model,
            provider="ollama",
            max_iterations=args.max_iterations,
            ollama_base_url=args.base_url,
        )
        repair_results = repairer.repair_batch(to_repair, verbose=True)

        repair_path = os.path.join(args.output_dir, "repair_results.json")
        RepairLoop.save_repair_results(repair_results, repair_path)

        repair_llvm_checks = []
        for rr in repair_results:
            llvm_check = validate_ir_with_llvm_tools(
                rr.final_ir,
                llvm_as_path=llvm_tools["llvm_as"],
                opt_path=llvm_tools["opt"],
            )
            repair_llvm_checks.append(
                {
                    "construct_id": rr.construct_id,
                    "repair_successful": rr.repair_successful,
                    "final_errors": rr.final_errors,
                    "llvm_tool_validation": llvm_check,
                }
            )
        with open(os.path.join(args.output_dir, "repair_llvm_tool_validation.json"), "w", encoding="utf-8") as f:
            json.dump(repair_llvm_checks, f, indent=2)

        repair_stats = compute_repair_statistics(repair_results)
        print("\n--- Repair Statistics ---")
        with open(os.path.join(args.output_dir, "repair_statistics.json"), "w", encoding="utf-8") as f:
            json.dump(repair_stats, f, indent=2)
    else:
        print("All generations were valid — no repair needed!")

    # ================================================================
    # PHASE 4: EXAMPLES + REPORT
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: EXAMPLES & REPORT")
    print("=" * 70)

    examples = []
    for analysis in analyzer.analyses:
        construct = constructs_map[analysis.construct_id]
        examples.append(
            {
                "construct_id": analysis.construct_id,
                "construct_name": construct.name,
                "level": construct.level,
                "category": construct.category,
                "model": analysis.model_name,
                "prompt_strategy": analysis.prompt_strategy,
                "source_code": construct.source_code,
                "reference_ir": construct.expected_ir,
                "generated_ir": analysis.generated_ir,
                "is_valid": analysis.is_valid,
                "is_compilable": analysis.is_compilable,
                "is_structurally_correct": analysis.is_structurally_correct,
                "failure_count": len(analysis.failures),
                "failures": [f.to_dict() for f in analysis.failures],
                "key_ir_features_expected": construct.key_ir_features,
            }
        )

    with open(os.path.join(args.output_dir, "detailed_examples.json"), "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
    print(f"Saved {len(examples)} detailed examples")

    generate_report(args.output_dir, os.path.join(args.output_dir, "REPORT.md"))

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"{'=' * 70}")
    for filename in sorted(os.listdir(args.output_dir)):
        file_path = os.path.join(args.output_dir, filename)
        size = os.path.getsize(file_path)
        print(f"  {filename} ({size:,} bytes)")


if __name__ == "__main__":
    main()
