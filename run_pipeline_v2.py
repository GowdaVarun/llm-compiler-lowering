"""
Full pipeline runner v2 — with working models and rate-limit handling.
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, "/app/project")

from source_constructs import ALL_CONSTRUCTS, CONSTRUCT_MAP, get_summary_table
from ir_generator import IRGenerationPipeline, ModelConfig, GenerationResult, extract_ir_from_response
from ir_validator import validate_ir, validate_and_compare
from failure_analyzer import FailureModeAnalyzer, FAILURE_TAXONOMY
from repair_loop import RepairLoop, compute_repair_statistics
from report_generator import generate_report

OUTPUT_DIR = "/app/project/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    start_time = time.time()
    print("=" * 70)
    print(f"AI-ASSISTED COMPILER LOWERING: LLM -> LLVM IR")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    constructs = ALL_CONSTRUCTS
    constructs_map = CONSTRUCT_MAP

    models = [
        ModelConfig(
            model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            display_name="Qwen2.5-Coder-32B",
            provider="auto",
            max_tokens=4096,
            temperature=0.1,
        ),
        ModelConfig(
            model_id="Qwen/Qwen2.5-72B-Instruct",
            display_name="Qwen2.5-72B",
            provider="auto",
            max_tokens=4096,
            temperature=0.1,
        ),
        ModelConfig(
            model_id="meta-llama/Llama-3.3-70B-Instruct",
            display_name="Llama-3.3-70B",
            provider="auto",
            max_tokens=4096,
            temperature=0.1,
        ),
        ModelConfig(
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            display_name="Llama-3.1-8B",
            provider="auto",
            max_tokens=4096,
            temperature=0.1,
        ),
    ]

    strategies = ["basic", "cot"]

    # ================================================================
    # PHASE 1: GENERATION with rate-limit handling
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: LLM IR GENERATION")
    print("=" * 70)

    pipeline = IRGenerationPipeline(models=models)
    total = len(constructs) * len(models) * len(strategies)
    count = 0
    retries_total = 0

    for construct in constructs:
        for model in models:
            for strategy in strategies:
                count += 1
                print(f"\n[{count}/{total}] {construct.id} | {model.display_name} | {strategy}")

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        result = pipeline.generate_single(construct, model, strategy)
                        if result.error and "402" in str(result.error):
                            wait = 2 ** attempt * 2
                            print(f"  Rate limited, waiting {wait}s... (attempt {attempt+1})")
                            time.sleep(wait)
                            # Remove the failed result
                            pipeline.results.pop()
                            retries_total += 1
                            continue
                        elif result.error:
                            print(f"  ERROR: {result.error[:80]}")
                        else:
                            # Quick validation peek
                            r = validate_ir(result.generated_ir)
                            v_status = "VALID" if r.is_valid else f"{r.error_count} errors"
                            print(f"  OK ({result.generation_time_s}s) -> {v_status}")
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait = 2 ** attempt * 2
                            print(f"  Exception: {str(e)[:60]}, retrying in {wait}s...")
                            time.sleep(wait)
                            retries_total += 1
                        else:
                            print(f"  FAILED after {max_retries} attempts: {str(e)[:80]}")
                            result = GenerationResult(
                                construct_id=construct.id,
                                model_name=model.display_name,
                                model_id=model.model_id,
                                prompt_strategy=strategy,
                                source_code=construct.source_code,
                                generated_ir="",
                                raw_response="",
                                generation_time_s=0,
                                error=str(e),
                            )
                            pipeline.results.append(result)

                # Small delay between requests
                time.sleep(0.5)

    print(f"\nGeneration complete. Total retries: {retries_total}")
    gen_path = os.path.join(OUTPUT_DIR, "generation_results.json")
    pipeline.save_results(gen_path)

    # ================================================================
    # PHASE 2: FAILURE ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: FAILURE MODE ANALYSIS")
    print("=" * 70)

    analyzer = FailureModeAnalyzer(pipeline.results, constructs_map)
    analyses = analyzer.analyze_all(verbose=True)
    analysis_path = os.path.join(OUTPUT_DIR, "failure_analysis.json")
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
        repairer = RepairLoop(max_iterations=3)
        repair_results = repairer.repair_batch(to_repair, verbose=True)

        repair_path = os.path.join(OUTPUT_DIR, "repair_results.json")
        RepairLoop.save_repair_results(repair_results, repair_path)

        repair_stats = compute_repair_statistics(repair_results)
        print("\n--- Repair Statistics ---")
        for key, val in repair_stats.items():
            print(f"  {key}: {val}")
        with open(os.path.join(OUTPUT_DIR, "repair_statistics.json"), "w") as f:
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
        examples.append({
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
        })

    with open(os.path.join(OUTPUT_DIR, "detailed_examples.json"), "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Saved {len(examples)} detailed examples")

    generate_report(OUTPUT_DIR, os.path.join(OUTPUT_DIR, "REPORT.md"))

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"{'=' * 70}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f} ({size:,} bytes)")


if __name__ == "__main__":
    main()
