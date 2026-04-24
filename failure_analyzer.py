"""
Failure Mode Analyzer — Categorizes and quantifies LLM IR generation failures.

Based on the taxonomy from arXiv:2502.06854, 2309.07062, 2403.05286, and 2407.06153:
  CLASS 1: Structural/Syntactic (SSA violations, incomplete syntax, malformed types)
  CLASS 2: Control Flow (missing labels, wrong branches, bad phi nodes, loop approx.)
  CLASS 3: Type System (type mismatch, pointer confusion, missing definitions)
  CLASS 4: Semantic/Functional (wrong computation, pattern matching, constant folding)
  CLASS 5: Scale/Context (overflow, inter-procedural loss)
"""

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

from ir_validator import (
    validate_ir, validate_and_compare,
    ErrorCategory, ErrorSeverity, ValidationReport
)


# ============================================================================
# Failure Mode Taxonomy
# ============================================================================

FAILURE_TAXONOMY = {
    "CLASS_1": {
        "name": "Structural/Syntactic Failures",
        "description": "Violations of basic LLVM IR syntax and SSA form",
        "sub_categories": {
            "1.1": "SSA violation: reuse of %name, multiple definitions",
            "1.2": "Incomplete syntax: unclosed blocks, missing terminators",
            "1.3": "Malformed type annotations: wrong integer width, float mismatch",
            "1.4": "Invalid instruction format: wrong operand count, illegal opcode",
            "1.5": "Code fence artifacts: markdown or text mixed into IR",
        }
    },
    "CLASS_2": {
        "name": "Control Flow Failures",
        "description": "Incorrect control flow graph structure",
        "sub_categories": {
            "2.1": "Missing basic block labels or duplicate labels",
            "2.2": "Branch to non-existent block",
            "2.3": "Incorrect phi node predecessors",
            "2.4": "Loops approximated/simplified rather than faithfully reconstructed",
            "2.5": "Unreachable code inserted (dead branches)",
            "2.6": "Missing terminator in basic block",
        }
    },
    "CLASS_3": {
        "name": "Type System Failures",
        "description": "Type mismatches and incorrect type usage",
        "sub_categories": {
            "3.1": "Type mismatch in operations (e.g., add i64 on i32 operands)",
            "3.2": "Pointer vs value confusion (wrong load/store types)",
            "3.3": "Missing struct/aggregate type definitions",
            "3.4": "Undefined function signatures (hallucinated callee types)",
            "3.5": "Integer/float operation confusion (add vs fadd)",
        }
    },
    "CLASS_4": {
        "name": "Semantic/Functional Failures",
        "description": "Valid IR that computes wrong result",
        "sub_categories": {
            "4.1": "Wrong computation despite valid IR structure",
            "4.2": "Heuristic pattern matching (plausible but incorrect IR)",
            "4.3": "Failed constant folding / arithmetic errors",
            "4.4": "Data-flow analysis errors (incorrect use-def chains)",
            "4.5": "Hallucinated intrinsics or functions",
        }
    },
    "CLASS_5": {
        "name": "Scale/Context Failures",
        "description": "Failures related to input size and context limitations",
        "sub_categories": {
            "5.1": "Context overflow causing truncated IR",
            "5.2": "Inter-procedural context loss",
            "5.3": "Empty or no IR generated",
        }
    },
}


@dataclass
class FailureInstance:
    """A single failure instance with its categorization."""
    construct_id: str
    model_name: str
    prompt_strategy: str
    failure_class: str  # "CLASS_1", "CLASS_2", etc.
    sub_category: str   # "1.1", "1.2", etc.
    description: str
    severity: str  # "error" or "warning"
    evidence: Optional[str] = None  # The specific line/pattern

    def to_dict(self):
        return asdict(self)


@dataclass
class AnalysisResult:
    """Complete analysis of a single generation."""
    construct_id: str
    model_name: str
    prompt_strategy: str
    generated_ir: str
    reference_ir: str
    validation_report: dict
    comparison: dict
    failures: list = field(default_factory=list)  # list of FailureInstance
    is_valid: bool = False
    is_compilable: bool = False
    is_structurally_correct: bool = False  # matches reference structure
    generation_time_s: float = 0.0

    def to_dict(self):
        d = asdict(self)
        d['failure_count'] = len(self.failures)
        d['failure_classes'] = list(set(f.failure_class for f in self.failures))
        return d


# ============================================================================
# Failure Classifier
# ============================================================================

def classify_failures(validation_report: dict, comparison: dict,
                      construct_id: str, model_name: str,
                      prompt_strategy: str, generated_ir: str) -> list:
    """
    Classify validation errors and comparison mismatches into the failure taxonomy.
    Returns a list of FailureInstance objects.
    """
    failures = []

    def add(cls, sub, desc, sev="error", evidence=None):
        failures.append(FailureInstance(
            construct_id=construct_id,
            model_name=model_name,
            prompt_strategy=prompt_strategy,
            failure_class=cls,
            sub_category=sub,
            description=desc,
            severity=sev,
            evidence=evidence,
        ))

    # Check for empty/missing IR
    if not generated_ir or not generated_ir.strip():
        add("CLASS_5", "5.3", "No IR generated", evidence="empty output")
        return failures

    if 'define ' not in generated_ir and 'declare ' not in generated_ir:
        add("CLASS_1", "1.5", "Output does not contain valid LLVM IR",
            evidence=generated_ir[:200])
        return failures

    # Classify validation errors
    for error in validation_report.get("errors", []):
        cat = error["category"]
        msg = error["message"]
        sev = error["severity"]

        if cat == "ssa":
            if "defined more than once" in msg:
                add("CLASS_1", "1.1", msg, sev, error.get("line_content"))
            elif "undefined SSA variable" in msg:
                add("CLASS_1", "1.1", msg, sev, error.get("line_content"))
            else:
                add("CLASS_1", "1.1", msg, sev, error.get("line_content"))

        elif cat == "syntax":
            add("CLASS_1", "1.2", msg, sev, error.get("line_content"))

        elif cat == "type":
            if "Integer operation" in msg or "Float operation" in msg:
                add("CLASS_3", "3.5", msg, sev, error.get("line_content"))
            elif "Void function" in msg or "returns" in msg:
                add("CLASS_3", "3.1", msg, sev, error.get("line_content"))
            else:
                add("CLASS_3", "3.1", msg, sev, error.get("line_content"))

        elif cat == "control_flow":
            if "no terminator" in msg:
                add("CLASS_2", "2.6", msg, sev, error.get("line_content"))
            elif "Branch target" in msg and "not found" in msg:
                add("CLASS_2", "2.2", msg, sev, error.get("line_content"))
            elif "Duplicate basic block" in msg:
                add("CLASS_2", "2.1", msg, sev, error.get("line_content"))
            elif "phi" in msg.lower() or "Phi" in msg:
                add("CLASS_2", "2.3", msg, sev, error.get("line_content"))
            else:
                add("CLASS_2", "2.5", msg, sev, error.get("line_content"))

        elif cat == "semantic":
            if "predicate" in msg:
                add("CLASS_1", "1.4", msg, sev, error.get("line_content"))
            elif "Phi node after" in msg:
                add("CLASS_2", "2.3", msg, sev, error.get("line_content"))
            else:
                add("CLASS_4", "4.2", msg, sev, error.get("line_content"))

    # Classify structural comparison failures
    if comparison:
        if not comparison.get("function_match"):
            missing = comparison.get("missing_functions", set())
            extra = comparison.get("extra_functions", set())
            if missing:
                add("CLASS_4", "4.5",
                    f"Missing functions: {missing}", "error")
            if extra:
                add("CLASS_4", "4.5",
                    f"Extra/hallucinated functions: {extra}", "warning")

        for fname, block_info in comparison.get("block_count_match", {}).items():
            if not block_info.get("match"):
                ref_count = block_info["reference"]
                gen_count = block_info["generated"]
                if gen_count < ref_count:
                    add("CLASS_2", "2.4",
                        f"@{fname}: {gen_count} blocks vs {ref_count} expected "
                        f"(simplified control flow)", "warning")
                elif gen_count > ref_count:
                    add("CLASS_2", "2.5",
                        f"@{fname}: {gen_count} blocks vs {ref_count} expected "
                        f"(extra code paths)", "warning")

    return failures


# ============================================================================
# Bulk Analysis
# ============================================================================

class FailureModeAnalyzer:
    """Analyzes a batch of LLM generation results."""

    def __init__(self, results, constructs_map):
        """
        Args:
            results: list of GenerationResult
            constructs_map: dict mapping construct_id -> SourceConstruct
        """
        self.results = results
        self.constructs = constructs_map
        self.analyses = []

    def analyze_all(self, verbose=True):
        """Run analysis on all results."""
        for i, result in enumerate(self.results):
            if result.error:
                # Generation itself failed
                analysis = AnalysisResult(
                    construct_id=result.construct_id,
                    model_name=result.model_name,
                    prompt_strategy=result.prompt_strategy,
                    generated_ir=result.generated_ir,
                    reference_ir=self.constructs[result.construct_id].expected_ir,
                    validation_report={},
                    comparison={},
                    is_valid=False,
                    is_compilable=False,
                    generation_time_s=result.generation_time_s,
                )
                analysis.failures = [FailureInstance(
                    construct_id=result.construct_id,
                    model_name=result.model_name,
                    prompt_strategy=result.prompt_strategy,
                    failure_class="CLASS_5",
                    sub_category="5.3",
                    description=f"API/generation error: {result.error}",
                    severity="error",
                )]
                self.analyses.append(analysis)
                if verbose:
                    print(f"[{i+1}/{len(self.results)}] {result.construct_id} | "
                          f"{result.model_name} | API ERROR")
                continue

            construct = self.constructs[result.construct_id]
            reference_ir = construct.expected_ir

            # Validate
            vc = validate_and_compare(result.generated_ir, reference_ir)

            # Classify failures
            failures = classify_failures(
                vc["validation"],
                vc["comparison"],
                result.construct_id,
                result.model_name,
                result.prompt_strategy,
                result.generated_ir,
            )

            analysis = AnalysisResult(
                construct_id=result.construct_id,
                model_name=result.model_name,
                prompt_strategy=result.prompt_strategy,
                generated_ir=result.generated_ir,
                reference_ir=reference_ir,
                validation_report=vc["validation"],
                comparison=vc["comparison"],
                failures=failures,
                is_valid=vc["validation"]["is_valid"],
                is_compilable=vc["validation"]["is_compilable"],
                is_structurally_correct=(
                    vc["comparison"].get("function_match", False) and
                    all(bi.get("match", False)
                        for bi in vc["comparison"].get("block_count_match", {}).values())
                ),
                generation_time_s=result.generation_time_s,
            )
            self.analyses.append(analysis)

            if verbose:
                status = "VALID" if analysis.is_valid else f"FAIL ({len(failures)} issues)"
                print(f"[{i+1}/{len(self.results)}] {result.construct_id} | "
                      f"{result.model_name} | {status}")

        return self.analyses

    def get_statistics(self) -> dict:
        """Compute aggregate statistics across all analyses."""
        stats = {
            "total_generations": len(self.analyses),
            "valid_count": sum(1 for a in self.analyses if a.is_valid),
            "compilable_count": sum(1 for a in self.analyses if a.is_compilable),
            "structurally_correct_count": sum(1 for a in self.analyses if a.is_structurally_correct),
            "total_failures": sum(len(a.failures) for a in self.analyses),
            "by_model": {},
            "by_construct_level": {},
            "by_failure_class": Counter(),
            "by_sub_category": Counter(),
            "by_prompt_strategy": {},
            "failure_class_by_model": {},
            "avg_generation_time": {},
        }

        # Per-model stats
        models = set(a.model_name for a in self.analyses)
        for model in models:
            model_analyses = [a for a in self.analyses if a.model_name == model]
            valid = sum(1 for a in model_analyses if a.is_valid)
            compilable = sum(1 for a in model_analyses if a.is_compilable)
            structural = sum(1 for a in model_analyses if a.is_structurally_correct)
            total = len(model_analyses)
            avg_time = sum(a.generation_time_s for a in model_analyses) / max(total, 1)

            stats["by_model"][model] = {
                "total": total,
                "valid": valid,
                "compilable": compilable,
                "structurally_correct": structural,
                "valid_rate": round(valid / max(total, 1) * 100, 1),
                "compilable_rate": round(compilable / max(total, 1) * 100, 1),
                "avg_generation_time_s": round(avg_time, 2),
            }

            # Failure class breakdown per model
            class_counts = Counter()
            for a in model_analyses:
                for f in a.failures:
                    class_counts[f.failure_class] += 1
            stats["failure_class_by_model"][model] = dict(class_counts)

        # Per-level stats
        for a in self.analyses:
            construct = self.constructs.get(a.construct_id)
            if construct:
                level = f"L{construct.level}"
                if level not in stats["by_construct_level"]:
                    stats["by_construct_level"][level] = {"total": 0, "valid": 0, "compilable": 0}
                stats["by_construct_level"][level]["total"] += 1
                if a.is_valid:
                    stats["by_construct_level"][level]["valid"] += 1
                if a.is_compilable:
                    stats["by_construct_level"][level]["compilable"] += 1

        # Failure class and sub-category counts
        for a in self.analyses:
            for f in a.failures:
                stats["by_failure_class"][f.failure_class] += 1
                stats["by_sub_category"][f.sub_category] += 1

        stats["by_failure_class"] = dict(stats["by_failure_class"])
        stats["by_sub_category"] = dict(stats["by_sub_category"])

        # Per-strategy stats
        strategies = set(a.prompt_strategy for a in self.analyses)
        for strat in strategies:
            strat_analyses = [a for a in self.analyses if a.prompt_strategy == strat]
            valid = sum(1 for a in strat_analyses if a.is_valid)
            total = len(strat_analyses)
            stats["by_prompt_strategy"][strat] = {
                "total": total,
                "valid": valid,
                "valid_rate": round(valid / max(total, 1) * 100, 1),
            }

        return stats

    def save_analysis(self, filepath: str):
        """Save full analysis to JSON."""
        data = {
            "analyses": [a.to_dict() for a in self.analyses],
            "statistics": self.get_statistics(),
            "taxonomy": FAILURE_TAXONOMY,
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved analysis to {filepath}")

    def print_summary(self):
        """Print a human-readable summary."""
        stats = self.get_statistics()
        total = stats["total_generations"]
        print("=" * 70)
        print("FAILURE MODE ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Total generations: {total}")
        print(f"Valid (no errors): {stats['valid_count']} ({stats['valid_count']/max(total,1)*100:.1f}%)")
        print(f"Compilable: {stats['compilable_count']} ({stats['compilable_count']/max(total,1)*100:.1f}%)")
        print(f"Structurally correct: {stats['structurally_correct_count']} ({stats['structurally_correct_count']/max(total,1)*100:.1f}%)")
        print(f"Total failure instances: {stats['total_failures']}")

        print("\n--- By Model ---")
        for model, ms in sorted(stats["by_model"].items()):
            print(f"  {model}: {ms['valid']}/{ms['total']} valid "
                  f"({ms['valid_rate']}%), compilable: {ms['compilable_rate']}%, "
                  f"avg time: {ms['avg_generation_time_s']}s")

        print("\n--- By Construct Level ---")
        for level, ls in sorted(stats["by_construct_level"].items()):
            rate = ls['valid'] / max(ls['total'], 1) * 100
            print(f"  {level}: {ls['valid']}/{ls['total']} valid ({rate:.1f}%)")

        print("\n--- By Failure Class ---")
        for cls, count in sorted(stats["by_failure_class"].items()):
            name = FAILURE_TAXONOMY.get(cls, {}).get("name", cls)
            print(f"  {cls} ({name}): {count}")

        print("\n--- By Sub-Category (top 10) ---")
        sorted_subs = sorted(stats["by_sub_category"].items(), key=lambda x: -x[1])
        for sub, count in sorted_subs[:10]:
            cls = "CLASS_" + sub.split(".")[0]
            desc = FAILURE_TAXONOMY.get(cls, {}).get("sub_categories", {}).get(sub, sub)
            print(f"  {sub}: {count} — {desc}")

        if stats.get("by_prompt_strategy"):
            print("\n--- By Prompt Strategy ---")
            for strat, ss in stats["by_prompt_strategy"].items():
                print(f"  {strat}: {ss['valid']}/{ss['total']} valid ({ss['valid_rate']}%)")

        print("\n--- Failure Class by Model ---")
        for model, classes in sorted(stats["failure_class_by_model"].items()):
            parts = [f"{cls}:{cnt}" for cls, cnt in sorted(classes.items())]
            print(f"  {model}: {', '.join(parts)}")
