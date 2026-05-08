"""
Repair Loop Architecture — Iterative validation and LLM-based repair.

Three-tier architecture:
  Tier 1: Syntactic repair (compile errors)
  Tier 2: SSA/Type repair (verification errors)
  Tier 3: Semantic repair (functional errors via test comparison)

And compiler feedback approach from arXiv:2403.14714.
"""

import os
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from huggingface_hub import InferenceClient

from ir_validator import validate_ir, LLVMIRValidator, ErrorCategory


# ============================================================================
# Repair prompts
# ============================================================================

REPAIR_SYSTEM_PROMPT = """You are an expert LLVM IR debugger and repair assistant.
You will be given:
1. The original C source code
2. LLVM IR that was generated but has errors
3. A list of specific validation errors

Your task: Fix the LLVM IR to be valid while preserving the original semantics.

Rules:
- Output ONLY the corrected LLVM IR, nothing else
- Fix all reported errors
- Maintain SSA form (each %name defined exactly once)
- Every basic block must have a terminator (ret, br, switch, unreachable)
- Use correct types consistently
- Preserve the function signature
- Use opaque pointers (ptr)"""

REPAIR_USER_TEMPLATE = """Original C code:
```c
{source_code}
```

Generated LLVM IR with errors:
```llvm
{generated_ir}
```

Validation errors found:
{error_list}

Please fix the LLVM IR to resolve all errors while preserving the semantics of the original C code.
Output only the corrected LLVM IR."""


# ============================================================================
# Repair result tracking
# ============================================================================

@dataclass
class RepairIteration:
    iteration: int
    ir_before: str
    ir_after: str
    errors_before: int
    errors_after: int
    errors_fixed: list = field(default_factory=list)
    errors_introduced: list = field(default_factory=list)
    repair_time_s: float = 0.0
    model_used: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class RepairResult:
    construct_id: str
    model_name: str
    original_ir: str
    final_ir: str
    source_code: str
    reference_ir: str
    iterations: list = field(default_factory=list)
    total_iterations: int = 0
    initial_errors: int = 0
    final_errors: int = 0
    repair_successful: bool = False
    total_repair_time_s: float = 0.0

    def to_dict(self):
        d = asdict(self)
        d['error_reduction'] = self.initial_errors - self.final_errors
        d['error_reduction_pct'] = (
            round((self.initial_errors - self.final_errors) / max(self.initial_errors, 1) * 100, 1)
        )
        return d


# ============================================================================
# Repair Loop
# ============================================================================

class RepairLoop:
    """Iterative LLM-based repair of invalid LLVM IR."""

    def __init__(self, hf_token=None, repair_model_id=None, provider="auto", max_iterations=3):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.repair_model_id = repair_model_id or "Qwen/Qwen2.5-Coder-32B-Instruct"
        self.provider = provider
        self.max_iterations = max_iterations
        self.client = InferenceClient(
            provider=self.provider,
            api_key=self.hf_token,
        )

    def repair(self, construct, generated_ir: str, verbose=True) -> RepairResult:
        """
        Attempt to repair generated IR through iterative validation-repair cycles.

        Architecture (from AIvril 2, arXiv:2412.04485):
          1. Validate IR
          2. If errors found, format them as feedback
          3. Ask LLM to fix
          4. Repeat until valid or max_iterations reached
        """
        from ir_generator import extract_ir_from_response

        result = RepairResult(
            construct_id=construct.id,
            model_name=self.repair_model_id,
            original_ir=generated_ir,
            final_ir=generated_ir,
            source_code=construct.source_code,
            reference_ir=construct.expected_ir,
        )

        current_ir = generated_ir

        # Initial validation
        report = validate_ir(current_ir)
        result.initial_errors = report.error_count

        if report.is_valid:
            result.repair_successful = True
            result.final_ir = current_ir
            if verbose:
                print(f"  [{construct.id}] Already valid, no repair needed")
            return result

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"  [{construct.id}] Repair iteration {iteration + 1}/{self.max_iterations} "
                      f"({report.error_count} errors)...")

            # Format error feedback
            error_list = self._format_errors(report)

            # Build repair prompt
            messages = [
                {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                {"role": "user", "content": REPAIR_USER_TEMPLATE.format(
                    source_code=construct.source_code,
                    generated_ir=current_ir,
                    error_list=error_list,
                )},
            ]

            start_time = time.time()
            try:
                response = self.client.chat.completions.create(
                    model=self.repair_model_id,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.1,
                )
                raw = response.choices[0].message.content
                repaired_ir, _ = extract_ir_from_response(raw)
                elapsed = time.time() - start_time
            except Exception as e:
                elapsed = time.time() - start_time
                if verbose:
                    print(f"    Repair API error: {e}")
                break

            # Validate repaired IR
            new_report = validate_ir(repaired_ir)

            iteration_result = RepairIteration(
                iteration=iteration + 1,
                ir_before=current_ir,
                ir_after=repaired_ir,
                errors_before=report.error_count,
                errors_after=new_report.error_count,
                repair_time_s=round(elapsed, 2),
                model_used=self.repair_model_id,
            )
            result.iterations.append(iteration_result)
            result.total_repair_time_s += elapsed

            if verbose:
                print(f"    Errors: {report.error_count} -> {new_report.error_count}")

            current_ir = repaired_ir
            report = new_report

            if report.is_valid:
                result.repair_successful = True
                break

        result.final_ir = current_ir
        result.final_errors = report.error_count
        result.total_iterations = len(result.iterations)

        if verbose:
            status = "SUCCESS" if result.repair_successful else "PARTIAL"
            print(f"  [{construct.id}] Repair {status}: "
                  f"{result.initial_errors} -> {result.final_errors} errors "
                  f"({result.total_repair_time_s:.1f}s)")

        return result

    def repair_batch(self, constructs_and_irs: list, verbose=True) -> list:
        """
        Repair multiple (construct, generated_ir) pairs.
        Args: list of (construct, generated_ir) tuples.
        """
        results = []
        for i, (construct, gen_ir) in enumerate(constructs_and_irs):
            if verbose:
                print(f"\nRepairing [{i+1}/{len(constructs_and_irs)}] {construct.id}...")
            result = self.repair(construct, gen_ir, verbose=verbose)
            results.append(result)
        return results

    def _format_errors(self, report) -> str:
        """Format validation errors as a numbered list for the LLM."""
        lines = []
        for i, error in enumerate(report.errors, 1):
            if error.severity.value == "error":
                loc = f" (line {error.line_number})" if error.line_number else ""
                content = f" | Code: {error.line_content}" if error.line_content else ""
                lines.append(f"{i}. [{error.category.value}] {error.message}{loc}{content}")
        return "\n".join(lines) if lines else "No specific errors (general validation failure)"

    @staticmethod
    def save_repair_results(results: list, filepath: str):
        """Save repair results to JSON."""
        data = [r.to_dict() for r in results]
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} repair results to {filepath}")


# ============================================================================
# Repair Statistics
# ============================================================================

def compute_repair_statistics(repair_results: list) -> dict:
    """Compute aggregate repair statistics."""
    total = len(repair_results)
    successful = sum(1 for r in repair_results if r.repair_successful)
    total_iters = sum(r.total_iterations for r in repair_results)
    total_time = sum(r.total_repair_time_s for r in repair_results)

    # Error reduction
    initial_errors = sum(r.initial_errors for r in repair_results)
    final_errors = sum(r.final_errors for r in repair_results)

    # Per-iteration statistics
    iter_improvements = []
    for r in repair_results:
        for it in r.iterations:
            improvement = it.errors_before - it.errors_after
            iter_improvements.append(improvement)

    stats = {
        "total_repairs_attempted": total,
        "successful_repairs": successful,
        "success_rate_pct": round(successful / max(total, 1) * 100, 1),
        "total_iterations": total_iters,
        "avg_iterations_per_repair": round(total_iters / max(total, 1), 1),
        "total_initial_errors": initial_errors,
        "total_final_errors": final_errors,
        "total_errors_fixed": initial_errors - final_errors,
        "error_reduction_pct": round(
            (initial_errors - final_errors) / max(initial_errors, 1) * 100, 1
        ),
        "total_repair_time_s": round(total_time, 1),
        "avg_repair_time_s": round(total_time / max(total, 1), 1),
        "avg_error_reduction_per_iteration": round(
            sum(iter_improvements) / max(len(iter_improvements), 1), 2
        ),
        "iterations_with_improvement": sum(1 for i in iter_improvements if i > 0),
        "iterations_with_regression": sum(1 for i in iter_improvements if i < 0),
        "iterations_with_no_change": sum(1 for i in iter_improvements if i == 0),
    }

    return stats
