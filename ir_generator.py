"""
LLM IR Generation Pipeline — Calls multiple LLMs via HF Inference API
to generate LLVM IR from high-level C constructs.

Uses InferenceClient with provider routing for model access.
"""

import os
import json
import time
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from llm_clients import chat_completion


# ============================================================================
# Prompt Templates (based on Expert Meta-Template from arXiv:2502.06854)
# ============================================================================

SYSTEM_PROMPT_BASIC = """You are an expert compiler engineer specializing in LLVM IR generation.
Given a C function, generate the equivalent LLVM IR.

Rules:
- Output ONLY valid LLVM IR text, nothing else
- Use SSA form: each variable (%name) is defined exactly once
- Every basic block must end with a terminator (ret, br, switch, unreachable)
- Use correct LLVM types (i32, i64, float, double, ptr, void)
- Use correct instruction syntax (add i32 %a, %b — not add %a, %b)
- Use phi nodes for values that merge from different control flow paths
- Use opaque pointers (ptr) not typed pointers (i32*)
- Do not include target datalayout or target triple
- Do not add comments or explanations"""

SYSTEM_PROMPT_COT = """You are an expert compiler engineer specializing in LLVM IR generation.
Given a C function, generate the equivalent LLVM IR.

Think step by step:
1. Identify the function signature (return type, parameter types)
2. Identify the control flow structure (branches, loops)
3. Map each variable to an SSA value
4. For loops, identify where phi nodes are needed
5. Ensure every basic block has exactly one terminator

Then output the LLVM IR enclosed in ```llvm ... ``` markers.

Rules:
- Use SSA form: each %name defined exactly once
- Every basic block ends with a terminator (ret, br, switch, unreachable)
- Use correct LLVM types and instruction syntax
- Use opaque pointers (ptr)
- Phi nodes must list all predecessor blocks"""

USER_PROMPT_TEMPLATE = """Convert this C code to LLVM IR:

```c
{source_code}
```"""

USER_PROMPT_TEMPLATE_WITH_HINTS = """Convert this C code to LLVM IR.

Key IR features to include: {hints}

```c
{source_code}
```"""


# ============================================================================
# Model configurations
# ============================================================================

@dataclass
class ModelConfig:
    model_id: str
    display_name: str
    provider: str = "auto"
    max_tokens: int = 4096
    temperature: float = 0.1
    supports_cot: bool = False
    base_url: Optional[str] = None


# Models ranked by IR generation capability (from literature)
MODELS = [
    ModelConfig(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        display_name="Qwen2.5-Coder-32B",
        provider="auto",
        supports_cot=False,
    ),
    ModelConfig(
        model_id="deepseek-ai/DeepSeek-R1",
        display_name="DeepSeek-R1",
        provider="auto",
        supports_cot=True,  # Best with CoT per arXiv:2502.06854
    ),
    ModelConfig(
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        display_name="Llama-3.3-70B",
        provider="auto",
        supports_cot=False,
    ),
    ModelConfig(
        model_id="mistralai/Mistral-Small-24B-Instruct-2501",
        display_name="Mistral-Small-24B",
        provider="auto",
        supports_cot=False,
    ),
]


# ============================================================================
# Generation result
# ============================================================================

@dataclass
class GenerationResult:
    construct_id: str
    model_name: str
    model_id: str
    prompt_strategy: str  # "basic" or "cot" or "with_hints"
    source_code: str
    generated_ir: str
    raw_response: str
    generation_time_s: float
    error: Optional[str] = None
    thinking: Optional[str] = None  # For CoT models

    def to_dict(self):
        return asdict(self)


# ============================================================================
# IR Extraction
# ============================================================================

def extract_ir_from_response(response_text: str) -> tuple:
    """
    Extract LLVM IR from LLM response, handling various formatting.
    Returns (ir_text, thinking_text).
    """
    thinking = None
    text = response_text

    # Extract thinking/reasoning if present (DeepSeek-R1)
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        text = text[think_match.end():].strip()

    # Try to extract from code blocks
    # Try ```llvm ... ```
    llvm_match = re.search(r'```(?:llvm|ll|llvm-ir)\s*\n(.*?)```', text, re.DOTALL)
    if llvm_match:
        return llvm_match.group(1).strip(), thinking

    # Try ``` ... ``` (generic code block)
    code_match = re.search(r'```\s*\n(.*?)```', text, re.DOTALL)
    if code_match:
        candidate = code_match.group(1).strip()
        # Verify it looks like IR (has 'define' or starts with %)
        if 'define ' in candidate or candidate.startswith('%') or 'declare ' in candidate:
            return candidate, thinking

    # Try to find IR directly (no code fences)
    # Look for 'define' keyword
    define_match = re.search(r'((?:%[^\n]*= type[^\n]*\n)*\s*define\s+.+?\n(?:.*?\n)*?})', text, re.DOTALL)
    if define_match:
        return define_match.group(1).strip(), thinking

    # Last resort: return everything after removing obvious non-IR text
    lines = text.split('\n')
    ir_lines = []
    in_ir = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('define ') or stripped.startswith('declare ') or stripped.startswith('%'):
            in_ir = True
        if in_ir:
            if stripped and not stripped.startswith('//') and not stripped.startswith('#'):
                ir_lines.append(line)
            if stripped == '}':
                # Could be end of function, keep collecting if more define follows
                pass

    if ir_lines:
        return '\n'.join(ir_lines), thinking

    return text, thinking


# ============================================================================
# Generation Pipeline
# ============================================================================

class IRGenerationPipeline:
    """Generates LLVM IR using multiple LLMs and prompt strategies."""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        models: Optional[list] = None,
        ollama_base_url: Optional[str] = None,
    ):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.models = models or MODELS
        self.results = []
        self.ollama_base_url = ollama_base_url or os.environ.get("OLLAMA_BASE_URL")

    def generate_single(self, construct, model_config: ModelConfig,
                        prompt_strategy: str = "basic") -> GenerationResult:
        """Generate IR for a single construct with a single model."""
        # Build prompt
        if prompt_strategy == "cot":
            system = SYSTEM_PROMPT_COT
        else:
            system = SYSTEM_PROMPT_BASIC

        if prompt_strategy == "with_hints":
            user = USER_PROMPT_TEMPLATE_WITH_HINTS.format(
                source_code=construct.source_code,
                hints=", ".join(construct.key_ir_features),
            )
        else:
            user = USER_PROMPT_TEMPLATE.format(source_code=construct.source_code)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        start_time = time.time()
        try:
            raw = chat_completion(
                model_id=model_config.model_id,
                messages=messages,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                provider=model_config.provider,
                hf_token=self.hf_token,
                base_url=model_config.base_url or self.ollama_base_url,
            )
            elapsed = time.time() - start_time

            ir_text, thinking = extract_ir_from_response(raw)

            result = GenerationResult(
                construct_id=construct.id,
                model_name=model_config.display_name,
                model_id=model_config.model_id,
                prompt_strategy=prompt_strategy,
                source_code=construct.source_code,
                generated_ir=ir_text,
                raw_response=raw,
                generation_time_s=round(elapsed, 2),
                thinking=thinking,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            result = GenerationResult(
                construct_id=construct.id,
                model_name=model_config.display_name,
                model_id=model_config.model_id,
                prompt_strategy=prompt_strategy,
                source_code=construct.source_code,
                generated_ir="",
                raw_response="",
                generation_time_s=round(elapsed, 2),
                error=str(e),
            )

        self.results.append(result)
        return result

    def generate_all(self, constructs, prompt_strategies=None,
                     model_indices=None, verbose=True):
        """Generate IR for all constructs with all models and strategies."""
        if prompt_strategies is None:
            prompt_strategies = ["basic"]
        if model_indices is None:
            model_indices = range(len(self.models))

        total = len(constructs) * len(list(model_indices)) * len(prompt_strategies)
        count = 0

        for construct in constructs:
            for mi in model_indices:
                model = self.models[mi]
                for strategy in prompt_strategies:
                    count += 1
                    if verbose:
                        print(f"[{count}/{total}] {construct.id} | "
                              f"{model.display_name} | {strategy}...", end=" ", flush=True)
                    result = self.generate_single(construct, model, strategy)
                    if verbose:
                        status = "OK" if not result.error else f"ERROR: {result.error[:60]}"
                        print(f"{status} ({result.generation_time_s}s)")

        return self.results

    def save_results(self, filepath: str):
        """Save all results to JSON."""
        data = [r.to_dict() for r in self.results]
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} results to {filepath}")

    @staticmethod
    def load_results(filepath: str) -> list:
        """Load results from JSON."""
        with open(filepath) as f:
            data = json.load(f)
        return [GenerationResult(**d) for d in data]
