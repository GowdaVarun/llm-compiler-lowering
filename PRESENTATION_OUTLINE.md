# Presentation Outline — AI-Assisted Compiler Lowering

## Slide 1: Title
**AI-Assisted Lowering from High-Level Code to Compiler IR**
- Can LLMs translate C → LLVM IR?
- Where do they fail? Can we fix it?

## Slide 2: Motivation & Research Questions
- Compiler lowering: C → LLVM IR (strict rules: SSA, types, CFG)
- LLMs excel at code generation — but can they handle **formal IR**?
- RQ: (a) Map constructs? (b) Preserve control flow? (c) Valid IR? (d) Repair loop?

## Slide 3: Approach
- **15 C constructs** across 6 complexity levels (L1=arithmetic → L6=composite)
- **4 LLMs**: Llama-3.3-70B, Qwen2.5-Coder-32B, Qwen2.5-72B, Llama-3.1-8B
- **2 strategies**: Basic prompting vs Chain-of-Thought
- **5-pass validator**: Syntax → SSA → Types → Control Flow → Semantics
- **Repair loop**: Validation errors → LLM feedback → re-generate (max 3 iters)
- **118 total generations** analyzed

## Slide 4: Results Overview
| | Valid | Compilable | Structurally Correct |
|-|-------|-----------|---------------------|
| Overall | **89.8%** | 93.2% | 80.5% |
| L1 (Arithmetic) | **100%** | 100% | 100% |
| L6 (Composite) | 75.0% | — | — |

## Slide 5: Model Comparison
```
Llama-3.3-70B:     ████████████████████ 100.0%
Qwen2.5-Coder-32B: ██████████████████░░ 93.3%
Llama-3.1-8B:      █████████████████░░░ 89.7%
Qwen2.5-72B:       ███████████████░░░░░ 75.9%
```
**Insight**: Model size ≠ IR quality. The 70B general model beats 72B and 32B coding model.

## Slide 6: Failure Taxonomy
5 classes, 25 sub-categories:
- **CLASS 1** — Structural/Syntactic (24.6%) — SSA violations, wrong predicates
- **CLASS 2** — Control Flow (66.7%) — **THE BOTTLENECK**: phi nodes, loop approx
- **CLASS 3** — Type System (0%) — Not observed in our study
- **CLASS 4** — Semantic/Functional (8.8%) — Hallucinated functions
- **CLASS 5** — Scale/Context (0%) — Within context limits

## Slide 7: Top Failure Modes (with examples)
1. **Incorrect phi node predecessors** (13 instances)
   - Model uses wrong block labels in `phi` instruction
2. **Extra code paths** (13 instances)
   - Model generates unnecessary basic blocks
3. **SSA violations** (12 instances)
   - Same `%name` defined twice (especially in alloca/load patterns)
4. **Loop approximation** (10 instances)
   - Model simplifies loop structure, loses control flow edges

### Example: Qwen2.5-Coder-32B on `max(a,b)`
```llvm
%cmp = icmp gt i32 %a, %b    ← ERROR: 'gt' not valid, should be 'sgt'
```

### Example: Qwen2.5-72B on `factorial(n)` — SSA violation
```llvm
loop_condition:
  %i_val = load i32, i32* %i      ← first definition
loop_body:
  %i_val = load i32, i32* %i      ← SECOND definition — SSA violation!
```

## Slide 8: Prompting Strategy Comparison
| Strategy | Valid Rate |
|----------|-----------|
| **Basic** | **95.0%** |
| Chain-of-Thought | 84.5% |

**Counterintuitive finding**: CoT makes models generate verbose -O0 style IR,
which introduces more SSA violations. For structured output, direct instruction wins.

## Slide 9: Repair Loop Architecture
```
C Source → LLM Generator → LLVM IR
                              ↓
                       5-Pass Validator
                         ↓         ↓
                     Valid?    No → Format Error Feedback
                       ↓              ↓
                    Accept       LLM Repair Agent
                                      ↓
                                 Re-validate
                                 (max 3 iters)
```

## Slide 10: Repair Loop Results
- **87.5% success rate** (7/8 failed constructs fully repaired)
- **90.9% error reduction** (11 → 1 total errors)
- **Average 1.2 iterations** — most repairs succeed first try
- Only Bubble Sort partially failed (2 → 1 errors after 3 iterations)

## Slide 11: Key Takeaways
1. ✅ **LLMs can do compiler lowering** — 89.8% overall validity
2. ⚠️ **Control flow is the bottleneck** — phi nodes, loops account for 67% of errors
3. 🔄 **Validation + repair is essential** — lifts success from 89.8% to ~99%
4. 💡 **Basic prompting > CoT** for structured output generation
5. 📏 **Model size ≠ quality** — instruction following matters more than parameters

## Slide 12: Future Directions
- Grammar-constrained decoding (force syntactically valid IR at generation time)
- Fine-tuning on paired C→IR data (ComPile dataset, 1.4T tokens)
- Extend to MLIR dialects (func, arith, memref) — uncharted territory
- Runtime equivalence checking (lli execution comparison)
- Integration with actual LLVM toolchain (llvm-as, opt --verify)

## Slide 13: References
1. LLM4IR (arXiv:2502.06854) — IR comprehension study
2. LLM Compiler Opt (arXiv:2309.07062) — ICLR 2024
3. AIvril 2 (arXiv:2412.04485) — Repair loop architecture
4. Compiler Feedback (arXiv:2403.14714) — Feedback loop
5. New Compiler Stack Survey (arXiv:2601.02045) — Jan 2026
