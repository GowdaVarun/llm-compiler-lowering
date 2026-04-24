# AI-Assisted Lowering from High-Level Code to Compiler IR

**Assignment 15 — Investigating LLM-assisted compiler lowering to LLVM IR**

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **Total Generations** | 118 (15 constructs × 4 models × 2 strategies) |
| **Overall Validity Rate** | 89.8% |
| **Best Model** | Llama-3.3-70B (100.0% valid) |
| **Repair Success Rate** | 87.5% (7/8 constructs fully repaired) |
| **Error Reduction via Repair** | 90.9% |

## 🏗️ Project Structure

```
├── source_constructs.py     # 15 C constructs across 6 complexity levels
├── ir_generator.py          # LLM-based LLVM IR generation pipeline
├── ir_validator.py          # 5-pass LLVM IR validator (SSA, types, CFG)
├── failure_analyzer.py      # 5-class failure taxonomy with 25 sub-categories
├── repair_loop.py           # Iterative validation-repair architecture
├── report_generator.py      # Automated report generation
├── run_pipeline_v2.py       # Main pipeline script
├── results/
│   ├── REPORT.md            # Full project report
│   ├── generation_results.json
│   ├── failure_analysis.json
│   ├── detailed_examples.json
│   ├── repair_results.json
│   └── repair_statistics.json
└── README.md
```

## 🔬 Source Language Subset

15 C constructs organized by complexity:

| Level | Category | Constructs | Validity Rate |
|-------|----------|-----------|---------------|
| L1 | Arithmetic & Assignment | 4 | **100.0%** |
| L2 | Control Flow | 4 | 84.4% |
| L3 | Functions & Calling | 2 | **100.0%** |
| L4 | Pointers & Memory | 2 | 86.7% |
| L5 | Structs & Aggregates | 1 | 87.5% |
| L6 | Composite | 2 | 75.0% |

## 🤖 Models Evaluated

| Model | Valid Rate | Compilable | Avg Time |
|-------|-----------|-----------|----------|
| **Llama-3.3-70B** | **100.0%** | 100.0% | 1.53s |
| Qwen2.5-Coder-32B | 93.3% | 100.0% | 3.67s |
| Llama-3.1-8B | 89.7% | 89.7% | 0.33s |
| Qwen2.5-72B | 75.9% | 82.8% | 22.8s |

## 🐛 Failure Taxonomy

| Class | Category | Count | % of Total |
|-------|----------|-------|-----------|
| CLASS_1 | Structural/Syntactic | 14 | 24.6% |
| CLASS_2 | **Control Flow** | **38** | **66.7%** |
| CLASS_4 | Semantic/Functional | 5 | 8.8% |

**Top failure sub-categories:**
1. Incorrect phi node predecessors (2.3): 13 instances
2. Unreachable/extra code paths (2.5): 13 instances  
3. SSA violations (1.1): 12 instances
4. Loop approximation (2.4): 10 instances

## 🔧 Repair Loop

Based on AIvril 2 (arXiv:2412.04485) architecture:
- **87.5% success rate** — 7 of 8 failed constructs fully repaired
- **90.9% error reduction** — 11 → 1 total errors
- **Average 1.2 iterations** — most repairs succeed in a single pass
- Only Bubble Sort (L6_01) could not be fully repaired

## 📝 Key Findings

1. **Arithmetic = solved** — 100% validity for all models on L1 constructs
2. **Control flow = frontier** — phi nodes and loop structures are the main challenge
3. **Model size ≠ quality** — Llama-3.3-70B (100%) > Qwen2.5-72B (75.9%)
4. **Basic > CoT** — Simple prompting (95.0%) outperforms chain-of-thought (84.5%)
5. **Repair loops are powerful** — Single-iteration repair fixes most errors

## 🚀 How to Run

```bash
# Install dependencies
pip install huggingface_hub openai

# Set your HF token
export HF_TOKEN=your_token_here

# Run the full pipeline
python run_pipeline_v2.py

# Generate report from existing results
python report_generator.py results/
```

## 📚 References

1. Jiang et al. (2025). "Can LLMs Understand IR?" arXiv:2502.06854
2. Cummins et al. (2023). "LLMs for Compiler Optimization." ICLR 2024. arXiv:2309.07062
3. AIvril 2 (2024). "EDA-Aware RTL Generation." arXiv:2412.04485
4. IRCoder (2024). arXiv:2403.03894
5. LLM4Decompile (2024). arXiv:2403.05286

## License

MIT
