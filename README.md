# AI-Assisted Lowering from High-Level Code to Compiler IR

**Assignment 15 — Investigating LLM-assisted compiler lowering to LLVM IR**

## ✅ What this repo does

This project evaluates whether LLMs can lower curated C constructs into valid LLVM IR, then
validates, analyzes, and repairs the outputs to produce measurable compiler-oriented results.

## 📊 Key Results

| Metric                         | Value                                         |
| ------------------------------ | --------------------------------------------- |
| **Total Generations**          | 118 (15 constructs × 4 models × 2 strategies) |
| **Overall Validity Rate**      | 89.8%                                         |
| **Best Model**                 | Llama-3.3-70B (100.0% valid)                  |
| **Repair Success Rate**        | 87.5% (7/8 constructs fully repaired)         |
| **Error Reduction via Repair** | 90.9%                                         |

## 🏗️ Project Structure

```
├── build.sh
├── build.bat
├── run.sh
├── run.bat
├── DESIGN
├── IMPLEMENTATION
├── EVALUATION
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
├── src/
├── testcases/
│   └── testcases.json
└── README.md
```

## 🔬 Source Language Subset

15 C constructs organized by complexity:

| Level | Category                | Constructs | Validity Rate |
| ----- | ----------------------- | ---------- | ------------- |
| L1    | Arithmetic & Assignment | 4          | **100.0%**    |
| L2    | Control Flow            | 4          | 84.4%         |
| L3    | Functions & Calling     | 2          | **100.0%**    |
| L4    | Pointers & Memory       | 2          | 86.7%         |
| L5    | Structs & Aggregates    | 1          | 87.5%         |
| L6    | Composite               | 2          | 75.0%         |

## 🛠️ Prerequisites & Installation

### 🪟 Windows Prerequisites
* **Python 3.10+**
* **For Hugging Face Pipeline:** A Hugging Face account and an API Token (`HF_TOKEN`).
* **For Local Ollama & LLVM Validation:**
  1. Install [Ollama for Windows](https://ollama.com).
  2. Install LLVM inside WSL (Ubuntu) so Windows PowerShell can call it:
     ```powershell
     wsl sudo apt-get update
     wsl sudo apt-get install -y llvm
     ```
  3. Verify the WSL LLVM installation from Windows PowerShell:
     ```powershell
     wsl llvm-as --version
     wsl opt --version
     ```

### 🍎🐧 macOS / Linux Prerequisites
* **Python 3.10+**
* **For Hugging Face Pipeline:** A Hugging Face account and an API Token (`HF_TOKEN`).
* **For Local Ollama & LLVM Validation:**
  1. Install Ollama (Linux: `curl -fsSL https://ollama.com/install.sh | sh` | macOS: download from [ollama.com](https://ollama.com)).
  2. Install LLVM tools:
     * **Ubuntu/Debian:** `sudo apt-get install -y llvm`
     * **macOS:** `brew install llvm` (and add it to your PATH)
  3. Verify the LLVM installation:
     ```bash
     llvm-as --version
     opt --version
     ```

---

## 🚀 How to Run the Pipeline

First, clone this repository and initialize the project:

### 🪟 Windows (PowerShell or Command Prompt)
Run the setup batch script to configure the virtual environment and install dependencies:
```powershell
build.bat
```

#### Option A: Run via Hugging Face (Cloud Models)
Set your token and run the main pipeline:
```powershell
set HF_TOKEN=your_token_here
run.bat
```

#### Option B: Run via Ollama (Local Models)
Ensure `ollama serve` is running, pull the model, and execute:
```powershell
ollama pull qwen2.5-coder:3b
run.bat ollama --model qwen2.5-coder:3b --levels 1
```

---

### 🍎🐧 macOS / Linux
Run the setup bash script to configure the virtual environment and install dependencies:
```bash
chmod +x build.sh run.sh
./build.sh
```

#### Option A: Run via Hugging Face (Cloud Models)
Set your token and run the main pipeline:
```bash
export HF_TOKEN=your_token_here
./run.sh
```

#### Option B: Run via Ollama (Local Models)
Ensure `ollama` is running, pull the model, and execute:
```bash
ollama pull qwen2.5-coder:3b
./run.sh ollama --model qwen2.5-coder:3b
```

Outputs are written to `results/` (Hugging Face pipeline) or `ollama_results/` (Ollama pipeline).

## 🧠 Design (approach + alternatives)

**Approach:** Curate a 15-construct C subset with ground-truth LLVM IR, run multi-model LLM
generation with prompt strategies, validate IR using a multi-pass LLVM-aware validator, analyze
failures with a taxonomy, and repair invalid outputs with a validator-in-the-loop architecture.

**Alternatives considered:** Rule-based lowering or a full compiler front-end as baseline (not the
target of this study), fine-tuning IR-specific models (out of scope), or using only LLVM toolchain
validation without a structured failure taxonomy (insufficient for analysis/repair feedback).

## ⚙️ Implementation (LLVM details)

The validator checks syntax, SSA form, type consistency, and control-flow correctness (terminators,
branch targets, phi-node predecessors). A CFG is built per function to validate block structure,
and errors are categorized for analysis. The Ollama pipeline optionally cross-validates IR using
`llvm-as` and `opt` for toolchain-level confirmation.

## 📈 Evaluation (metrics + comparison + test cases)

**Metrics**

| Metric                     | Value                   |
| -------------------------- | ----------------------- |
| Overall validity rate      | 89.8% (118 generations) |
| Repair success rate        | 87.5% (7/8)             |
| Error reduction via repair | 90.9% (11 → 1)          |
| Avg repair iterations      | 1.2                     |

**Baseline comparison**

| Baseline        | Total errors | Notes                               |
| --------------- | ------------ | ----------------------------------- |
| No repair       | 11           | Initial failures before repair loop |
| With repair     | 1            | Final errors after repair loop      |
| Error reduction | 90.9%        | 10 errors fixed                     |

**Test cases (all 15 constructs)** — stored in `testcases/testcases.json`, sourced from
`source_constructs.py`. Each case includes **IR expectations** (key LLVM IR features). Runtime
inputs exist where applicable; pointer/struct/composite constructs are **IR-only**.

| ID    | Function                                | Runtime inputs                       | Expected outputs |
| ----- | --------------------------------------- | ------------------------------------ | ---------------- |
| L1_01 | add(a, b)                               | (3,4), (0,0), (-1,1)                 | 7, 0, 0          |
| L1_02 | compute(x, y, z)                        | (2,3,4), (0,5,1)                     | 9, 0             |
| L1_03 | average(a, b)                           | (3.0,5.0), (0.0,0.0)                 | 4.0, 0.0         |
| L1_04 | bitops(a, b)                            | (5,3)                               | 7                |
| L2_01 | max(a, b)                               | (5,3), (2,7), (4,4)                  | 5, 7, 4          |
| L2_02 | sum_to_n(n)                             | (5), (0), (10)                       | 10, 0, 45        |
| L2_03 | factorial(n)                            | (5), (1), (0)                        | 120, 1, 1        |
| L2_04 | classify(x)                             | (150), (50), (-5), (0)               | 2, 1, -1, 0      |
| L3_01 | sum_of_squares(a, b)                    | (3,4)                               | 25               |
| L3_02 | fib(n)                                  | (0), (1), (5), (10)                  | 0, 1, 5, 55      |
| L4_01 | swap(int* a, int* b)                     | IR-only                              | IR-only          |
| L4_02 | array_sum(int* arr, int n)              | IR-only                              | IR-only          |
| L5_01 | manhattan_distance(struct Point* p1,p2) | IR-only                              | IR-only          |
| L6_01 | bubble_sort(int* arr, int n)            | IR-only                              | IR-only          |
| L6_02 | list_length(struct Node* head)          | IR-only                              | IR-only          |

## ✅ Four-Model Summary (Project Completeness)

| Model             | Valid Rate | Compilable | Avg Time |
| ----------------- | ---------- | ---------- | -------- |
| **Llama-3.3-70B** | **100.0%** | 100.0%     | 1.53s    |
| Qwen2.5-Coder-32B | 93.3%      | 100.0%     | 3.67s    |
| Llama-3.1-8B      | 89.7%      | 89.7%      | 0.33s    |
| Qwen2.5-72B       | 75.9%      | 82.8%      | 22.8s    |

## 🐛 Failure Taxonomy

| Class   | Category             | Count  | % of Total |
| ------- | -------------------- | ------ | ---------- |
| CLASS_1 | Structural/Syntactic | 14     | 24.6%      |
| CLASS_2 | **Control Flow**     | **38** | **66.7%**  |
| CLASS_4 | Semantic/Functional  | 5      | 8.8%       |

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

`source_constructs.py` is the construct registry itself (15 predefined constructs).
There is currently no separate generator script that auto-creates those constructs.

`run_pipeline_ollama.py` requires LLVM CLI tools in PATH (`llvm-as`, `opt`) and
saves outputs by default to `ollama_results/` (including LLVM tool validation JSON files).

## 🎥 Demo evidence (video or screenshots)

Capture these items:

1. Run `./run.sh` (or `run.bat`) and show `PIPELINE COMPLETE` plus the generated files in `results/`.
2. Show a working case from `results/REPORT.md` or `results/detailed_examples.json`.
3. Show a failure case and repair stats from `results/failure_analysis.json` and `results/repair_statistics.json`
   (e.g., Bubble Sort L6_01 is not fully repaired in the sample results).

## 📚 References

1. Jiang et al. (2025). "Can LLMs Understand IR?" arXiv:2502.06854
2. Cummins et al. (2023). "LLMs for Compiler Optimization." ICLR 2024. arXiv:2309.07062
3. AIvril 2 (2024). "EDA-Aware RTL Generation." arXiv:2412.04485
4. IRCoder (2024). arXiv:2403.03894
5. LLM4Decompile (2024). arXiv:2403.05286

## License

MIT
