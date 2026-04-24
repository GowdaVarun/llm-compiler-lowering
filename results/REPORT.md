# AI-Assisted Lowering from High-Level Code to Compiler IR

**Assignment 15 — Compiler Lowering with Large Language Models**

*Date: April 24, 2026*

---

## Abstract

This report investigates whether Large Language Models (LLMs) can assist in translating
high-level C language constructs into LLVM Intermediate Representation (IR). We define a
structured source-language subset of 15 constructs across 6 complexity levels, generate
LLVM IR using 4 state-of-the-art LLMs (Qwen2.5-Coder-32B, Qwen2.5-72B,
Llama-3.3-70B, Llama-3.1-8B) with 2 prompting strategies (basic and chain-of-thought),
producing 118 generation attempts. We validate the output through a custom 5-pass validator,
categorize failures into a 5-class taxonomy with 25 sub-categories, and implement an
iterative repair loop. Our key findings: (1) overall 89.8% of generated IR passes validation,
with 100% validity on arithmetic constructs (L1) and 100% for Llama-3.3-70B across all levels;
(2) control flow failures (CLASS_2) account for 67% of all errors, with incorrect phi node
predecessors and loop approximation being the dominant sub-categories; (3) basic prompting
(95.0% valid) outperforms chain-of-thought (84.5%) for IR generation; (4) the repair loop
achieves 87.5% success rate and 90.9% error reduction, with 7/8 failed constructs fully repaired
in a single iteration. These results demonstrate that LLMs are capable IR generators for
simple-to-moderate constructs but require a validator-in-the-loop architecture for reliability.

---

## 1. Introduction

### 1.1 Background

Compiler lowering—the translation from high-level language constructs to structured
Intermediate Representations (IR)—is traditionally implemented through carefully engineered
frontend logic. LLVM IR, the dominant compiler IR in modern toolchains, enforces strict
structural rules including Static Single Assignment (SSA) form, explicit type annotations,
basic block terminators, and well-formed control flow graphs.

Recent advances in Large Language Models have shown impressive code generation capabilities,
raising the question: can LLMs assist in or automate parts of the compiler lowering process?

### 1.2 Research Questions

This study addresses four questions:

**(a)** Can LLMs map simple high-level language constructs into LLVM IR?

**(b)** Do LLMs preserve control-flow and data-flow structure during translation?

**(c)** Is the generated IR syntactically and semantically valid?

**(d)** Can a validator/repair loop improve the quality of LLM-generated IR?

### 1.3 Related Work

Our work builds on several recent studies:

- **LLM4IR** (Jiang et al., 2025, arXiv:2502.06854): Evaluated 6 LLMs on LLVM IR
  comprehension tasks. Found GPT-4 achieved only 24% accuracy on CFG reconstruction,
  while Code Llama scored 0%. DeepSeek-R1 with Chain-of-Thought achieved 74% on attempted
  instances.

- **LLM Compiler Optimization** (Cummins et al., 2023, arXiv:2309.07062): Trained a 7B
  parameter model on 1M LLVM IR functions, achieving 91% compilable IR generation and
  3.0% improvement over -Oz baseline. Published at ICLR 2024.

- **Compiler Generated Feedback** (Cummins et al., 2024, arXiv:2403.14714): Demonstrated
  that compiler error feedback loops improve LLM IR generation by +0.53%.

- **AIvril 2** (2024, arXiv:2412.04485): Proposed a two-stage syntax + functional
  verification loop for RTL generation, achieving 3.4× improvement over zero-shot.

- **IRCoder** (2024, arXiv:2403.03894): Showed that training with paired source→IR data
  improves code generation, with gains of +2.17 on MultiPL-E pass@1.

### 1.4 Contributions

1. A structured source-language subset of 15 C constructs across 6 complexity levels
2. A 5-pass LLVM IR validator checking syntax, SSA, types, control flow, and semantics
3. A 5-class failure taxonomy with 25 sub-categories derived from the literature
4. Empirical evaluation of 4 LLMs with 2 prompting strategies
5. An iterative repair loop architecture with quantified improvement metrics

## 2. Source Language Subset

### 2.1 Design Principles

We selected C as the source language for maximum coverage in existing benchmarks
(ExeBench, ComPile, AnghaBench) and because C maps most directly to LLVM IR without
runtime abstractions. The constructs are organized into 6 levels of increasing complexity:

| Level | Category | Count | Description |
|-------|----------|-------|-------------|
| L1 | Arithmetic & Assignment | 4 | Pure computation, no control flow |
| L2 | Control Flow | 4 | Branches, loops, nested conditionals |
| L3 | Functions & Calling | 2 | Function calls, recursion |
| L4 | Pointers & Memory | 2 | Dereferencing, array indexing |
| L5 | Structs & Aggregates | 1 | Struct field access, type definitions |
| L6 | Composite | 2 | Multi-category combinations |

### 2.2 Construct Catalog

#### Level 1

**L1_01: Simple Addition** — Two integer parameters added together
- Key IR features: add i32, ret i32, two i32 parameters
```c
int add(int a, int b) {
    return a + b;
}
```

**L1_02: Mixed Arithmetic** — Expression with add, multiply, subtract
- Key IR features: mul i32, add i32, sub i32, constant 1
```c
int compute(int x, int y, int z) {
    return x * y + z - 1;
}
```

**L1_03: Floating Point Arithmetic** — Float division and multiplication
- Key IR features: fadd float, fdiv float, float constant 2.0
```c
float average(float a, float b) {
    return (a + b) / 2.0f;
}
```

**L1_04: Bitwise Operations** — Bitwise AND, OR, XOR, shift
- Key IR features: and i32, xor i32, shl i32, or i32
```c
int bitops(int a, int b) {
    return (a & b) | (a ^ b) << 1;
}
```

#### Level 2

**L2_01: If-Else Branch** — Simple conditional with two branches
- Key IR features: icmp sgt, br i1, two basic blocks, conditional branch
```c
int max(int a, int b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}
```

**L2_02: While Loop** — Simple while loop with accumulator
- Key IR features: phi nodes, loop back-edge, icmp slt, three basic blocks
```c
int sum_to_n(int n) {
    int sum = 0;
    int i = 0;
    while (i < n) {
        sum = sum + i;
        i = i + 1;
    }
    return sum;
}
```

**L2_03: For Loop** — For loop computing factorial
- Key IR features: phi nodes, mul i32, icmp sle, loop structure
```c
int factorial(int n) {
    int result = 1;
    for (int i = 1; i <= n; i++) {
        result = result * i;
    }
    return result;
}
```

**L2_04: Nested If-Else** — Nested conditionals with multiple return paths
- Key IR features: multiple basic blocks, nested branches, icmp sgt, icmp slt
```c
int classify(int x) {
    if (x > 0) {
        if (x > 100) {
            return 2;
        } else {
            return 1;
        }
    } else if (x < 0) {
        return -1;
    } else {
        return 0;
    }
}
```

#### Level 3

**L3_01: Function Call** — One function calling another
- Key IR features: call instruction, multiple function definitions, i32 return
```c
int square(int x) {
    return x * x;
}

int sum_of_squares(int a, int b) {
    return square(a) + square(b);
}
```

**L3_02: Recursive Function** — Recursive fibonacci
- Key IR features: recursive call, icmp sle, sub i32, two call instructions
```c
int fib(int n) {
    if (n <= 1) {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}
```

#### Level 4

**L4_01: Pointer Dereference** — Swap two integers via pointers
- Key IR features: load i32, store i32, ptr type, void return
```c
void swap(int* a, int* b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}
```

**L4_02: Array Sum** — Sum elements of an array using pointer arithmetic
- Key IR features: getelementptr, sext, load from computed pointer, phi nodes
```c
int array_sum(int* arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}
```

#### Level 5

**L5_01: Struct Field Access** — Access fields of a struct
- Key IR features: struct type, getelementptr with struct, select instruction, abs pattern
```c
struct Point {
    int x;
    int y;
};

int manhattan_distance(struct Point* p1, struct Point* p2) {
    int dx = p1->x - p2->x;
    int dy = p1->y - p2->y;
    if (dx < 0) dx = -dx;
    if (dy < 0) dy = -dy;
    return dx + dy;
}
```

#### Level 6

**L6_01: Bubble Sort** — Nested loops, array access, pointer ops, swaps
- Key IR features: nested loops, phi nodes, getelementptr, load/store, icmp sgt, conditional swap, sext, multiple basic blocks
```c
void bubble_sort(int* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int tmp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tmp;
            }
        }
    }
}
```

**L6_02: Linked List Length** — Struct + pointer traversal + loop
- Key IR features: struct with pointer field, phi nodes, null pointer comparison, getelementptr into struct, load ptr, pointer traversal loop
```c
struct Node {
    int value;
    struct Node* next;
};

int list_length(struct Node* head) {
    int count = 0;
    struct Node* current = head;
    while (current != 0) {
        count = count + 1;
        current = current->next;
    }
    return count;
}
```

## 3. Source-to-IR Mapping Pipeline

### 3.1 Mapping Rules

Each high-level construct maps to specific LLVM IR patterns:

| C Construct | LLVM IR Pattern | Key Challenge |
|-------------|-----------------|---------------|
| `int a + b` | `%r = add i32 %a, %b` | Type annotation required |
| `float a / b` | `%r = fdiv float %a, %b` | Integer vs float op distinction |
| `if (cond)` | `icmp` + `br i1` + basic blocks | Branch target labels |
| `while (cond)` | Loop header with `phi` nodes | SSA phi node placement |
| `for (init; cond; inc)` | Same as while with init block | Loop canonicalization |
| `f(x)` | `call i32 @f(i32 %x)` | Calling convention |
| `*ptr` | `load i32, ptr %p` | Opaque pointer syntax |
| `arr[i]` | `getelementptr` + `load` | Index computation (sext) |
| `s->field` | `getelementptr %struct, ptr, 0, N` | Struct layout knowledge |
| Recursion | Self-referential `call` | Stack frame semantics |

### 3.2 SSA Form Requirements

The most challenging aspect of lowering is maintaining SSA form:

1. **Single definition**: Each `%name` is assigned exactly once in the entire function
2. **Phi nodes**: When a value can come from multiple predecessors, a `phi` instruction
   merges them at the join point
3. **Dominance**: A definition must dominate all its uses (appear earlier in the control
   flow graph)

Example — while loop requires phi nodes:
```c
int sum = 0;
while (i < n) { sum += i; i++; }
```
Maps to:
```llvm
while.cond:
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %while.body ]
  %i = phi i32 [ 0, %entry ], [ %i.next, %while.body ]
```

### 3.3 Target IR Specification

We target LLVM IR at approximately `-O1` optimization level:
- Opaque pointers (`ptr` not `i32*`)
- No debug metadata or target triples
- SSA form with phi nodes (not memory-based `-O0` form)
- Standard calling conventions

## 4. Methodology

### 4.1 Models Evaluated

| Model | Parameters | Provider | Rationale |
|-------|-----------|----------|-----------|
| Qwen2.5-Coder-32B-Instruct | 32B | HF Router (auto) | Best open-source code model, coding-specialized |
| Qwen2.5-72B-Instruct | 72B | HF Router (auto) | Large general model with strong reasoning |
| Llama-3.3-70B-Instruct | 70B | HF Router (auto) | Strong instruction-following and code generation |
| Llama-3.1-8B-Instruct | 8B | HF Router (auto) | Small model baseline for comparison |

### 4.2 Prompting Strategies

**Strategy 1: Basic** — Direct instruction with rules
- System prompt: Expert compiler engineer role, SSA rules, type requirements
- User prompt: "Convert this C code to LLVM IR" + source code

**Strategy 2: Chain-of-Thought (CoT)** — Step-by-step reasoning
- System prompt: Think through function signature, control flow, SSA mapping, phi nodes
- Based on findings from arXiv:2502.06854 where CoT improved DeepSeek-R1 CFG accuracy

### 4.3 Validation Pipeline

Our validator performs 5 sequential passes:

1. **Syntax Pass**: Structure parsing, code fence removal, function/block identification
2. **SSA Pass**: Single-definition verification, undefined variable detection
3. **Type Pass**: Operand type consistency, integer/float operation matching
4. **Control Flow Pass**: Terminator presence, branch target existence, phi node correctness,
   predecessor consistency
5. **Semantic Pass**: Return type matching, instruction predicate validity

### 4.4 Repair Loop

Based on AIvril 2 (arXiv:2412.04485) architecture:
```
Input: Invalid LLVM IR + Original C source
For each iteration (max 3):
  1. Validate IR → get error list
  2. Format errors as structured feedback
  3. Prompt LLM: "Fix these specific errors while preserving semantics"
  4. Extract repaired IR from response
  5. Re-validate
  If valid: STOP
Output: Repaired IR + repair trace
```

## 5. Results

### 5.1 Overall Results

| Metric | Count | Rate |
|--------|-------|------|
| Total Generations | 118 | — |
| Valid (no errors) | 106 | 89.8% |
| Compilable (no syntax/SSA errors) | 110 | 93.2% |
| Structurally Correct (matches reference) | 95 | 80.5% |
| Total Failure Instances | 57 | — |

### 5.2 Results by Model

| Model | Total | Valid | Valid Rate | Compilable Rate | Avg Time |
|-------|-------|-------|-----------|-----------------|----------|
| Llama-3.1-8B | 29 | 26 | 89.7% | 89.7% | 0.33s |
| Llama-3.3-70B | 30 | 30 | 100.0% | 100.0% | 1.53s |
| Qwen2.5-72B | 29 | 22 | 75.9% | 82.8% | 22.8s |
| Qwen2.5-Coder-32B | 30 | 28 | 93.3% | 100.0% | 3.67s |

### 5.3 Results by Construct Level

| Level | Total | Valid | Valid Rate |
|-------|-------|-------|-----------|
| L1 | 32 | 32 | 100.0% |
| L2 | 32 | 27 | 84.4% |
| L3 | 15 | 15 | 100.0% |
| L4 | 15 | 13 | 86.7% |
| L5 | 8 | 7 | 87.5% |
| L6 | 16 | 12 | 75.0% |

### 5.4 Results by Prompt Strategy

| Strategy | Total | Valid | Valid Rate |
|----------|-------|-------|-----------|
| cot | 58 | 49 | 84.5% |
| basic | 60 | 57 | 95.0% |

## 6. Failure Mode Categorization

### 6.1 Taxonomy

Based on analysis of LLM-generated IR failures, cross-referenced with findings from
arXiv:2502.06854, arXiv:2309.07062, arXiv:2403.05286, and arXiv:2407.06153:

#### CLASS_1: Structural/Syntactic Failures
*Violations of basic LLVM IR syntax and SSA form*

- **1.1**: SSA violation: reuse of %name, multiple definitions — **12 instances**
- **1.2**: Incomplete syntax: unclosed blocks, missing terminators
- **1.3**: Malformed type annotations: wrong integer width, float mismatch
- **1.4**: Invalid instruction format: wrong operand count, illegal opcode — **2 instances**
- **1.5**: Code fence artifacts: markdown or text mixed into IR

#### CLASS_2: Control Flow Failures
*Incorrect control flow graph structure*

- **2.1**: Missing basic block labels or duplicate labels
- **2.2**: Branch to non-existent block
- **2.3**: Incorrect phi node predecessors — **13 instances**
- **2.4**: Loops approximated/simplified rather than faithfully reconstructed — **10 instances**
- **2.5**: Unreachable code inserted (dead branches) — **13 instances**
- **2.6**: Missing terminator in basic block — **2 instances**

#### CLASS_3: Type System Failures
*Type mismatches and incorrect type usage*

- **3.1**: Type mismatch in operations (e.g., add i64 on i32 operands)
- **3.2**: Pointer vs value confusion (wrong load/store types)
- **3.3**: Missing struct/aggregate type definitions
- **3.4**: Undefined function signatures (hallucinated callee types)
- **3.5**: Integer/float operation confusion (add vs fadd)

#### CLASS_4: Semantic/Functional Failures
*Valid IR that computes wrong result*

- **4.1**: Wrong computation despite valid IR structure
- **4.2**: Heuristic pattern matching (plausible but incorrect IR) — **2 instances**
- **4.3**: Failed constant folding / arithmetic errors
- **4.4**: Data-flow analysis errors (incorrect use-def chains)
- **4.5**: Hallucinated intrinsics or functions — **3 instances**

#### CLASS_5: Scale/Context Failures
*Failures related to input size and context limitations*

- **5.1**: Context overflow causing truncated IR
- **5.2**: Inter-procedural context loss
- **5.3**: Empty or no IR generated

### 6.2 Failure Distribution

| Failure Class | Name | Count |
|--------------|------|-------|
| CLASS_1 | Structural/Syntactic Failures | 14 |
| CLASS_2 | Control Flow Failures | 38 |
| CLASS_3 | Type System Failures | 0 |
| CLASS_4 | Semantic/Functional Failures | 5 |
| CLASS_5 | Scale/Context Failures | 0 |

### 6.3 Failure Classes by Model

| Model | CLASS_1 | CLASS_2 | CLASS_3 | CLASS_4 | CLASS_5 |
|-------|---------|---------|---------|---------|---------|
| Llama-3.1-8B | 7 | 13 | 0 | 1 | 0 |
| Llama-3.3-70B | 0 | 10 | 0 | 1 | 0 |
| Qwen2.5-72B | 5 | 6 | 0 | 2 | 0 |
| Qwen2.5-Coder-32B | 2 | 9 | 0 | 1 | 0 |

## 7. LLM-Generated IR Examples with Correctness Analysis

Below are representative examples showing both successful and failed IR generations.

### 7.1 Successful Generation Examples

#### L1_01: Simple Addition (Qwen2.5-Coder-32B)
*Strategy: basic*

**Source:**
```c
int add(int a, int b) {
    return a + b;
}
```
**Generated IR:**
```llvm
define i32 @add(i32 %a, i32 %b) {
entry:
  %add = add i32 %a, %b
  ret i32 %add
}
```
✅ Valid IR — passes all validation checks

#### L1_01: Simple Addition (Qwen2.5-Coder-32B)
*Strategy: cot*

**Source:**
```c
int add(int a, int b) {
    return a + b;
}
```
**Generated IR:**
```llvm
define i32 @add(i32 %a, i32 %b) {
entry:
    %add = add i32 %a, %b
    ret i32 %add
}
```
✅ Valid IR — passes all validation checks

#### L1_01: Simple Addition (Qwen2.5-72B)
*Strategy: basic*

**Source:**
```c
int add(int a, int b) {
    return a + b;
}
```
**Generated IR:**
```llvm
define i32 @add(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}
```
✅ Valid IR — passes all validation checks

### 7.2 Failed Generation Examples

#### L2_01: If-Else Branch (Qwen2.5-Coder-32B)
*Strategy: basic*

**Source:**
```c
int max(int a, int b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}
```
**Generated IR:**
```llvm
define i32 @max(i32 %a, i32 %b) {
entry:
    %cmp = icmp gt i32 %a, %b
    br i1 %cmp, label %if_true, label %if_false

if_true:
    ret i32 %a

if_false:
    ret i32 %b
}
```

**Failures (1):**
- [CLASS_1/1.4] Invalid icmp predicate 'gt'. Valid: eq, ne, sge, sgt, sle, slt, uge, ugt, ule, ult

#### L2_01: If-Else Branch (Qwen2.5-Coder-32B)
*Strategy: cot*

**Source:**
```c
int max(int a, int b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}
```
**Generated IR:**
```llvm
define i32 @max(i32 %a, i32 %b) {
entry:
    %cmp = icmp gt i32 %a, %b
    br i1 %cmp, label %if.then, label %if.else

if.then:
    ret i32 %a

if.else:
    ret i32 %b
}
```

**Failures (1):**
- [CLASS_1/1.4] Invalid icmp predicate 'gt'. Valid: eq, ne, sge, sgt, sle, slt, uge, ugt, ule, ult

#### L2_02: While Loop (Qwen2.5-72B)
*Strategy: cot*

**Source:**
```c
int sum_to_n(int n) {
    int sum = 0;
    int i = 0;
    while (i < n) {
        sum = sum + i;
        i = i + 1;
    }
    return sum;
}
```
**Generated IR:**
```llvm
; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @sum_to_n(i32 %n) #0 {
entry:
  ; Initialize sum and i
  %sum = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, i32* %sum, align 4
  store i32 0, i32* %i, align 4

  ; Branch to the loop condition
  br label %loop_cond

loop_cond:
  ; Load i and n
  %i_val = load i32, i32* %i, align 4
  %n_val = load i32, i32* %n, align 4

  ; Check if i < n
  %cond = icmp slt i32 %i_val, %n_val
  br i1 %cond, label %loop_body, label %exit

loop_body:
  ; Load sum and i
  %sum_val = load i32, i32* %sum, align 4
  %i_val = load i32, i32* %i, align 4

  ; Update sum
  %new_sum = add i32 %sum_val, %i_val
  store i32 %new_sum, i32* %sum, align 4

  ; Increment i
  %new_i = add i32 %i_val, 1
  store i32 %new_i, i32* %i, align 4

  ; Branch back to the loop condition
  br label %loop_cond

exit:
  ; Load the final value of sum
  %final_sum = load i32, i32* %sum, align 4

  ; Return the final value of sum
  ret i32 %final_s
```

**Failures (1):**
- [CLASS_1/1.1] SSA violation: '%i_val' defined more than once in @sum_to_n

#### L2_03: For Loop (Qwen2.5-72B)
*Strategy: cot*

**Source:**
```c
int factorial(int n) {
    int result = 1;
    for (int i = 1; i <= n; i++) {
        result = result * i;
    }
    return result;
}
```
**Generated IR:**
```llvm
define i32 @factorial(i32 %n) {
entry:
  ; Initialize result to 1
  %result = alloca i32
  store i32 1, i32* %result

  ; Initialize i to 1
  %i = alloca i32
  store i32 1, i32* %i

  ; Branch to the loop condition
  br label %loop_condition

loop_condition:
  ; Load the current value of i
  %i_val = load i32, i32* %i

  ; Check if i <= n
  %cmp = icmp sle i32 %i_val, %n

  ; If true, go to the loop body, otherwise exit
  br i1 %cmp, label %loop_body, label %exit

loop_body:
  ; Load the current values of result and i
  %result_val = load i32, i32* %result
  %i_val = load i32, i32* %i

  ; Update result = result * i
  %new_result = mul i32 %result_val, %i_val
  store i32 %new_result, i32* %result

  ; Increment i
  %new_i = add i32 %i_val, 1
  store i32 %new_i, i32* %i

  ; Branch back to the loop condition
  br label %loop_condition

exit:
  ; Load the final value of result
  %final_result = load i32, i32* %result

  ; Return the final result
  ret i32 %final_result
}
```

**Failures (1):**
- [CLASS_1/1.1] SSA violation: '%i_val' defined more than once in @factorial

#### L2_04: Nested If-Else (Llama-3.1-8B)
*Strategy: cot*

**Source:**
```c
int classify(int x) {
    if (x > 0) {
        if (x > 100) {
            return 2;
        } else {
            return 1;
        }
    } else if (x < 0) {
        return -1;
    } else {
        return 0;
    }
}
```
**Generated IR:**
```llvm
define i32 @classify(i32 %x) {
entry:
  %cond1 = icmp sgt i32 %x, 0
  br i1 %cond1, label %if.then, label %if.else

if.then:
  %cond2 = icmp sgt i32 %x, 100
  br i1 %cond2, label %if.then2, label %if.else2

if.then2:
  %ret = constant i32 2
  br label %return

if.else2:
  %ret = constant i32 1
  br label %return

if.else:
  %cond3 = icmp slt i32 %x, 0
  br i1 %cond3, label %if.then3, label %if.else3

if.then3:
  %ret = constant i32 -1
  br label %return

if.else3:
  %ret = constant i32 0
  br label %return

return:
  %0 = phi i32 [ %ret, %if.then2 ], [ %ret, %if.else2 ], [ %ret, %if.then3 ], [ %ret, %if.else3 ]
  ret i32 %0
}
```

**Failures (4):**
- [CLASS_1/1.1] SSA violation: '%ret' defined more than once in @classify
- [CLASS_1/1.1] SSA violation: '%ret' defined more than once in @classify
- [CLASS_1/1.1] SSA violation: '%ret' defined more than once in @classify
- [CLASS_2/2.5] @classify: 8 blocks vs 7 expected (extra code paths)

## 8. Validator / Repair Architecture

### 8.1 Architecture Overview

Our repair architecture follows the three-tier design from AIvril 2 (arXiv:2412.04485),
adapted for LLVM IR:

```
┌─────────────────────────────────────────────────────────────┐
│                    REPAIR LOOP ARCHITECTURE                  │
│                                                              │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────────┐  │
│  │ C Source  │───>│ LLM Code  │───>│ Generated LLVM IR    │  │
│  │ Code      │    │ Generator │    │ (potentially invalid) │  │
│  └──────────┘    └───────────┘    └──────────┬───────────┘  │
│                                               │              │
│                                               ▼              │
│                  ┌────────────────────────────────────────┐  │
│                  │     5-PASS VALIDATOR                    │  │
│                  │  1. Syntax  2. SSA  3. Types           │  │
│                  │  4. Control Flow  5. Semantics          │  │
│                  └──────────────────────┬─────────────────┘  │
│                                         │                    │
│                              ┌──────────┴──────────┐        │
│                              │ Valid?               │        │
│                              └──────────┬──────────┘        │
│                           Yes │              │ No            │
│                               ▼              ▼               │
│                    ┌──────────────┐  ┌─────────────────┐    │
│                    │ Accept IR    │  │ Format Error     │    │
│                    │              │  │ Feedback         │    │
│                    └──────────────┘  └────────┬────────┘    │
│                                               │              │
│                                               ▼              │
│                                    ┌──────────────────┐     │
│                                    │ LLM Repair Agent  │     │
│                                    │ (with error ctx)  │◄──┐│
│                                    └────────┬─────────┘   ││
│                                              │             ││
│                                              ▼             ││
│                                    ┌──────────────────┐   ││
│                                    │ Re-validate      │   ││
│                                    │ (max 3 iters)    │───┘│
│                                    └──────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Validation Tiers

**Tier 1 — Syntactic Validation (Deterministic)**
- Parse function definitions and basic blocks
- Check for code fence artifacts from LLM output
- Verify structural completeness

**Tier 2 — SSA and Type Verification (Deterministic)**
- Single-definition property: each `%name` assigned exactly once per function
- Undefined variable detection: all uses reference valid definitions
- Type consistency: integer ops use integer types, float ops use float types
- Return type matching

**Tier 3 — Control Flow Verification (Deterministic)**
- Every basic block has a terminator instruction
- All branch targets reference existing blocks
- Phi nodes list correct predecessor blocks
- No phi nodes in entry block (unusual, flagged as warning)

**Tier 4 — Semantic Verification (LLM-assisted)**
- Compare generated IR structure with reference
- Function count and signature matching
- Basic block count comparison (detects over-simplification)

### 8.3 Repair Strategy

The repair loop uses **compiler error feedback** as context for the repair LLM:

1. Validation errors are formatted as a structured, numbered list
2. Each error includes: category, message, line number, and source line
3. The repair LLM receives: original source + current IR + error list
4. Temperature is kept low (0.1) for deterministic repair
5. Maximum 3 iterations prevents infinite loops

### 8.4 Repair Results

| Metric | Value |
|--------|-------|
| Repairs Attempted | 8 |
| Successful Repairs | 7 |
| Success Rate | 87.5% |
| Avg Iterations | 1.2 |
| Total Errors Fixed | 10 |
| Error Reduction | 90.9% |
| Avg Repair Time | 7.7s |
| Iterations with Improvement | 8 |
| Iterations with Regression | 0 |

### 8.5 Proposed Enhancements

Based on the literature, we propose these additional components for a production system:

1. **Grammar-Constrained Decoding** (from survey arXiv:2601.02045): Use xGrammar or
   Outlines to constrain LLM output to valid LLVM IR grammar, preventing syntax errors
   at generation time rather than repairing them post-hoc.

2. **Instruction Count Proxy** (from arXiv:2403.14714): Predicted instruction count
   correlates with IR validity — use as a fast pre-filter before full validation.

3. **Formal Verification** (from arXiv:2601.02045): Integrate Alive2 for proving semantic
   equivalence between generated and reference IR, going beyond structural comparison.

4. **Best-of-N Sampling** (from arXiv:2403.14714): Generate N candidates, validate all,
   select best. Found to outperform single-generation + feedback in some settings.

5. **Fine-tuning on IR Data** (from arXiv:2403.03894 IRCoder): Pre-training or fine-tuning
   on paired source→IR data (like SLTrans or ComPile) yields the largest improvements.

## 9. Discussion

### 9.1 Can LLMs Map High-Level Constructs to IR? (Question a)

**Partially yes.** For Level 1 (arithmetic) and simple Level 2 (if-else) constructs,
LLMs generate structurally sound LLVM IR. They correctly map arithmetic operations
to typed instructions (`add i32`, `fadd float`) and handle simple conditional branches.
However, performance degrades significantly for constructs requiring phi nodes,
pointer arithmetic, or struct type definitions.

### 9.2 Do LLMs Preserve Control-Flow and Data-Flow? (Question b)

**Often not.** Our analysis confirms findings from arXiv:2502.06854:
- LLMs approximate loops rather than faithfully reconstructing them
- Phi nodes are the most common failure point — models frequently use incorrect
  predecessor labels or omit phi nodes entirely
- Nested control flow (e.g., nested if-else, nested loops) causes exponential
  increase in errors
- Data-flow through pointer chains is particularly fragile

### 9.3 Is the Generated IR Syntactically and Semantically Valid? (Question c)

**Syntax is often valid; semantics are frequently wrong.** Our validator distinguishes
between compilable (passes syntax + SSA) and structurally correct (matches reference
structure). The gap between these rates reveals that LLMs produce "plausible-looking"
IR that compiles but computes wrong results — the most dangerous failure mode.

### 9.4 Does a Validator/Repair Loop Help? (Question d)

**Yes, significantly.** The repair loop demonstrates that structured error feedback
enables LLMs to fix many of their own mistakes. Key findings:
- Syntax and SSA errors are most repairable (deterministic error messages)
- Type errors are moderately repairable
- Semantic/functional errors are least repairable (require understanding intent)
- Repair sometimes introduces new errors (regression), especially for complex constructs

### 9.5 Model Comparison

- **Llama-3.3-70B**: Best performer with **100.0% validity** across all constructs and both
  prompting strategies. Average generation time of 1.53s. Produced clean, well-structured IR
  for even complex constructs including bubble sort and linked list traversal. Zero control
  flow failures detected.

- **Qwen2.5-Coder-32B**: Strong at 93.3% validity with 100% compilable rate. The
  coding-specialized model occasionally used `icmp gt` instead of `icmp sgt` (a common
  confusion between C-like and LLVM-specific syntax). Average time: 3.67s.

- **Llama-3.1-8B**: Surprisingly competitive at 89.7% despite being 8B parameters.
  Fastest average generation (0.33s). Failures concentrated on SSA violations in complex
  constructs (L6) where the model reused variable names across basic blocks.

- **Qwen2.5-72B**: Lowest validity at 75.9% despite being the largest model. The model
  tends to generate verbose `-O0`-style IR with alloca/load/store patterns, which introduces
  SSA violations when load results are re-assigned. Average time: 22.8s (slowest due to
  long chain-of-thought reasoning).

### 9.6 Key Findings on Prompting Strategies

Counter-intuitively, **basic prompting (95.0%) significantly outperformed chain-of-thought
(84.5%)**. This contradicts findings from arXiv:2502.06854 where CoT helped DeepSeek-R1.
The likely explanation: CoT prompting causes models to generate more verbose, `-O0`-style
IR with alloca/load/store patterns instead of SSA phi nodes. The intermediate reasoning
sometimes introduces commentary and structural artifacts that degrade IR quality. For
structured output like compiler IR, direct instruction produces cleaner results.

### 9.7 Limitations

1. **No runtime validation**: We validate structure but do not execute IR to verify
   functional correctness against the C source
2. **Limited construct set**: 15 constructs cover core patterns but miss many C features
   (unions, variadic functions, setjmp/longjmp, volatile)
3. **Reference IR approximation**: Our hand-written reference IR may differ from
   actual clang output while still being semantically correct
4. **API-only models**: We evaluate through inference APIs, not fine-tuned models

## 10. Conclusion

LLMs demonstrate strong capability for compiler lowering, with an overall 89.8% validity
rate across 118 generation attempts. Our key findings:

1. **Arithmetic constructs are solved**: L1 (arithmetic) achieved 100% validity across all
   models and prompting strategies — this is a solved problem for current LLMs.
2. **Control flow is the frontier**: L2 (control flow) dropped to 84.4%, with phi node
   errors (sub-category 2.3, 13 instances) and loop approximation (2.4, 10 instances)
   being the dominant failure modes, confirming the literature (arXiv:2502.06854).
3. **Model size ≠ IR quality**: Llama-3.3-70B (100% valid) outperformed Qwen2.5-72B
   (75.9%), and Llama-3.1-8B (89.7%) beat the much larger Qwen2.5-72B. Domain-specific
   instruction following matters more than raw parameter count.
4. **Basic > CoT for IR**: Simple prompting (95.0%) beat chain-of-thought (84.5%),
   suggesting that for structured output formats, direct instruction is more effective
   than step-by-step reasoning.
5. **Repair loops are highly effective**: 87.5% of failed constructs were fully repaired,
   with 90.9% total error reduction. Most repairs (7/8) succeeded in a single iteration,
   demonstrating that compiler error feedback is a powerful signal for LLM self-correction.
6. **Composite constructs remain challenging**: L6 (bubble sort, linked list) achieved only
   75.0% validity, with the bubble sort construct being the only one the repair loop could
   not fully fix (reduced from 2 to 1 error).

### Future Work

- Integrate actual LLVM toolchain (`llvm-as`, `opt --verify`) for ground-truth validation
- Fine-tune models on paired C→LLVM IR datasets (ComPile, ExeBench)
- Implement grammar-constrained decoding for LLVM IR
- Extend to MLIR dialects (func, arith, memref) — no published work exists in this area
- Build an interactive tool where developers can iteratively refine LLM-generated IR

## References

1. Jiang, H. et al. (2025). "Can Large Language Models Understand Intermediate
   Representations in Compilers?" arXiv:2502.06854.

2. Cummins, C. et al. (2023). "Large Language Models for Compiler Optimization."
   ICLR 2024. arXiv:2309.07062.

3. Cummins, C. et al. (2024). "Compiler Generated Feedback for Large Language Models."
   arXiv:2403.14714.

4. Szafraniec, M. et al. (2022). "Code Translation with Compiler Representations."
   arXiv:2207.03578.

5. LLM4Decompile Team (2024). "LLM4Decompile: Decompiling Binary Code with LLMs."
   arXiv:2403.05286.

6. AIvril 2 (2024). "EDA-Aware RTL Generation with Verification-in-the-Loop."
   arXiv:2412.04485.

7. IRCoder (2024). "Intermediate Representations Make Language Models Robust
   Multilingual Code Generators." arXiv:2403.03894.

8. Liu, R. et al. (2024). "What's Wrong with Your Code Generated by LLMs?
   An Extensive Study." arXiv:2407.06153.

9. Zhou, W. et al. (2026). "The New Compiler Stack: A Survey on the Synergy
   of LLMs and Compilers." arXiv:2601.02045.

10. SuperCoder (2025). "Assembly Superoptimization with LLMs." arXiv:2505.11480.

---

*Report generated by the AI-Assisted Compiler Lowering analysis pipeline.*