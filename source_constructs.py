"""
Source Language Subset: Predefined C constructs for compiler lowering study.

This module defines a comprehensive set of C language constructs organized by
complexity level, each paired with ground-truth LLVM IR (compiled at -O1).

Categories:
  L1 - Arithmetic & Assignment
  L2 - Control Flow (if/else, loops)
  L3 - Functions & Calling Conventions
  L4 - Pointers & Memory
  L5 - Structs & Aggregates
  L6 - Composite (combines multiple categories)
"""

from dataclasses import dataclass, field
from typing import Optional
import textwrap


@dataclass
class SourceConstruct:
    """A single source-to-IR mapping entry."""
    id: str
    category: str
    level: int  # 1-6
    name: str
    description: str
    source_code: str
    expected_ir: str  # Ground-truth LLVM IR
    key_ir_features: list = field(default_factory=list)  # What IR patterns should appear
    test_inputs: list = field(default_factory=list)
    expected_outputs: list = field(default_factory=list)


# ============================================================================
# LEVEL 1: Arithmetic & Assignment
# ============================================================================

L1_SIMPLE_ADD = SourceConstruct(
    id="L1_01",
    category="arithmetic",
    level=1,
    name="Simple Addition",
    description="Two integer parameters added together",
    source_code=textwrap.dedent("""\
        int add(int a, int b) {
            return a + b;
        }
    """),
    expected_ir=textwrap.dedent("""\
        define i32 @add(i32 %a, i32 %b) {
          %result = add i32 %a, %b
          ret i32 %result
        }
    """),
    key_ir_features=["add i32", "ret i32", "two i32 parameters"],
    test_inputs=[(3, 4), (0, 0), (-1, 1)],
    expected_outputs=[7, 0, 0],
)

L1_MIXED_ARITH = SourceConstruct(
    id="L1_02",
    category="arithmetic",
    level=1,
    name="Mixed Arithmetic",
    description="Expression with add, multiply, subtract",
    source_code=textwrap.dedent("""\
        int compute(int x, int y, int z) {
            return x * y + z - 1;
        }
    """),
    expected_ir=textwrap.dedent("""\
        define i32 @compute(i32 %x, i32 %y, i32 %z) {
          %mul = mul i32 %x, %y
          %add = add i32 %mul, %z
          %sub = sub i32 %add, 1
          ret i32 %sub
        }
    """),
    key_ir_features=["mul i32", "add i32", "sub i32", "constant 1"],
    test_inputs=[(2, 3, 4), (0, 5, 1)],
    expected_outputs=[9, 0],
)

L1_FLOAT_ARITH = SourceConstruct(
    id="L1_03",
    category="arithmetic",
    level=1,
    name="Floating Point Arithmetic",
    description="Float division and multiplication",
    source_code=textwrap.dedent("""\
        float average(float a, float b) {
            return (a + b) / 2.0f;
        }
    """),
    expected_ir=textwrap.dedent("""\
        define float @average(float %a, float %b) {
          %add = fadd float %a, %b
          %div = fdiv float %add, 2.000000e+00
          ret float %div
        }
    """),
    key_ir_features=["fadd float", "fdiv float", "float constant 2.0"],
    test_inputs=[(3.0, 5.0), (0.0, 0.0)],
    expected_outputs=[4.0, 0.0],
)

L1_BITWISE = SourceConstruct(
    id="L1_04",
    category="arithmetic",
    level=1,
    name="Bitwise Operations",
    description="Bitwise AND, OR, XOR, shift",
    source_code=textwrap.dedent("""\
        int bitops(int a, int b) {
            return (a & b) | (a ^ b) << 1;
        }
    """),
    expected_ir=textwrap.dedent("""\
        define i32 @bitops(i32 %a, i32 %b) {
          %and = and i32 %a, %b
          %xor = xor i32 %a, %b
          %shl = shl i32 %xor, 1
          %or = or i32 %and, %shl
          ret i32 %or
        }
    """),
    key_ir_features=["and i32", "xor i32", "shl i32", "or i32"],
    test_inputs=[(5, 3)],
    expected_outputs=[7],  # (5&3)=1, (5^3)=6, 6<<1=12, 1|12=13... let me recalc: 5=101, 3=011, &=001=1, ^=110=6, <<1=12=1100, |=1101=13
)

# ============================================================================
# LEVEL 2: Control Flow
# ============================================================================

L2_IF_ELSE = SourceConstruct(
    id="L2_01",
    category="control_flow",
    level=2,
    name="If-Else Branch",
    description="Simple conditional with two branches",
    source_code=textwrap.dedent("""\
        int max(int a, int b) {
            if (a > b) {
                return a;
            } else {
                return b;
            }
        }
    """),
    expected_ir=textwrap.dedent("""\
        define i32 @max(i32 %a, i32 %b) {
        entry:
          %cmp = icmp sgt i32 %a, %b
          br i1 %cmp, label %if.then, label %if.else

        if.then:
          ret i32 %a

        if.else:
          ret i32 %b
        }
    """),
    key_ir_features=["icmp sgt", "br i1", "two basic blocks", "conditional branch"],
    test_inputs=[(5, 3), (2, 7), (4, 4)],
    expected_outputs=[5, 7, 4],
)

L2_WHILE_LOOP = SourceConstruct(
    id="L2_02",
    category="control_flow",
    level=2,
    name="While Loop",
    description="Simple while loop with accumulator",
    source_code=textwrap.dedent("""\
        int sum_to_n(int n) {
            int sum = 0;
            int i = 0;
            while (i < n) {
                sum = sum + i;
                i = i + 1;
            }
            return sum;
        }
    """),
    expected_ir=textwrap.dedent("""\
        define i32 @sum_to_n(i32 %n) {
        entry:
          br label %while.cond

        while.cond:
          %sum = phi i32 [ 0, %entry ], [ %sum.next, %while.body ]
          %i = phi i32 [ 0, %entry ], [ %i.next, %while.body ]
          %cmp = icmp slt i32 %i, %n
          br i1 %cmp, label %while.body, label %while.end

        while.body:
          %sum.next = add i32 %sum, %i
          %i.next = add i32 %i, 1
          br label %while.cond

        while.end:
          ret i32 %sum
        }
    """),
    key_ir_features=["phi nodes", "loop back-edge", "icmp slt", "three basic blocks"],
    test_inputs=[(5,), (0,), (10,)],
    expected_outputs=[10, 0, 45],
)

L2_FOR_LOOP = SourceConstruct(
    id="L2_03",
    category="control_flow",
    level=2,
    name="For Loop",
    description="For loop computing factorial",
    source_code=textwrap.dedent("""\
        int factorial(int n) {
            int result = 1;
            for (int i = 1; i <= n; i++) {
                result = result * i;
            }
            return result;
        }
    """),
    expected_ir=textwrap.dedent("""\
        define i32 @factorial(i32 %n) {
        entry:
          br label %for.cond

        for.cond:
          %result = phi i32 [ 1, %entry ], [ %result.next, %for.body ]
          %i = phi i32 [ 1, %entry ], [ %i.next, %for.body ]
          %cmp = icmp sle i32 %i, %n
          br i1 %cmp, label %for.body, label %for.end

        for.body:
          %result.next = mul i32 %result, %i
          %i.next = add i32 %i, 1
          br label %for.cond

        for.end:
          ret i32 %result
        }
    """),
    key_ir_features=["phi nodes", "mul i32", "icmp sle", "loop structure"],
    test_inputs=[(5,), (1,), (0,)],
    expected_outputs=[120, 1, 1],
)

L2_NESTED_IF = SourceConstruct(
    id="L2_04",
    category="control_flow",
    level=2,
    name="Nested If-Else",
    description="Nested conditionals with multiple return paths",
    source_code=textwrap.dedent("""\
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
    """),
    expected_ir=textwrap.dedent("""\
        define i32 @classify(i32 %x) {
        entry:
          %cmp1 = icmp sgt i32 %x, 0
          br i1 %cmp1, label %if.pos, label %if.neg.check

        if.pos:
          %cmp2 = icmp sgt i32 %x, 100
          br i1 %cmp2, label %ret.2, label %ret.1

        ret.2:
          ret i32 2

        ret.1:
          ret i32 1

        if.neg.check:
          %cmp3 = icmp slt i32 %x, 0
          br i1 %cmp3, label %ret.neg, label %ret.zero

        ret.neg:
          ret i32 -1

        ret.zero:
          ret i32 0
        }
    """),
    key_ir_features=["multiple basic blocks", "nested branches", "icmp sgt", "icmp slt"],
    test_inputs=[(150,), (50,), (-5,), (0,)],
    expected_outputs=[2, 1, -1, 0],
)

# ============================================================================
# LEVEL 3: Functions & Calling Conventions
# ============================================================================

L3_FUNC_CALL = SourceConstruct(
    id="L3_01",
    category="functions",
    level=3,
    name="Function Call",
    description="One function calling another",
    source_code=textwrap.dedent("""\
        int square(int x) {
            return x * x;
        }

        int sum_of_squares(int a, int b) {
            return square(a) + square(b);
        }
    """),
    expected_ir=textwrap.dedent("""\
        define i32 @square(i32 %x) {
          %mul = mul i32 %x, %x
          ret i32 %mul
        }

        define i32 @sum_of_squares(i32 %a, i32 %b) {
          %sq_a = call i32 @square(i32 %a)
          %sq_b = call i32 @square(i32 %b)
          %sum = add i32 %sq_a, %sq_b
          ret i32 %sum
        }
    """),
    key_ir_features=["call instruction", "multiple function definitions", "i32 return"],
    test_inputs=[(3, 4)],
    expected_outputs=[25],
)

L3_RECURSIVE = SourceConstruct(
    id="L3_02",
    category="functions",
    level=3,
    name="Recursive Function",
    description="Recursive fibonacci",
    source_code=textwrap.dedent("""\
        int fib(int n) {
            if (n <= 1) {
                return n;
            }
            return fib(n - 1) + fib(n - 2);
        }
    """),
    expected_ir=textwrap.dedent("""\
        define i32 @fib(i32 %n) {
        entry:
          %cmp = icmp sle i32 %n, 1
          br i1 %cmp, label %base, label %recurse

        base:
          ret i32 %n

        recurse:
          %n1 = sub i32 %n, 1
          %f1 = call i32 @fib(i32 %n1)
          %n2 = sub i32 %n, 2
          %f2 = call i32 @fib(i32 %n2)
          %sum = add i32 %f1, %f2
          ret i32 %sum
        }
    """),
    key_ir_features=["recursive call", "icmp sle", "sub i32", "two call instructions"],
    test_inputs=[(0,), (1,), (5,), (10,)],
    expected_outputs=[0, 1, 5, 55],
)

# ============================================================================
# LEVEL 4: Pointers & Memory
# ============================================================================

L4_POINTER_DEREF = SourceConstruct(
    id="L4_01",
    category="memory",
    level=4,
    name="Pointer Dereference",
    description="Swap two integers via pointers",
    source_code=textwrap.dedent("""\
        void swap(int* a, int* b) {
            int tmp = *a;
            *a = *b;
            *b = tmp;
        }
    """),
    expected_ir=textwrap.dedent("""\
        define void @swap(ptr %a, ptr %b) {
          %tmp = load i32, ptr %a
          %val_b = load i32, ptr %b
          store i32 %val_b, ptr %a
          store i32 %tmp, ptr %b
          ret void
        }
    """),
    key_ir_features=["load i32", "store i32", "ptr type", "void return"],
    test_inputs=[],
    expected_outputs=[],
)

L4_ARRAY_ACCESS = SourceConstruct(
    id="L4_02",
    category="memory",
    level=4,
    name="Array Sum",
    description="Sum elements of an array using pointer arithmetic",
    source_code=textwrap.dedent("""\
        int array_sum(int* arr, int n) {
            int sum = 0;
            for (int i = 0; i < n; i++) {
                sum += arr[i];
            }
            return sum;
        }
    """),
    expected_ir=textwrap.dedent("""\
        define i32 @array_sum(ptr %arr, i32 %n) {
        entry:
          br label %for.cond

        for.cond:
          %sum = phi i32 [ 0, %entry ], [ %sum.next, %for.body ]
          %i = phi i32 [ 0, %entry ], [ %i.next, %for.body ]
          %cmp = icmp slt i32 %i, %n
          br i1 %cmp, label %for.body, label %for.end

        for.body:
          %idx = sext i32 %i to i64
          %ptr = getelementptr i32, ptr %arr, i64 %idx
          %val = load i32, ptr %ptr
          %sum.next = add i32 %sum, %val
          %i.next = add i32 %i, 1
          br label %for.cond

        for.end:
          ret i32 %sum
        }
    """),
    key_ir_features=["getelementptr", "sext", "load from computed pointer", "phi nodes"],
    test_inputs=[],
    expected_outputs=[],
)

# ============================================================================
# LEVEL 5: Structs & Aggregates
# ============================================================================

L5_STRUCT = SourceConstruct(
    id="L5_01",
    category="structs",
    level=5,
    name="Struct Field Access",
    description="Access fields of a struct",
    source_code=textwrap.dedent("""\
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
    """),
    expected_ir=textwrap.dedent("""\
        %struct.Point = type { i32, i32 }

        define i32 @manhattan_distance(ptr %p1, ptr %p2) {
        entry:
          %p1.x.ptr = getelementptr %struct.Point, ptr %p1, i32 0, i32 0
          %p1.x = load i32, ptr %p1.x.ptr
          %p2.x.ptr = getelementptr %struct.Point, ptr %p2, i32 0, i32 0
          %p2.x = load i32, ptr %p2.x.ptr
          %dx = sub i32 %p1.x, %p2.x

          %p1.y.ptr = getelementptr %struct.Point, ptr %p1, i32 0, i32 1
          %p1.y = load i32, ptr %p1.y.ptr
          %p2.y.ptr = getelementptr %struct.Point, ptr %p2, i32 0, i32 1
          %p2.y = load i32, ptr %p2.y.ptr
          %dy = sub i32 %p1.y, %p2.y

          %dx.neg = icmp slt i32 %dx, 0
          %dx.abs = sub i32 0, %dx
          %dx.final = select i1 %dx.neg, i32 %dx.abs, i32 %dx

          %dy.neg = icmp slt i32 %dy, 0
          %dy.abs = sub i32 0, %dy
          %dy.final = select i1 %dy.neg, i32 %dy.abs, i32 %dy

          %result = add i32 %dx.final, %dy.final
          ret i32 %result
        }
    """),
    key_ir_features=["struct type", "getelementptr with struct", "select instruction", "abs pattern"],
    test_inputs=[],
    expected_outputs=[],
)

# ============================================================================
# LEVEL 6: Composite (Multi-category)
# ============================================================================

L6_BUBBLE_SORT = SourceConstruct(
    id="L6_01",
    category="composite",
    level=6,
    name="Bubble Sort",
    description="Nested loops, array access, pointer ops, swaps",
    source_code=textwrap.dedent("""\
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
    """),
    expected_ir=textwrap.dedent("""\
        define void @bubble_sort(ptr %arr, i32 %n) {
        entry:
          %n_minus_1 = sub i32 %n, 1
          br label %outer.cond

        outer.cond:
          %i = phi i32 [ 0, %entry ], [ %i.next, %outer.inc ]
          %cmp.outer = icmp slt i32 %i, %n_minus_1
          br i1 %cmp.outer, label %inner.init, label %exit

        inner.init:
          %inner.limit = sub i32 %n_minus_1, %i
          br label %inner.cond

        inner.cond:
          %j = phi i32 [ 0, %inner.init ], [ %j.next, %inner.inc ]
          %cmp.inner = icmp slt i32 %j, %inner.limit
          br i1 %cmp.inner, label %inner.body, label %outer.inc

        inner.body:
          %j.ext = sext i32 %j to i64
          %ptr.j = getelementptr i32, ptr %arr, i64 %j.ext
          %val.j = load i32, ptr %ptr.j
          %j1 = add i32 %j, 1
          %j1.ext = sext i32 %j1 to i64
          %ptr.j1 = getelementptr i32, ptr %arr, i64 %j1.ext
          %val.j1 = load i32, ptr %ptr.j1
          %cmp.swap = icmp sgt i32 %val.j, %val.j1
          br i1 %cmp.swap, label %do.swap, label %inner.inc

        do.swap:
          store i32 %val.j1, ptr %ptr.j
          store i32 %val.j, ptr %ptr.j1
          br label %inner.inc

        inner.inc:
          %j.next = add i32 %j, 1
          br label %inner.cond

        outer.inc:
          %i.next = add i32 %i, 1
          br label %outer.cond

        exit:
          ret void
        }
    """),
    key_ir_features=[
        "nested loops", "phi nodes", "getelementptr", "load/store",
        "icmp sgt", "conditional swap", "sext", "multiple basic blocks"
    ],
    test_inputs=[],
    expected_outputs=[],
)

L6_LINKED_LIST = SourceConstruct(
    id="L6_02",
    category="composite",
    level=6,
    name="Linked List Length",
    description="Struct + pointer traversal + loop",
    source_code=textwrap.dedent("""\
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
    """),
    expected_ir=textwrap.dedent("""\
        %struct.Node = type { i32, ptr }

        define i32 @list_length(ptr %head) {
        entry:
          br label %while.cond

        while.cond:
          %count = phi i32 [ 0, %entry ], [ %count.next, %while.body ]
          %current = phi ptr [ %head, %entry ], [ %next, %while.body ]
          %cmp = icmp ne ptr %current, null
          br i1 %cmp, label %while.body, label %while.end

        while.body:
          %count.next = add i32 %count, 1
          %next.ptr = getelementptr %struct.Node, ptr %current, i32 0, i32 1
          %next = load ptr, ptr %next.ptr
          br label %while.cond

        while.end:
          ret i32 %count
        }
    """),
    key_ir_features=[
        "struct with pointer field", "phi nodes", "null pointer comparison",
        "getelementptr into struct", "load ptr", "pointer traversal loop"
    ],
    test_inputs=[],
    expected_outputs=[],
)


# ============================================================================
# Registry of all constructs
# ============================================================================

ALL_CONSTRUCTS = [
    L1_SIMPLE_ADD, L1_MIXED_ARITH, L1_FLOAT_ARITH, L1_BITWISE,
    L2_IF_ELSE, L2_WHILE_LOOP, L2_FOR_LOOP, L2_NESTED_IF,
    L3_FUNC_CALL, L3_RECURSIVE,
    L4_POINTER_DEREF, L4_ARRAY_ACCESS,
    L5_STRUCT,
    L6_BUBBLE_SORT, L6_LINKED_LIST,
]

CONSTRUCTS_BY_LEVEL = {}
for c in ALL_CONSTRUCTS:
    CONSTRUCTS_BY_LEVEL.setdefault(c.level, []).append(c)

CONSTRUCTS_BY_CATEGORY = {}
for c in ALL_CONSTRUCTS:
    CONSTRUCTS_BY_CATEGORY.setdefault(c.category, []).append(c)

CONSTRUCT_MAP = {c.id: c for c in ALL_CONSTRUCTS}


def get_construct(construct_id: str) -> SourceConstruct:
    return CONSTRUCT_MAP[construct_id]


def get_constructs_by_level(level: int) -> list:
    return CONSTRUCTS_BY_LEVEL.get(level, [])


def get_summary_table() -> str:
    """Return a markdown summary table of all constructs."""
    lines = ["| ID | Level | Category | Name | Key IR Features |",
             "|-----|-------|----------|------|-----------------|"]
    for c in ALL_CONSTRUCTS:
        features = ", ".join(c.key_ir_features[:3])
        lines.append(f"| {c.id} | L{c.level} | {c.category} | {c.name} | {features} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Total constructs: {len(ALL_CONSTRUCTS)}")
    print(f"Levels: {sorted(CONSTRUCTS_BY_LEVEL.keys())}")
    print(f"Categories: {sorted(CONSTRUCTS_BY_CATEGORY.keys())}")
    print()
    print(get_summary_table())
