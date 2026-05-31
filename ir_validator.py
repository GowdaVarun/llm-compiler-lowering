"""
LLVM IR Validator — Static analysis of LLM-generated LLVM IR.

Validates:
  1. Syntax: basic structural correctness
  2. SSA: single static assignment form
  3. Type system: operand type consistency
  4. Control flow: basic block structure, terminators, branch targets, phi nodes
  5. Semantic: instruction format, operand counts

Returns a detailed ValidationReport with categorized errors.
"""

import re
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ErrorCategory(Enum):
    SYNTAX = "syntax"
    SSA = "ssa"
    TYPE = "type"
    CONTROL_FLOW = "control_flow"
    SEMANTIC = "semantic"


class ErrorSeverity(Enum):
    ERROR = "error"      # Will not compile
    WARNING = "warning"  # May compile but semantically suspicious


@dataclass
class ValidationError:
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    line_number: Optional[int] = None
    line_content: Optional[str] = None

    def to_dict(self):
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "line_number": self.line_number,
            "line_content": self.line_content,
        }


@dataclass
class BasicBlock:
    label: str
    instructions: list = field(default_factory=list)
    terminator: Optional[str] = None
    predecessors: list = field(default_factory=list)
    successors: list = field(default_factory=list)
    phi_nodes: list = field(default_factory=list)
    defined_vars: set = field(default_factory=set)
    used_vars: set = field(default_factory=set)


@dataclass
class FunctionInfo:
    name: str
    return_type: str
    params: list = field(default_factory=list)
    blocks: dict = field(default_factory=dict)  # label -> BasicBlock
    all_defined: set = field(default_factory=set)
    all_used: set = field(default_factory=set)


@dataclass
class ValidationReport:
    errors: list = field(default_factory=list)
    functions_found: int = 0
    basic_blocks_found: int = 0
    instructions_found: int = 0
    is_valid: bool = True  # No errors (warnings ok)
    is_compilable: bool = True  # No syntax/ssa errors

    def add_error(self, error: ValidationError):
        self.errors.append(error)
        if error.severity == ErrorSeverity.ERROR:
            self.is_valid = False
            if error.category in (ErrorCategory.SYNTAX, ErrorCategory.SSA):
                self.is_compilable = False

    @property
    def error_count(self):
        return sum(1 for e in self.errors if e.severity == ErrorSeverity.ERROR)

    @property
    def warning_count(self):
        return sum(1 for e in self.errors if e.severity == ErrorSeverity.WARNING)

    def errors_by_category(self):
        cats = {}
        for e in self.errors:
            cats.setdefault(e.category.value, []).append(e)
        return cats

    def summary(self):
        cats = self.errors_by_category()
        lines = [
            f"Validation: {'PASS' if self.is_valid else 'FAIL'}",
            f"  Functions: {self.functions_found}",
            f"  Basic Blocks: {self.basic_blocks_found}",
            f"  Instructions: {self.instructions_found}",
            f"  Errors: {self.error_count}, Warnings: {self.warning_count}",
        ]
        for cat, errs in cats.items():
            lines.append(f"  [{cat}]: {len(errs)} issues")
            for e in errs[:5]:
                loc = f" (line {e.line_number})" if e.line_number else ""
                lines.append(f"    - {e.severity.value}: {e.message}{loc}")
            if len(errs) > 5:
                lines.append(f"    ... and {len(errs)-5} more")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "is_valid": self.is_valid,
            "is_compilable": self.is_compilable,
            "functions_found": self.functions_found,
            "basic_blocks_found": self.basic_blocks_found,
            "instructions_found": self.instructions_found,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors": [e.to_dict() for e in self.errors],
        }


# ============================================================================
# LLVM IR Types
# ============================================================================

LLVM_INT_TYPES = {"i1", "i8", "i16", "i32", "i64", "i128"}
LLVM_FLOAT_TYPES = {"half", "float", "double", "fp128"}
LLVM_VOID = "void"
LLVM_PTR = "ptr"

LLVM_TERMINATOR_OPS = {"ret", "br", "switch", "unreachable", "invoke", "resume", "indirectbr"}
LLVM_BINARY_OPS = {"add", "sub", "mul", "udiv", "sdiv", "urem", "srem",
                    "fadd", "fsub", "fmul", "fdiv", "frem"}
LLVM_BITWISE_OPS = {"shl", "lshr", "ashr", "and", "or", "xor"}
LLVM_COMPARE_OPS = {"icmp", "fcmp"}
LLVM_MEMORY_OPS = {"alloca", "load", "store", "getelementptr"}
LLVM_CAST_OPS = {"trunc", "zext", "sext", "fptrunc", "fpext", "fptoui",
                 "fptosi", "uitofp", "sitofp", "ptrtoint", "inttoptr", "bitcast"}
LLVM_OTHER_OPS = {"phi", "select", "call", "extractvalue", "insertvalue"}

ALL_OPS = (LLVM_TERMINATOR_OPS | LLVM_BINARY_OPS | LLVM_BITWISE_OPS |
           LLVM_COMPARE_OPS | LLVM_MEMORY_OPS | LLVM_CAST_OPS | LLVM_OTHER_OPS)

# ICMP predicates
ICMP_PREDS = {"eq", "ne", "ugt", "uge", "ult", "ule", "sgt", "sge", "slt", "sle"}
FCMP_PREDS = {"oeq", "ogt", "oge", "olt", "ole", "one", "ord", "ueq", "ugt",
              "uge", "ult", "ule", "une", "uno", "false", "true"}


# ============================================================================
# Parser helpers
# ============================================================================

_TYPE_RE = re.compile(
    r'^(void|ptr|i\d+|half|float|double|fp128|'
    r'\[.+?\]|<.+?>|{.+?}|%[a-zA-Z_.][a-zA-Z0-9_.]*'
    r')(\*)*$'
)

_SSA_VAR_RE = re.compile(r'%[a-zA-Z_.][a-zA-Z0-9_.]*|%\d+')
_LABEL_RE = re.compile(r'^([a-zA-Z_.][a-zA-Z0-9_.]*):')
_FUNC_DEF_RE = re.compile(
    r'define\s+(?:(?:internal|external|private|linkonce|weak|common|appending|'
    r'extern_weak|linkonce_odr|weak_odr|available_externally|dllimport|dllexport)\s+)*'
    r'(.+?)\s+@([a-zA-Z_.][a-zA-Z0-9_.]*)\s*\(([^)]*)\)'
)
_FUNC_DECL_RE = re.compile(r'declare\s+(.+?)\s+@([a-zA-Z_.][a-zA-Z0-9_.]*)\s*\(([^)]*)\)')
_STRUCT_DEF_RE = re.compile(r'(%[a-zA-Z_.][a-zA-Z0-9_.]*)\s*=\s*type\s*\{([^}]*)\}')
_ASSIGN_RE = re.compile(r'(%[a-zA-Z_.][a-zA-Z0-9_.]*|%\d+)\s*=\s*(.*)')


def strip_label_refs(text):
    """Remove label references (label %name) and phi-block refs ([ val, %block ]) 
    so they aren't mistaken for SSA variable uses."""
    # Remove "label %name" patterns (from br, switch instructions)
    cleaned = re.sub(r'label\s+%[a-zA-Z_.][a-zA-Z0-9_.]*', 'label STRIPPED', text)
    # Remove ", %blockname ]" patterns in phi nodes: [ value, %block ]
    cleaned = re.sub(r',\s*%([a-zA-Z_.][a-zA-Z0-9_.]*)\s*\]', ', STRIPPED_LABEL ]', cleaned)
    return cleaned


def extract_ssa_vars(text):
    """Extract all %name or %number references from a string, 
    excluding label references in branch/phi instructions."""
    cleaned = strip_label_refs(text)
    return set(_SSA_VAR_RE.findall(cleaned))


def parse_type(token):
    """Basic type extraction from a token."""
    token = token.strip()
    if token in LLVM_INT_TYPES | LLVM_FLOAT_TYPES | {LLVM_VOID, LLVM_PTR}:
        return token
    if token.endswith("*"):
        return "ptr"  # Opaque pointers
    if token.startswith("%"):
        return token  # Named struct type
    if token.startswith("["):
        return "array"
    if token.startswith("<"):
        return "vector"
    if token.startswith("{"):
        return "struct"
    return None


# ============================================================================
# Main Validator
# ============================================================================

class LLVMIRValidator:
    """Validates LLVM IR text for structural and semantic correctness."""

    def __init__(self, ir_text: str):
        self.ir_text = ir_text
        self.lines = ir_text.strip().split("\n")
        self.report = ValidationReport()
        self.functions = {}       # name -> FunctionInfo
        self.struct_types = {}    # %name -> field types
        self.declared_funcs = set()
        self.defined_funcs = set()

    def validate(self) -> ValidationReport:
        """Run all validation passes and return the report."""
        # Clean up markdown code fences if present
        self._strip_code_fences()

        # Pass 1: Structural parse
        self._parse_structure()

        # Pass 2: SSA validation
        self._validate_ssa()

        # Pass 3: Type checking
        self._validate_types()

        # Pass 4: Control flow validation
        self._validate_control_flow()

        # Pass 5: Semantic checks
        self._validate_semantics()

        return self.report

    def _strip_code_fences(self):
        """Remove markdown code fences that LLMs often add."""
        cleaned = []
        in_fence = False
        for line in self.lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            cleaned.append(line)
        self.lines = cleaned
        self.ir_text = "\n".join(cleaned)

    def _parse_structure(self):
        """Parse the IR into functions, basic blocks, and instructions."""
        # Parse struct definitions
        for match in _STRUCT_DEF_RE.finditer(self.ir_text):
            name = match.group(1)
            fields = match.group(2).strip()
            self.struct_types[name] = fields

        # Parse function declarations
        for match in _FUNC_DECL_RE.finditer(self.ir_text):
            self.declared_funcs.add(match.group(2))

        # Parse function definitions
        current_func = None
        current_block = None
        brace_depth = 0

        for line_num, line in enumerate(self.lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith(";") or stripped.startswith("source_filename"):
                continue
            if stripped.startswith("target ") or stripped.startswith("attributes "):
                continue
            if stripped.startswith("%") and "= type" in stripped:
                continue  # struct definition

            # Function definition start
            func_match = _FUNC_DEF_RE.search(stripped)
            if func_match:
                ret_type = func_match.group(1).strip()
                func_name = func_match.group(2)
                params_str = func_match.group(3)
                params = self._parse_params(params_str)

                current_func = FunctionInfo(
                    name=func_name,
                    return_type=ret_type,
                    params=params,
                )
                self.functions[func_name] = current_func
                self.defined_funcs.add(func_name)
                self.report.functions_found += 1

                # First block is implicit "entry" or explicit label
                current_block = BasicBlock(label="entry")
                current_func.blocks["entry"] = current_block
                self.report.basic_blocks_found += 1

                # Add param names as defined vars
                for p_type, p_name in params:
                    if p_name:
                        current_func.all_defined.add(p_name)

                if "{" in stripped:
                    brace_depth += stripped.count("{") - stripped.count("}")
                continue

            if stripped == "}":
                brace_depth -= 1
                current_func = None
                current_block = None
                continue

            if not current_func:
                # Handle declare
                if stripped.startswith("declare"):
                    decl_match = _FUNC_DECL_RE.search(stripped)
                    if decl_match:
                        self.declared_funcs.add(decl_match.group(2))
                continue

            # Label
            label_match = _LABEL_RE.match(stripped)
            if label_match:
                label_name = label_match.group(1)
                if label_name in current_func.blocks and label_name != "entry":
                    self.report.add_error(ValidationError(
                        ErrorCategory.CONTROL_FLOW, ErrorSeverity.ERROR,
                        f"Duplicate basic block label '{label_name}' in function @{current_func.name}",
                        line_num, stripped,
                    ))
                current_block = BasicBlock(label=label_name)
                current_func.blocks[label_name] = current_block
                self.report.basic_blocks_found += 1
                continue

            # Instruction
            if current_block is not None:
                self._parse_instruction(stripped, line_num, current_func, current_block)

    def _parse_params(self, params_str):
        """Parse function parameter list."""
        params = []
        if not params_str.strip():
            return params
        for param in params_str.split(","):
            param = param.strip()
            parts = param.split()
            if len(parts) >= 2:
                p_type = parts[0]
                p_name = parts[1] if parts[1].startswith("%") else None
                params.append((p_type, p_name))
            elif len(parts) == 1:
                params.append((parts[0], None))
        return params

    def _parse_instruction(self, line, line_num, func, block):
        """Parse a single instruction and record definitions/uses."""
        self.report.instructions_found += 1

        # Check for assignment (definition)
        assign_match = _ASSIGN_RE.match(line)
        if assign_match:
            defined_var = assign_match.group(1)
            rhs = assign_match.group(2)

            # SSA check: variable defined more than once in same function
            if defined_var in func.all_defined:
                self.report.add_error(ValidationError(
                    ErrorCategory.SSA, ErrorSeverity.ERROR,
                    f"SSA violation: '{defined_var}' defined more than once in @{func.name}",
                    line_num, line,
                ))
            func.all_defined.add(defined_var)
            block.defined_vars.add(defined_var)

            # Parse RHS for used variables
            used = extract_ssa_vars(rhs)
            # Don't count the defined var itself (in phi, it appears on both sides)
            used.discard(defined_var)
            func.all_used.update(used)
            block.used_vars.update(used)

            # Check for phi nodes
            if rhs.strip().startswith("phi"):
                block.phi_nodes.append((defined_var, rhs, line_num))

            block.instructions.append(("assign", defined_var, rhs, line_num))
        else:
            # Non-assignment instruction (store, br, ret, etc.)
            tokens = line.split()
            if tokens:
                op = tokens[0]
                if op in LLVM_TERMINATOR_OPS:
                    block.terminator = line
                    # Extract branch targets for control flow
                    if op == "br":
                        labels = re.findall(r'label\s+%([a-zA-Z_.][a-zA-Z0-9_.]*)', line)
                        block.successors.extend(labels)
                    elif op == "switch":
                        labels = re.findall(r'label\s+%([a-zA-Z_.][a-zA-Z0-9_.]*)', line)
                        block.successors.extend(labels)

                used = extract_ssa_vars(line)
                func.all_used.update(used)
                block.used_vars.update(used)
                block.instructions.append(("stmt", None, line, line_num))

    def _validate_ssa(self):
        """Validate SSA properties across all functions."""
        for fname, func in self.functions.items():
            # Check for uses of undefined variables
            # Parameters and function-level definitions are both valid
            all_available = func.all_defined.copy()
            for p_type, p_name in func.params:
                if p_name:
                    all_available.add(p_name)

            for var in func.all_used:
                if var not in all_available:
                    # Could be a global or function reference
                    if not var.startswith("%struct.") and var not in self.struct_types:
                        self.report.add_error(ValidationError(
                            ErrorCategory.SSA, ErrorSeverity.ERROR,
                            f"Use of undefined SSA variable '{var}' in @{fname}",
                        ))

    def _validate_types(self):
        """Validate type consistency in instructions."""
        for fname, func in self.functions.items():
            for blabel, block in func.blocks.items():
                for inst_type, defined, rhs, line_num in block.instructions:
                    if rhs is None:
                        continue
                    rhs_stripped = rhs.strip()

                    # Check binary ops have matching types
                    for op in LLVM_BINARY_OPS:
                        if rhs_stripped.startswith(op + " "):
                            self._check_binary_op_types(op, rhs_stripped, line_num)
                            break

                    # Check icmp/fcmp predicates
                    if rhs_stripped.startswith("icmp "):
                        self._check_icmp(rhs_stripped, line_num)
                    elif rhs_stripped.startswith("fcmp "):
                        self._check_fcmp(rhs_stripped, line_num)

    def _check_binary_op_types(self, op, rhs, line_num):
        """Check that a binary operation has consistent types."""
        # e.g., "add i32 %a, %b" or "add nsw i32 %a, %b"
        tokens = rhs.split()
        # Find the type token
        type_token = None
        for t in tokens[1:]:
            if t in ("nsw", "nuw", "exact", "fast", "nnan", "ninf"):
                continue
            type_token = t
            break

        if type_token:
            # Integer ops should use integer types
            if op in ("add", "sub", "mul", "udiv", "sdiv", "urem", "srem",
                       "shl", "lshr", "ashr", "and", "or", "xor"):
                if type_token in LLVM_FLOAT_TYPES:
                    self.report.add_error(ValidationError(
                        ErrorCategory.TYPE, ErrorSeverity.ERROR,
                        f"Integer operation '{op}' used with float type '{type_token}'",
                        line_num,
                    ))
            # Float ops should use float types
            elif op in ("fadd", "fsub", "fmul", "fdiv", "frem"):
                if type_token in LLVM_INT_TYPES:
                    self.report.add_error(ValidationError(
                        ErrorCategory.TYPE, ErrorSeverity.ERROR,
                        f"Float operation '{op}' used with integer type '{type_token}'",
                        line_num,
                    ))

    def _check_icmp(self, rhs, line_num):
        """Check icmp instruction validity."""
        tokens = rhs.split()
        if len(tokens) >= 2:
            pred = tokens[1]
            if pred not in ICMP_PREDS:
                self.report.add_error(ValidationError(
                    ErrorCategory.SEMANTIC, ErrorSeverity.ERROR,
                    f"Invalid icmp predicate '{pred}'. Valid: {', '.join(sorted(ICMP_PREDS))}",
                    line_num,
                ))

    def _check_fcmp(self, rhs, line_num):
        """Check fcmp instruction validity."""
        tokens = rhs.split()
        if len(tokens) >= 2:
            pred = tokens[1]
            if pred not in FCMP_PREDS:
                self.report.add_error(ValidationError(
                    ErrorCategory.SEMANTIC, ErrorSeverity.ERROR,
                    f"Invalid fcmp predicate '{pred}'. Valid: {', '.join(sorted(FCMP_PREDS))}",
                    line_num,
                ))

    def _validate_control_flow(self):
        """Validate control flow properties."""
        for fname, func in self.functions.items():
            # Check every block has a terminator
            for blabel, block in func.blocks.items():
                if not block.terminator:
                    self.report.add_error(ValidationError(
                        ErrorCategory.CONTROL_FLOW, ErrorSeverity.ERROR,
                        f"Basic block '{blabel}' in @{fname} has no terminator instruction",
                    ))

                # Check branch targets exist
                for succ in block.successors:
                    if succ not in func.blocks:
                        self.report.add_error(ValidationError(
                            ErrorCategory.CONTROL_FLOW, ErrorSeverity.ERROR,
                            f"Branch target '%{succ}' not found in @{fname} (from block '{blabel}')",
                        ))

            # Check phi nodes
            self._validate_phi_nodes(func)

            # Check entry block (first block shouldn't have phi nodes normally)
            if func.blocks:
                first_label = list(func.blocks.keys())[0]
                first_block = func.blocks[first_label]
                if first_block.phi_nodes:
                    self.report.add_error(ValidationError(
                        ErrorCategory.CONTROL_FLOW, ErrorSeverity.WARNING,
                        f"Entry block '{first_label}' in @{fname} contains phi nodes (unusual)",
                    ))

    def _validate_phi_nodes(self, func):
        """Validate phi node correctness."""
        # Build predecessor map
        predecessors = {label: [] for label in func.blocks}
        for blabel, block in func.blocks.items():
            for succ in block.successors:
                if succ in predecessors:
                    predecessors[succ].append(blabel)

        for blabel, block in func.blocks.items():
            for var, phi_rhs, line_num in block.phi_nodes:
                # Extract block labels from phi
                phi_blocks = re.findall(r'%([a-zA-Z_.][a-zA-Z0-9_.]*)\s*\]', phi_rhs)

                # Check each phi predecessor exists
                for pb in phi_blocks:
                    if pb not in func.blocks:
                        self.report.add_error(ValidationError(
                            ErrorCategory.CONTROL_FLOW, ErrorSeverity.ERROR,
                            f"Phi node in '{blabel}': predecessor '%{pb}' does not exist in @{func.name}",
                            line_num,
                        ))

                # Check phi has entry for each actual predecessor
                actual_preds = set(predecessors.get(blabel, []))
                phi_pred_set = set(phi_blocks)
                missing = actual_preds - phi_pred_set
                extra = phi_pred_set - actual_preds - set(func.blocks.keys())

                if missing and actual_preds:
                    self.report.add_error(ValidationError(
                        ErrorCategory.CONTROL_FLOW, ErrorSeverity.WARNING,
                        f"Phi node '{var}' in '{blabel}' missing entries for predecessors: {missing}",
                        line_num,
                    ))

            # Check phi nodes appear before other instructions
            seen_non_phi = False
            for inst_type, defined, rhs, ln in block.instructions:
                if rhs and rhs.strip().startswith("phi"):
                    if seen_non_phi:
                        self.report.add_error(ValidationError(
                            ErrorCategory.SEMANTIC, ErrorSeverity.ERROR,
                            f"Phi node after non-phi instruction in block '{blabel}' of @{func.name}",
                            ln,
                        ))
                elif inst_type == "assign":
                    seen_non_phi = True

    def _validate_semantics(self):
        """Additional semantic checks."""
        for fname, func in self.functions.items():
            # Check that non-void functions have ret with value
            if func.return_type != "void":
                has_return_with_value = False
                for blabel, block in func.blocks.items():
                    if block.terminator and block.terminator.strip().startswith("ret"):
                        term = block.terminator.strip()
                        if term != "ret void" and len(term.split()) >= 3:
                            has_return_with_value = True
                if not has_return_with_value:
                    self.report.add_error(ValidationError(
                        ErrorCategory.SEMANTIC, ErrorSeverity.WARNING,
                        f"Function @{fname} returns '{func.return_type}' but no ret with value found",
                    ))

            # Check void functions don't return values
            if func.return_type == "void":
                for blabel, block in func.blocks.items():
                    if block.terminator:
                        term = block.terminator.strip()
                        if term.startswith("ret") and term != "ret void":
                            parts = term.split()
                            if len(parts) >= 3 and parts[1] != "void":
                                self.report.add_error(ValidationError(
                                    ErrorCategory.TYPE, ErrorSeverity.ERROR,
                                    f"Void function @{fname} returns a value in block '{blabel}'",
                                ))

    # ============================================================================
    # Comparison with ground truth
    # ============================================================================

    def compare_with_reference(self, reference_ir: str) -> dict:
        """Compare generated IR with reference IR structurally."""
        ref_validator = LLVMIRValidator(reference_ir)
        ref_validator._strip_code_fences()
        ref_validator._parse_structure()

        comparison = {
            "function_match": set(self.defined_funcs) == set(ref_validator.defined_funcs),
            "missing_functions": set(ref_validator.defined_funcs) - set(self.defined_funcs),
            "extra_functions": set(self.defined_funcs) - set(ref_validator.defined_funcs),
            "block_count_match": {},
            "instruction_patterns": {},
        }

        # Compare block counts per function
        for fname in ref_validator.functions:
            ref_blocks = len(ref_validator.functions[fname].blocks)
            gen_blocks = len(self.functions.get(fname, FunctionInfo(fname, "")).blocks)
            comparison["block_count_match"][fname] = {
                "reference": ref_blocks,
                "generated": gen_blocks,
                "match": ref_blocks == gen_blocks,
            }

        return comparison


def validate_ir(ir_text: str) -> ValidationReport:
    """Convenience function to validate LLVM IR text."""
    validator = LLVMIRValidator(ir_text)
    return validator.validate()


def detect_llvm_tools() -> dict:
    """Detect LLVM tools used for assembly/verification."""
    llvm_as = shutil.which("llvm-as")
    opt = shutil.which("opt")

    # If on Windows and native tools are not found, check WSL
    if os.name == 'nt' and (not llvm_as or not opt):
        try:
            res_as = subprocess.run(
                ["wsl", "llvm-as", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            res_opt = subprocess.run(
                ["wsl", "opt", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            if res_as.returncode == 0 and res_opt.returncode == 0:
                llvm_as = "wsl llvm-as"
                opt = "wsl opt"
        except Exception:
            pass

    return {
        "llvm_as": llvm_as,
        "opt": opt,
        "missing": [name for name, path in (("llvm-as", llvm_as), ("opt", opt)) if not path],
    }


def validate_ir_with_llvm_tools(
    ir_text: str,
    llvm_as_path: Optional[str] = None,
    opt_path: Optional[str] = None,
    timeout_s: int = 20,
) -> dict:
    """
    Validate LLVM IR using LLVM CLI tools:
      1) llvm-as (parse/assemble)
      2) opt verifier pass (supports old and new opt flag styles)
    """
    if llvm_as_path is None or opt_path is None:
        detected = detect_llvm_tools()
        llvm_as_path = llvm_as_path or detected["llvm_as"]
        opt_path = opt_path or detected["opt"]

    result = {
        "tools": {
            "llvm-as": llvm_as_path,
            "opt": opt_path,
        },
        "assembly_ok": False,
        "verify_ok": False,
        "is_valid": False,
        "commands": [],
        "stderr": [],
    }

    if not llvm_as_path or not opt_path:
        missing = []
        if not llvm_as_path:
            missing.append("llvm-as")
        if not opt_path:
            missing.append("opt")
        result["stderr"].append(f"Missing LLVM tools: {', '.join(missing)}")
        return result

    validator = LLVMIRValidator(ir_text)
    validator._strip_code_fences()
    normalized_ir = validator.ir_text

    def to_wsl_path(win_path):
        win_path = os.path.abspath(win_path)
        if len(win_path) >= 2 and win_path[1] == ':':
            drive = win_path[0].lower()
            rel_path = win_path[2:].replace('\\', '/')
            return f"/mnt/{drive}{rel_path}"
        return win_path.replace('\\', '/')

    is_wsl_llvm_as = llvm_as_path.startswith("wsl")
    is_wsl_opt = opt_path.startswith("wsl")

    with tempfile.TemporaryDirectory(prefix="llvm_ir_check_") as tmpdir:
        input_ll = os.path.join(tmpdir, "input.ll")
        output_bc = os.path.join(tmpdir, "output.bc")
        with open(input_ll, "w", encoding="utf-8") as f:
            f.write(normalized_ir)

        if is_wsl_llvm_as:
            wsl_input = to_wsl_path(input_ll)
            wsl_output = to_wsl_path(output_bc)
            llvm_as_parts = llvm_as_path.split()
            assemble_cmd = llvm_as_parts + [wsl_input, "-o", wsl_output]
        else:
            assemble_cmd = [llvm_as_path, input_ll, "-o", output_bc]

        result["commands"].append(" ".join(assemble_cmd))
        assemble_proc = subprocess.run(
            assemble_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if assemble_proc.returncode != 0:
            result["stderr"].append(assemble_proc.stderr.strip())
            return result
        result["assembly_ok"] = True

        if is_wsl_opt:
            wsl_output = to_wsl_path(output_bc)
            opt_parts = opt_path.split()
            verify_cmd_candidates = [
                opt_parts + ["-verify", "-disable-output", wsl_output],
                opt_parts + ["-disable-output", "-verify", wsl_output],
                opt_parts + ["-passes=verify", "-disable-output", wsl_output],
                opt_parts + ["-disable-output", "-passes=verify", wsl_output],
            ]
        else:
            verify_cmd_candidates = [
                [opt_path, "-verify", "-disable-output", output_bc],
                [opt_path, "-disable-output", "-verify", output_bc],
                [opt_path, "-passes=verify", "-disable-output", output_bc],
                [opt_path, "-disable-output", "-passes=verify", output_bc],
            ]

        compatibility_markers = (
            "unknown command line argument",
            "unknown argument",
            "unknown pass name",
            "for the --passes option",
            "did you mean",
            "syntax for the new pass manager is not supported",
            "please use `opt -passes=",
        )

        last_error = ""
        for verify_cmd in verify_cmd_candidates:
            result["commands"].append(" ".join(verify_cmd))
            verify_proc = subprocess.run(
                verify_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            if verify_proc.returncode == 0:
                result["verify_ok"] = True
                result["is_valid"] = True
                return result

            stderr = (verify_proc.stderr or "").strip()
            stdout = (verify_proc.stdout or "").strip()
            message = stderr or stdout or f"opt exited with code {verify_proc.returncode}"
            last_error = message

            if not any(marker in message.lower() for marker in compatibility_markers):
                result["stderr"].append(message)
                return result

        result["stderr"].append(last_error)
        return result


def validate_and_compare(generated_ir: str, reference_ir: str) -> dict:
    """Validate generated IR and compare with reference."""
    validator = LLVMIRValidator(generated_ir)
    report = validator.validate()
    comparison = validator.compare_with_reference(reference_ir)
    return {
        "validation": report.to_dict(),
        "comparison": comparison,
        "summary": report.summary(),
    }


if __name__ == "__main__":
    # Test with a sample IR
    test_ir = """
    define i32 @add(i32 %a, i32 %b) {
    entry:
      %result = add i32 %a, %b
      ret i32 %result
    }
    """
    report = validate_ir(test_ir)
    print(report.summary())
