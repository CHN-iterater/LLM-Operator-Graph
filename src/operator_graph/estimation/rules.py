"""Versioned static arithmetic and logical payload estimation rules."""
from __future__ import annotations

import math
from itertools import zip_longest

COST_MODEL_VERSION = "static-v2.0"
_BYTES = {"bool": 1, "int8": 1, "uint8": 1, "int16": 2, "uint16": 2,
          "int32": 4, "uint32": 4, "int64": 8, "uint64": 8,
          "float16": 2, "half": 2, "bfloat16": 2, "float32": 4,
          "float": 4, "float64": 8, "double": 8}
_FLOATS = {"float16", "half", "bfloat16", "float32", "float", "float64", "double"}
_VIEWS = {"RESHAPE", "VIEW", "SQUEEZE", "UNSQUEEZE", "EXPAND", "DETACH", "IDENTITY"}
_ZERO = _VIEWS | {"TRANSPOSE", "PERMUTE", "SLICE", "SPLIT", "CONCAT", "CAT",
    "CLONE", "CONTIGUOUS", "SELECT", "UNFLATTEN", "GATHER", "GATHERELEMENTS",
    "EMBEDDING", "CAST", "WHERE", "EQUAL", "LESS", "GREATER", "LESSOREQUAL",
    "GREATEROREQUAL", "ISNAN", "ISINF", "AND", "OR", "NOT", "SHAPE", "SIZE",
    "CONSTANT", "CONSTANTOFSHAPE", "DROPOUT"}
_ELEMENT = {"ADD", "SUB", "MUL", "DIV", "NEG", "ABS", "RECIPROCAL"}
_SPECIAL = {"POW", "SQRT", "RSQRT", "EXP", "LOG", "ERF", "TANH"}
_ALIASES = {"GEMM": "MATMUL", "LINEAR": "LINEAR", "BMM": "MATMUL",
            "MEAN": "REDUCEMEAN", "LAYER_NORM": "LAYERNORMALIZATION",
            "RMS_NORM": "RMSNORMALIZATION"}

def _dtype(t):
    return str(t.get("dtype", "unknown")).removeprefix("torch.").lower()


def _numel(t):
    shape = t.get("shape")
    if not isinstance(shape, (list, tuple)):
        return None
    if any(type(x) is not int or x < 0 for x in shape):
        return None
    # [] with a known dtype denotes a scalar; old unknown-rank placeholders
    # have dtype=unknown and are not silently treated as one element.
    if not shape and _dtype(t) not in _BYTES:
        return None
    return math.prod(shape)


def _tensor_bytes(t):
    n, b = _numel(t), _BYTES.get(_dtype(t))
    return n * b if n is not None and b is not None else None


def _matmul_shape(a, b):
    """ONNX/NumPy MatMul vector promotion and batch broadcasting."""
    if not a or not b:
        return None
    left_vector, right_vector = len(a) == 1, len(b) == 1
    a = [1, *a] if left_vector else list(a)
    b = [*b, 1] if right_vector else list(b)
    if a[-1] != b[-2]:
        return None
    batch = []
    for x, y in zip_longest(reversed(a[:-2]), reversed(b[:-2]), fillvalue=1):
        if x != y and x != 1 and y != 1:
            return None
        batch.append(y if x == 1 else x)
    result = list(reversed(batch)) + [a[-2], b[-1]]
    if right_vector:
        result.pop()
    if left_vector:
        result.pop(-1 if right_vector else -2)
    return result


def _estimate_builtin(node):
    """Estimate floating scalar operations and logical bytes with provenance.

    FMA=2; special functions=1 scalar-op equivalent (approximate). Integer,
    comparison, indexing and conversion work is excluded from FLOPs. Norm,
    activation and softmax formulas are explicitly approximate, not instructions.
    """
    raw = node.get("metadata", {}).get("onnx_op_type") or node.get("raw_target") or node.get("op_type", "")
    op = str(raw).upper()
    op = _ALIASES.get(op, op)
    ins, outs = node.get("input_tensors", []), node.get("output_tensors", [])
    attrs = node.get("metadata", {}).get("onnx_attributes", {})
    sizes = [_tensor_bytes(t) for t in ins + outs]
    declared = sum(sizes) if sizes and all(s is not None for s in sizes) else None
    io, io_rule = declared, "read each declared input once + write each output once"
    if op in _VIEWS:
        io, io_rule = 0, "view/alias assumption: no payload materialization"
    elif op in {"GATHER", "GATHERELEMENTS", "EMBEDDING"}:
        output_sizes = [_tensor_bytes(t) for t in outs]
        index_size = _tensor_bytes(ins[1]) if len(ins) >= 2 else None
        io = (2 * sum(output_sizes) + index_size
              if output_sizes and all(s is not None for s in output_sizes) and index_size is not None else None)
        io_rule = "read selected payload + indices + write output (no whole-table read)"

    f, status, formula = None, "unsupported", "no FLOP rule"
    out_n = _numel(outs[0]) if outs else None
    in_n = _numel(ins[0]) if ins else None
    if op in _ZERO:
        f, status, formula = 0, "known_zero", "zero floating arithmetic; other work may exist"
    elif not outs or not ins or out_n is None or in_n is None:
        status, formula = "unknown_shape", "required tensor shape unavailable"
    elif _dtype(outs[0]) not in _BYTES:
        status, formula = "unknown_dtype", "output dtype unavailable"
    elif _dtype(outs[0]) not in _FLOATS:
        f, status, formula = 0, "known_zero", "non-floating output; integer operations excluded"
    elif op in {"MATMUL", "GEMM", "LINEAR"}:
        shape = ins[0].get("shape", [])
        trans_a = attrs.get("transA", 0) if str(raw).upper() == "GEMM" else 0
        k = shape[-2 if trans_a else -1] if len(shape) >= (2 if trans_a else 1) else None
        valid_shape = len(ins) >= 2 and _numel(ins[1]) is not None
        raw_upper = str(raw).upper()
        if valid_shape and raw_upper in {"MATMUL", "BMM"}:
            valid_shape = _matmul_shape(shape, ins[1]["shape"]) == list(outs[0]["shape"])
        elif valid_shape and raw_upper == "GEMM" and "onnx_attributes" in node.get("metadata", {}):
            a, b = list(shape), list(ins[1]["shape"])
            if len(a) != 2 or len(b) != 2:
                valid_shape = False
            else:
                if trans_a:
                    a.reverse()
                if attrs.get("transB", 0):
                    b.reverse()
                valid_shape = _matmul_shape(a, b) == list(outs[0]["shape"])
        if k is not None and k >= 0 and valid_shape:
            f, status, formula = 2 * out_n * k, "exact", "2 * output_numel * K (all batch/broadcast dimensions)"
            if str(raw).upper() in {"GEMM", "LINEAR"}:
                alpha, beta = attrs.get("alpha", 1), attrs.get("beta", 1)
                if len(ins) >= 3 and beta != 0:
                    f += out_n
                    formula += " + bias addition"
                    if beta != 1:
                        f += out_n
                if alpha != 1:
                    f += out_n
                if raw_upper == "GEMM" and "onnx_attributes" not in node.get("metadata", {}):
                    status = "approximate"
                    formula += "; archive lacks attributes: assume transA=0, alpha=beta=1"
        else:
            status, formula = "invalid_shape", "missing/incompatible matrix operands or output shape"
    elif op in _ELEMENT | _SPECIAL:
        f, status, formula = out_n, "exact", "one scalar operation per output element"
        if op in _SPECIAL:
            status, formula = "approximate", "special function = one scalar-op equivalent per output"
    elif op in {"REDUCEMEAN", "REDUCESUM"}:
        if in_n >= out_n:
            f = in_n if op == "REDUCEMEAN" else in_n - out_n
            status, formula = "exact", "input_numel - output_numel additions; mean adds output_numel divisions"
        else:
            status, formula = "unknown_shape", "reduction input smaller than output"
    elif op in {"LAYERNORMALIZATION", "RMSNORMALIZATION", "SIMPLIFIEDLAYERNORMALIZATION",
                "SOFTMAX", "SIGMOID", "RELU", "SILU", "GELU", "FASTGELU"}:
        factor = {"LAYERNORMALIZATION": 7, "RMSNORMALIZATION": 4,
                  "SIMPLIFIEDLAYERNORMALIZATION": 4, "SOFTMAX": 5,
                  "SIGMOID": 4, "RELU": 1, "SILU": 5, "GELU": 8, "FASTGELU": 8}[op]
        f, status, formula = factor * out_n, "approximate", f"{factor} * output_numel scalar-op equivalents"
    ai = f / io if f is not None and io is not None and io > 0 else None
    return {"cost_model": COST_MODEL_VERSION, "flops": f, "flops_status": status,
            "flops_formula": formula, "declared_tensor_bytes": declared,
            "logical_io_bytes": io, "io_assumption": io_rule,
            "arithmetic_intensity": ai}


def estimate_node_cost(node):
    """Apply the preserved rules only within the supported ONNX namespace.

    Domain-less legacy archives use the standard-domain convention. Custom
    domains require an explicit rule implementation rather than name matching.
    """
    result = _estimate_builtin(node)
    domain = node.get('metadata', {}).get('domain') or ''
    if domain not in ('', 'ai.onnx'):
        result.update(flops=None, flops_status='unsupported_domain',
                      flops_formula='operator domain has no registered arithmetic rule',
                      logical_io_bytes=None, io_assumption='operator domain has no registered access rule',
                      arithmetic_intensity=None)
    return result


