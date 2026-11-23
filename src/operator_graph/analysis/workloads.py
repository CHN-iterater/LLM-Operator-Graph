"""Geometric workload descriptors and concentration statistics."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from operator_graph.estimation.rules import _dtype, _tensor_bytes
from .aggregation import _aggregate

def _shape_descriptor(node):
    """A geometric workload key, deliberately excluding attributes and values."""
    def specs(key):
        return [{"shape": t.get("shape"), "dtype": _dtype(t)} for t in node.get(key, [])]
    return {"domain": node.get("metadata", {}).get("domain") or "",
            "raw_op_type": node.get("metadata", {}).get("onnx_op_type") or node.get("raw_target") or node.get("op_type", ""),
            "input_specs": specs("input_tensors"), "output_specs": specs("output_tensors")}


def _workloads(nodes, rows):
    groups, descriptors = defaultdict(list), {}
    for node, row in zip(nodes, rows):
        descriptor = _shape_descriptor(node)
        canonical = json.dumps(descriptor, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        row["shape_signature_id"] = signature
        groups[signature].append(row)
        descriptors[signature] = descriptor
    result = []
    for signature, group in sorted(groups.items()):
        descriptor = descriptors[signature]
        tensors = descriptor["input_specs"] + descriptor["output_specs"]
        complete = bool(descriptor["raw_op_type"] and descriptor["output_specs"]) and all(
            _tensor_bytes(t) is not None for t in tensors)
        result.append({"stage": rows[0]["stage"], "signature_id": signature, **descriptor,
                       "signature_complete": complete, **_aggregate(group)})
    complete = [g for g in result if g["signature_complete"]]
    complete_nodes = sum(g["num_nodes"] for g in complete)
    matrix = lambda g: str(g["raw_op_type"]).upper() in {"MATMUL", "GEMM", "BMM", "LINEAR"}
    k = math.ceil(len(rows)*.1)
    def share(key):
        values = [r[key] for r in rows]
        if not values or any(v is None for v in values):
            return None
        total = sum(values)
        return sum(sorted(values, reverse=True)[:k])/total if total else None
    stats = {"num_shape_signatures": len(complete), "shape_signature_node_count": complete_nodes,
             "shape_signature_coverage": complete_nodes/len(rows) if rows else None,
             "shape_signature_reuse_factor": complete_nodes/len(complete) if complete else None,
             "matrix_call_count": sum(g["num_nodes"] for g in result if matrix(g)),
             "matrix_shape_signature_count": sum(matrix(g) for g in complete),
             "top10pct_node_count": k, "top10pct_node_flop_share": share("flops"),
             "top10pct_node_io_share": share("logical_io_bytes")}
    return stats, result


