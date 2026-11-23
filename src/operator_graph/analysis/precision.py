"""Declared output payloads and dtype conversion summaries."""
from __future__ import annotations

from collections import defaultdict
from operator_graph.estimation.rules import _dtype, _tensor_bytes, _FLOATS, _BYTES
from .aggregation import _aggregate

def _precision(nodes, rows, stage_bytes):
    transitions, dtype_groups = defaultdict(list), defaultdict(list)
    for node, row in zip(nodes, rows):
        if str(row["raw_op_type"]).upper() == "CAST":
            key = (tuple(_dtype(t) for t in node.get("input_tensors", [])),
                   tuple(_dtype(t) for t in node.get("output_tensors", [])))
            transitions[key].append(row)
        for t in node.get("output_tensors", []):
            dtype_groups[_dtype(t)].append(_tensor_bytes(t))
    cast_rows = [{"stage": rows[0]["stage"], "input_dtypes": list(i), "output_dtypes": list(o), **_aggregate(group)}
                 for (i, o), group in sorted(transitions.items())]
    dtype_rows = []
    for dtype, sizes in sorted(dtype_groups.items()):
        known = [b for b in sizes if b is not None]
        dtype_rows.append({"stage": rows[0]["stage"], "dtype": dtype, "num_output_tensors": len(sizes),
                           "output_payload_bytes": sum(known) if len(known) == len(sizes) else None,
                           "known_output_payload_bytes": sum(known), "payload_coverage": len(known)/len(sizes)})
    cast_count = sum(r["num_nodes"] for r in cast_rows)
    cast_bytes = (sum(r["logical_io_bytes"] for r in cast_rows)
                  if all(r["logical_io_bytes"] is not None for r in cast_rows) else None)
    float_sizes = [b for dtype, sizes in dtype_groups.items() if dtype in _FLOATS for b in sizes]
    float_known = (all(dtype in _BYTES for dtype in dtype_groups)
                   and all(b is not None for b in float_sizes))
    float_bytes = sum(float_sizes) if float_known else None
    fp32_bytes = sum(b for dtype, sizes in dtype_groups.items() if dtype in {"float", "float32"} for b in sizes) if float_known else None
    return {"cast_count": cast_count, "cast_node_fraction": cast_count/len(rows) if rows else None,
            "cast_logical_io_bytes": cast_bytes,
            "cast_logical_io_fraction": cast_bytes/stage_bytes if cast_bytes is not None and stage_bytes else None,
            "floating_output_payload_bytes": float_bytes,
            "fp32_floating_output_payload_fraction": fp32_bytes/float_bytes if float_bytes else None}, cast_rows, dtype_rows
