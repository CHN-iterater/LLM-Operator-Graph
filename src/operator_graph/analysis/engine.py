"""Analyze an archived graph without trusting archived arithmetic estimates."""
from __future__ import annotations

from collections import Counter, defaultdict
from operator_graph.estimation.rules import COST_MODEL_VERSION, _dtype, _numel, _BYTES, estimate_node_cost
from .aggregation import _aggregate
from .topology import _topology
from .workloads import _workloads
from .precision import _precision

def analyze_graph(data):
    """Recompute costs without trusting archived flops, memory or AI fields."""
    nodes = data.get("nodes", [])
    by_id = {n["op_id"]: n for n in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("Duplicate op_id: cannot compute unambiguous graph metrics")
    integrity = Counter(dangling_references=0, reciprocity_errors=0, duplicate_edges=0, cross_stage_edges=0)
    for n in nodes:
        for direction, inverse in (("parents", "children"), ("children", "parents")):
            refs = n.get(direction, [])
            integrity["duplicate_edges"] += len(refs)-len(set(refs))
            for ref in set(refs):
                if ref not in by_id:
                    integrity["dangling_references"] += 1
                elif n["op_id"] not in by_id[ref].get(inverse, []):
                    integrity["reciprocity_errors"] += 1
                if direction == "children" and ref in by_id and n.get("stage", "unknown") != by_id[ref].get("stage", "unknown"):
                    integrity["cross_stage_edges"] += 1
    # Stage-local paths assume independent exported forwards. A cross-stage
    # dependency violates that scope, even if each induced subgraph is acyclic.
    malformed = any(integrity.values())
    stage_nodes, cost_rows = defaultdict(list), []
    for n in nodes:
        stage = n.get("stage", "unknown")
        stage_nodes[stage].append(n)
        cost_rows.append({"op_id": n["op_id"], "op_type": n.get("op_type", ""),
            "raw_op_type": n.get("metadata", {}).get("onnx_op_type") or n.get("raw_target") or n.get("op_type", ""),
            "stage": stage, "layer_id": n.get("layer_id", ""), **estimate_node_cost(n)})
    stages, operator_rows, layer_rows = {}, [], []
    shape_rows, cast_rows, dtype_rows = [], [], []
    for stage, ns in sorted(stage_nodes.items()):
        rows = [r for r in cost_rows if r["stage"] == stage]
        stages[stage] = {**_aggregate(rows), **_topology(ns, malformed, rows)}
        workload_stats, workloads = _workloads(ns, rows)
        precision_stats, casts, dtypes = _precision(ns, rows, stages[stage]["logical_io_bytes"])
        stages[stage].update(workload_stats)
        stages[stage].update(precision_stats)
        shape_rows.extend(workloads)
        cast_rows.extend(casts)
        dtype_rows.extend(dtypes)
        tensors = [t for n in ns for t in n.get("output_tensors", [])]
        stages[stage]["output_tensor_count"] = len(tensors)
        stages[stage]["known_output_shape_count"] = sum(_numel(t) is not None for t in tensors)
        stages[stage]["known_output_dtype_count"] = sum(_dtype(t) in _BYTES for t in tensors)
        stages[stage]["output_shape_coverage"] = stages[stage]["known_output_shape_count"]/len(tensors) if tensors else None
        stages[stage]["output_dtype_coverage"] = stages[stage]["known_output_dtype_count"]/len(tensors) if tensors else None
        groups = defaultdict(list)
        for row in rows:
            groups[row["raw_op_type"]].append(row)
        stages[stage]["num_operator_types"] = len(groups)
        for op, group in sorted(groups.items()):
            agg = _aggregate(group)
            stage_f, stage_b = stages[stage]["estimated_flops"], stages[stage]["logical_io_bytes"]
            operator_rows.append({"stage": stage, "operator_type": op, **agg,
                "node_fraction": len(group)/len(rows),
                "estimated_flop_fraction": agg["estimated_flops"]/stage_f if stage_f and agg["estimated_flops"] is not None else None,
                "logical_io_fraction": agg["logical_io_bytes"]/stage_b if stage_b and agg["logical_io_bytes"] is not None else None})
        layers = defaultdict(list)
        for row in rows:
            layers[row["layer_id"]].append(row)
        for label, group in sorted(layers.items()):
            layer_rows.append({"stage": stage, "layer_id": label,
                               "annotation_status": "unvalidated labels", **_aggregate(group)})
    return {"report_version": "1.1", "cost_model": COST_MODEL_VERSION,
            "model_name": data.get("model_name", ""), "prompt": data.get("prompt", {}),
            "execution": data.get("execution", {}), "integrity": dict(integrity),
            "assumptions": ["Single exported forward graph per stage; no multiplication by generated tokens.",
                "FMA=2. Special functions and fused norm/activation costs use documented scalar-op approximations.",
                "Logical I/O: unfused read-once/write-once payload model; view operators alias; transpose/slice materialize.",
                "Gather reads selected payload, indices, and writes output. No cache, fusion or DRAM transaction model.",
                "Unknown FLOPs/shape/dtype yield null, not zero; paired AI uses the same known nodes in numerator/denominator.",
                "Per-stage AI is sum(FLOPs)/sum(bytes), not a mean of operator AIs. It is not measured hardware intensity.",
                "DAG widths/path lengths describe logical operator dependencies, not GPU kernel concurrency.",
                "Shape signature v1: domain + raw operator + ordered input/output shapes and dtypes. Attributes, values and layouts are excluded.",
                "Reuse counts only complete signatures; reuse is geometric workload grouping, not semantic/kernel equivalence or execution speedup.",
                "Top-10% shares select ceil(0.1*N) nodes, independently ranked by FLOPs and logical I/O; incomplete or zero totals yield null.",
                "Weighted FLOP span is max path sum of FLOPs, with genuine zero weights. Work/span describes FLOP distribution, not runtime speedup.",
                "Cast I/O follows the logical payload model. Output dtype payloads count every declared output, including aliases; they are not live/peak memory.",
                "One-producer/one-consumer edges and fanout describe dependencies; they do not establish fusibility or cache reuse."],
            "stages": stages, "operators": operator_rows, "layers": layer_rows, "nodes": cost_rows,
            "shape_signatures": shape_rows, "cast_transitions": cast_rows, "output_dtypes": dtype_rows}
