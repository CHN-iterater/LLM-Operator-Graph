"""Exercise the graph backend using archived JSON, without weights or torch.

Replay and internal consistency tests are not validation against source ONNX,
model modules, numerically executed logits, or hardware measurements.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from copy import deepcopy
from dataclasses import fields

from operator_graph.model.graph import ComputationGraph
from operator_graph.extensions.layers import LayerPartitioner
from operator_graph.model.records import OperatorNode, TensorMeta
from operator_graph.model.registry import OperatorRegistry
from operator_graph.serialization import graph_to_dict


def _restore_nodes(nodes, model_name):
    graph = ComputationGraph(model_name)
    allowed = {f.name for f in fields(OperatorNode)}
    for n in nodes:
        kwargs = {k: deepcopy(v) for k, v in n.items() if k in allowed}
        kwargs.setdefault("op_name", n["op_id"])
        for key in ("input_tensors", "output_tensors"):
            kwargs[key] = [TensorMeta(tuple(t["shape"]) if t.get("shape") is not None else None, t.get("dtype", "unknown"),
                                     t.get("device", "cpu")) for t in n.get(key, [])]
        graph.add_node(OperatorNode(**kwargs))
    return graph


def _reverse_flop_span(nodes, rows):
    """Independent sink-to-source reference for the source-to-sink report DP."""
    weights = {r["op_id"]: r["flops"] for r in rows}
    if not weights or any(w is None for w in weights.values()):
        return None
    pending = {n["op_id"]: len(n.get("children", [])) for n in nodes}
    parents = {n["op_id"]: n.get("parents", []) for n in nodes}
    remaining_span = dict.fromkeys(pending, 0)
    queue = deque(k for k, degree in pending.items() if degree == 0)
    visited = 0
    while queue:
        key = queue.popleft()
        visited += 1
        remaining_span[key] += weights[key]
        for parent in parents[key]:
            remaining_span[parent] = max(remaining_span[parent], remaining_span[key])
            pending[parent] -= 1
            if pending[parent] == 0:
                queue.append(parent)
    return max(remaining_span.values()) if visited == len(nodes) else None


def validate_replay(data, report):
    """Run structural roundtrip, cost aggregation, and layer-code replay checks."""
    nodes = data.get("nodes", [])
    checks = {
        "nonempty_graph": bool(nodes),
        "summary_node_count": data.get("summary", {}).get("num_nodes", len(nodes)) == len(nodes),
        "graph_integrity": not any(report["integrity"].values()),
        "all_stages_are_dags": all(s["is_dag"] is True for s in report["stages"].values()),
    }
    if "operator_counts" in data.get("summary", {}):
        checks["summary_operator_counts"] = dict(Counter(n["op_type"] for n in nodes)) == data["summary"]["operator_counts"]
    try:
        graph = _restore_nodes(nodes, data.get("model_name", ""))
        restored = graph_to_dict(graph, OperatorRegistry.get_default())["nodes"]
    except ValueError:
        checks['structural_serialization_roundtrip'] = False
        return {'passed': False, 'checks': checks, 'stages': {},
                'upstream_export_tested': False, 'source_fidelity_tested': False,
                'numeric_inference_tested': False, 'hardware_measurements_tested': False}
    # Cost fields intentionally change under a new estimator. Compare the
    # structural fields actually present in the archive, not old cost totals.
    keys = ("op_id", "op_type", "op_name", "stage", "layer_id", "parents", "children",
            "input_tensors", "output_tensors", "raw_target", "metadata")
    def normalized(value, key):
        if key in ('input_tensors', 'output_tensors'):
            return [{'shape': None, 'dtype': 'unknown', 'device': 'cpu', **t} for t in value]
        return value
    checks["structural_serialization_roundtrip"] = all(
        all(normalized(a[k], k) == normalized(b.get(k), k) for k in keys if k in a)
        for a, b in zip(nodes, restored))
    stages = {}
    for stage, stats in report["stages"].items():
        source_nodes = [n for n in nodes if n.get("stage", "unknown") == stage]
        rows = [r for r in report["nodes"] if r["stage"] == stage]
        known_f = sum(r["flops"] for r in rows if r["flops"] is not None)
        known_b = sum(r["logical_io_bytes"] for r in rows if r["logical_io_bytes"] is not None)
        checks[f"{stage}:node_cost_aggregation"] = known_f == stats["known_flops"] and known_b == stats["known_logical_io_bytes"]
        checks[f"{stage}:operator_counts"] = sum(r["num_nodes"] for r in report["operators"] if r["stage"] == stage) == stats["num_nodes"]
        checks[f"{stage}:layer_counts"] = sum(r["num_nodes"] for r in report["layers"] if r["stage"] == stage) == stats["num_nodes"]
        checks[f"{stage}:finite_nonnegative_costs"] = all(
            v is None or (math.isfinite(v) and v >= 0)
            for r in rows for v in (r["flops"], r["logical_io_bytes"], r["arithmetic_intensity"]))
        ai = stats["arithmetic_intensity"]
        checks[f"{stage}:intensity_ratio"] = (ai is None if stats["estimated_flops"] is None or not stats["logical_io_bytes"]
            else ai is not None and math.isclose(ai, stats["estimated_flops"]/stats["logical_io_bytes"], rel_tol=1e-12))
        groups = [g for g in report["shape_signatures"] if g["stage"] == stage]
        complete = [g for g in groups if g["signature_complete"]]
        observed = Counter(r["shape_signature_id"] for r in rows)
        checks[f"{stage}:shape_signature_partition"] = (
            observed == {g["signature_id"]: g["num_nodes"] for g in groups}
            and len(complete) == stats["num_shape_signatures"]
            and sum(g["num_nodes"] for g in complete) == stats["shape_signature_node_count"]
            and sum(g["known_flops"] for g in groups) == known_f
            and sum(g["known_logical_io_bytes"] for g in groups) == known_b)
        reference_span = _reverse_flop_span(source_nodes, rows) if checks["graph_integrity"] else None
        reference_ratio = known_f/reference_span if reference_span else None
        checks[f"{stage}:weighted_span_reference"] = (stats["weighted_flop_span"] == reference_span
                                                                    and stats["flop_work_over_span"] == reference_ratio)
        source_casts = sum(str(n.get("metadata", {}).get("onnx_op_type") or n.get("raw_target") or n.get("op_type", "")).upper() == "CAST"
                           for n in source_nodes)
        casts = [g for g in report["cast_transitions"] if g["stage"] == stage]
        source_dtypes = Counter(str(t.get("dtype", "unknown")).removeprefix("torch.").lower()
                                for n in source_nodes for t in n.get("output_tensors", []))
        checks[f"{stage}:precision_accounting"] = (
            source_casts == stats["cast_count"] == sum(g["num_nodes"] for g in casts)
            and source_dtypes == {g["dtype"]: g["num_output_tensors"] for g in report["output_dtypes"] if g["stage"] == stage})
        checks[f"{stage}:workload_fraction_ranges"] = all(
            v is None or (math.isfinite(v) and 0 <= v <= 1)
            for k, v in stats.items() if k.endswith(("_fraction", "_share", "_coverage")))
        if checks["graph_integrity"] and stats["is_dag"]:
            from operator_graph.parsers.onnx import annotate_graph as _annotate_flops
            sub = _restore_nodes(source_nodes, data.get("model_name", ""))
            _annotate_flops(sub)
            tree = LayerPartitioner(sub).partition()
            sub.set_layer_tree(tree)
            ids = tree.operator_ids()
            expected = {n.op_id for n in sub.nodes}
            checks[f"{stage}:layer_membership_coverage"] = ids == expected
            checks[f"{stage}:layer_unique_count"] = tree.num_ops() == len(expected)
            checks[f"{stage}:layer_flops_no_double_count"] = tree.total_flops(sub) == stats['estimated_flops']
            original_labels = {n["op_id"]: n.get("layer_id", "") for n in source_nodes}
            stages[stage] = {
                "unique_layer_operator_count": tree.num_ops(), "layer_total_flops": tree.total_flops(sub),
                "top_level_label_counts": dict(Counter(c.layer_type for c in tree.children)),
                "unchanged_node_label_count": sum(n.layer_id == original_labels[n.op_id] for n in sub.nodes),
                "num_nodes": len(expected),
                "interpretation": "Layer code replay and membership checks; no independent layer-boundary reference.",
            }
    return {"passed": all(checks.values()), "checks": checks, "stages": stages,
            "upstream_export_tested": False, "source_fidelity_tested": False,
            "numeric_inference_tested": False, "hardware_measurements_tested": False}
