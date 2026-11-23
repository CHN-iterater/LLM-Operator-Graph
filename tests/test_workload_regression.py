"""Offline workload metrics checked against hand-computed graphs."""
import csv
import json
from copy import deepcopy

import pytest

from operator_graph.analysis.engine import analyze_graph
from operator_graph.reports.batch import main
from operator_graph.reports.writer import write_reports


def tensor(size=4, dtype="float16"):
    return {"shape": [size], "dtype": dtype}


def node(name, op="Mul", size=4, dtype="float16", **extra):
    t = tensor(size, dtype)
    return {"op_id": name, "op_type": op, "raw_target": op, "stage": "prefill",
            "parents": [], "children": [], "input_tensors": [t, t],
            "output_tensors": [t], **extra}


def test_shape_groups_ignore_names_but_preserve_dtype_domain_and_stage():
    nodes = [node("a"), node("b"), node("c", dtype="float32"),
             node("d", metadata={"domain": "custom"}), node("e", stage="decode")]
    original = deepcopy(nodes)
    r = analyze_graph({"nodes": nodes})
    s = r["stages"]["prefill"]
    assert s["num_shape_signatures"] == 3
    assert s["shape_signature_reuse_factor"] == pytest.approx(4/3)
    assert s["shape_signature_coverage"] == 1
    groups = [g for g in r["shape_signatures"] if g["stage"] == "prefill"]
    assert sorted(g["num_nodes"] for g in groups) == [1, 1, 2]
    assert len({g["signature_id"] for g in r["shape_signatures"]}) == 3
    assert nodes == original


def test_unknown_shapes_are_reported_but_excluded_from_reuse():
    r = analyze_graph({"nodes": [node("a"), node("b", size="N"), node("c", size="N")]})
    s = r["stages"]["prefill"]
    assert s["num_shape_signatures"] == 1
    assert s["shape_signature_node_count"] == 1
    assert s["shape_signature_coverage"] == pytest.approx(1/3)
    assert s["shape_signature_reuse_factor"] == 1
    assert sum(g["num_nodes"] for g in r["shape_signatures"]) == 3


def test_matrix_shape_count_includes_gemm_and_batched_matmul():
    a = node("a", "MatMul", input_tensors=[tensor(4), tensor(4)],
             output_tensors=[{"shape": [], "dtype": "float16"}])
    b = {**a, "op_id": "b"}
    s = analyze_graph({"nodes": [a, b, node("c", "Gemm")]})["stages"]["prefill"]
    assert s["matrix_call_count"] == 3
    assert s["matrix_shape_signature_count"] == 2


def test_weighted_fork_join_span_uses_true_zero_cost():
    nodes = [node("a", "Identity", parents=[], children=["b", "c"]),
             node("b", size=100, parents=["a"], children=["d"]),
             node("c", size=20, parents=["a"], children=["d"]),
             node("d", "Identity", parents=["b", "c"])]
    s = analyze_graph({"nodes": nodes})["stages"]["prefill"]
    assert s["estimated_flops"] == 120
    assert s["weighted_flop_span"] == 100
    assert s["flop_work_over_span"] == 1.2
    assert s["single_producer_consumer_edge_fraction"] == 0
    assert s["fanout_node_fraction"] == .25
    assert s["max_fanout"] == 2


@pytest.mark.parametrize("kind", ["cycle", "unknown", "zero", "dangling"])
def test_span_and_concentration_do_not_invent_values(kind):
    a = node("a", "Custom" if kind == "unknown" else "Identity")
    if kind == "cycle":
        a.update(parents=["a"], children=["a"])
    if kind == "dangling":
        a["children"] = ["absent"]
    s = analyze_graph({"nodes": [a]})["stages"]["prefill"]
    assert s["flop_work_over_span"] is None
    assert s["weighted_flop_span"] == (0 if kind == "zero" else None)
    assert s["top10pct_node_flop_share"] is None
    if kind in ("cycle", "dangling"):
        assert s["single_producer_consumer_edge_fraction"] is None
        assert s["fanout_node_fraction"] is None


def test_top_ten_percent_rounds_up_and_sorts_costs_separately():
    nodes = [node(str(i), size=100 if i == 0 else 1) for i in range(11)]
    s = analyze_graph({"nodes": nodes})["stages"]["prefill"]
    assert s["top10pct_node_count"] == 2
    assert s["top10pct_node_flop_share"] == pytest.approx(101/110)
    assert s["top10pct_node_io_share"] == pytest.approx(101/110)


def test_cast_transitions_and_output_payload_are_not_live_memory():
    a = node("a", "Cast", input_tensors=[tensor(4, "bfloat16")],
             output_tensors=[tensor(4, "float32")])
    b = node("b", "Cast", input_tensors=[tensor(4, "float32")],
             output_tensors=[tensor(4, "bfloat16")])
    r = analyze_graph({"nodes": [a, b, node("c", dtype="bfloat16")]})
    s = r["stages"]["prefill"]
    assert s["cast_count"] == 2
    assert s["cast_node_fraction"] == pytest.approx(2/3)
    assert s["cast_logical_io_bytes"] == 48
    assert s["cast_logical_io_fraction"] == pytest.approx(48/72)
    assert s["floating_output_payload_bytes"] == 32
    assert s["fp32_floating_output_payload_fraction"] == .5
    assert len(r["cast_transitions"]) == 2
    assert {g["dtype"]: g["output_payload_bytes"] for g in r["output_dtypes"]} == {
        "float32": 16, "bfloat16": 16}


def test_unknown_payload_does_not_inflate_fp32_fraction():
    r = analyze_graph({"nodes": [node("a", dtype="float32"), node("b", dtype="unknown")]})
    s = r["stages"]["prefill"]
    assert s["floating_output_payload_bytes"] is None
    assert s["fp32_floating_output_payload_fraction"] is None
    assert next(g for g in r["output_dtypes"] if g["dtype"] == "unknown")["output_payload_bytes"] is None


def test_cross_stage_edges_do_not_create_independent_forward_span_claims():
    a = node("a", children=["b"])
    b = node("b", parents=["a"], stage="decode")
    r = analyze_graph({"nodes": [a, b]})
    assert r["integrity"]["cross_stage_edges"] == 1
    assert all(s["weighted_flop_span"] is None for s in r["stages"].values())


def test_reports_and_joint_shape_dictionary_preserve_inputs(tmp_path):
    source = tmp_path / "input"
    output = tmp_path / "output"
    original = []
    for name in ("one", "two"):
        folder = source / name
        folder.mkdir(parents=True)
        path = folder / "graph.json"
        path.write_text(json.dumps({"model_name": name, "nodes": [node("a"), node("b")]}), encoding="utf-8")
        original.append((path, path.read_bytes()))
    assert main([str(source), "--output", str(output)]) == 0
    for path, before in original:
        assert path.read_bytes() == before
    for filename in ("shape_signature_metrics.csv", "cast_metrics.csv", "dtype_metrics.csv"):
        assert (output / "one" / filename).exists()
    with (output / "shape_signature_catalog.csv").open(encoding="utf-8-sig", newline="") as f:
        catalog = list(csv.DictReader(f))
    assert len(catalog) == 1
    assert int(catalog[0]["num_nodes"]) == 4
    assert int(catalog[0]["num_models"]) == 2
    r = json.loads((output / "one" / "metrics.json").read_text(encoding="utf-8"))
    assert r["report_version"] == "1.1"
    assert "Shape reuse" in (output / "one" / "metrics.md").read_text(encoding="utf-8")


def test_empty_graph_is_serializable(tmp_path):
    r = analyze_graph({"nodes": []})
    write_reports(r, tmp_path)
    assert r["stages"] == {}
    assert r["shape_signatures"] == []
