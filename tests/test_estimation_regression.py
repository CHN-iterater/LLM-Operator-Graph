"""Numerical references for the static cost model and report aggregation."""
import pytest

from operator_graph.estimation.rules import estimate_node_cost
from operator_graph.analysis.engine import analyze_graph
from operator_graph.reports.writer import write_reports


def tensor(shape, dtype="float16"):
    return {"shape": list(shape), "dtype": dtype}


def node(op, inputs, outputs, **kwargs):
    return {"op_id": "a", "op_type": op, "raw_target": op,
            "input_tensors": inputs, "output_tensors": outputs,
            "parents": [], "children": [], "stage": "prefill", **kwargs}


def test_broadcast_batched_matmul_and_io():
    n = node("MatMul", [tensor((2, 4, 8, 64)), tensor((1, 4, 64, 128))],
             [tensor((2, 4, 8, 128))])
    c = estimate_node_cost(n)
    assert c["flops"] == 2 * 2 * 4 * 8 * 64 * 128
    assert c["logical_io_bytes"] == (2*4*8*64 + 4*64*128 + 2*4*8*128)*2
    assert c["arithmetic_intensity"] == c["flops"] / c["logical_io_bytes"]


def test_broadcast_elementwise_counts_output_and_integer_is_not_flops():
    n = node("Mul", [tensor(()), tensor((2, 8, 16))], [tensor((2, 8, 16))])
    assert estimate_node_cost(n)["flops"] == 256
    n = node("Add", [tensor((8,), "int64")], [tensor((8,), "int64")])
    assert estimate_node_cost(n)["flops"] == 0


def test_unknown_costs_are_not_zero_or_fake_dynamic_dimensions():
    assert estimate_node_cost(node("CustomOp", [tensor((4,))], [tensor((4,))]))["flops"] is None
    c = estimate_node_cost(node("Mul", [tensor((-1, 8))], [tensor((-1, 8))]))
    assert c["flops"] is None
    assert c["logical_io_bytes"] is None
    c = estimate_node_cost(node("Mul", [tensor((8,), "unknown")], [tensor((8,), "unknown")]))
    assert c["arithmetic_intensity"] is None


def test_gather_reads_selected_rows_not_entire_embedding_table():
    c = estimate_node_cost(node("Gather", [tensor((50000, 64)), tensor((1, 8), "int64")],
                                [tensor((1, 8, 64))]))
    assert c["flops"] == 0
    assert c["logical_io_bytes"] == 2 * 1*8*64*2 + 1*8*8


def test_gemm_transpose_and_bias_and_scalar_dtype_size():
    c = estimate_node_cost(node("Gemm", [tensor((5, 3)), tensor((7, 5)), tensor((7,))],
        [tensor((3, 7))], metadata={"onnx_attributes": {"transA": 1, "transB": 1}}))
    assert c["flops"] == 2*3*5*7 + 3*7
    c = estimate_node_cost(node("Where", [tensor((), "bool"), tensor((4,)), tensor((4,))], [tensor((4,))]))
    assert c["logical_io_bytes"] == 1 + 3*4*2


def test_reduction_and_view_assumptions_are_explicit():
    c = estimate_node_cost(node("ReduceMean", [tensor((2, 8))], [tensor((2, 1))]))
    assert c["flops"] == 16
    c = estimate_node_cost(node("Reshape", [tensor((2, 8))], [tensor((16,))]))
    assert c["logical_io_bytes"] == 0
    assert c["arithmetic_intensity"] is None


def test_stage_aggregation_ratio_coverage_and_dag(tmp_path):
    a = node("Mul", [tensor((4,)), tensor((4,))], [tensor((4,))], children=["b"])
    b = node("CustomOp", [tensor((4,))], [tensor((4,))], op_id="b", parents=["a"])
    c = node("Add", [tensor((1,)), tensor((1,))], [tensor((1,))], op_id="dc", stage="decode")
    r = analyze_graph({"model_name": "test", "nodes": [a, b, c]})
    pf = r["stages"]["prefill"]
    assert pf["num_nodes"] == 2 and pf["num_edges"] == 1
    assert pf["critical_path_nodes"] == 2 and pf["mean_level_width"] == 1
    assert pf["flop_model_coverage"] == .5
    assert pf["arithmetic_intensity"] is None
    assert pf["modeled_arithmetic_intensity"] == 4/24
    assert r["stages"]["decode"]["num_nodes"] == 1
    write_reports(r, tmp_path)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "operator_metrics.csv").exists()


def test_inconsistent_and_cyclic_graphs_do_not_get_plausible_path_lengths():
    a = node("Mul", [], [], parents=["b"], children=["b"])
    b = node("Mul", [], [], op_id="b", parents=["a"], children=["a"])
    r = analyze_graph({"nodes": [a, b]})
    assert r["stages"]["prefill"]["is_dag"] is False
    assert r["stages"]["prefill"]["critical_path_nodes"] is None
    a["children"] = ["absent"]
    r = analyze_graph({"nodes": [a, b]})
    assert r["integrity"]["dangling_references"] > 0
    assert r["integrity"]["reciprocity_errors"] > 0




def test_matmul_rejects_contraction_or_output_shape_mismatch():
    bad = node("MatMul", [tensor((2, 3)), tensor((5, 4))], [tensor((2, 4))])
    assert estimate_node_cost(bad)["flops"] is None
    bad = node("MatMul", [tensor((2, 3)), tensor((3, 4))], [tensor((2, 9))])
    assert estimate_node_cost(bad)["flops"] is None


def test_complete_graph_ai_is_ratio_of_sums():
    a = node("Mul", [tensor((4,)), tensor((4,))], [tensor((4,))])
    b = node("MatMul", [tensor((4, 8)), tensor((8, 16))], [tensor((4, 16))], op_id="b")
    r = analyze_graph({"nodes": [a, b]})["stages"]["prefill"]
    assert r["arithmetic_intensity"] == pytest.approx((4+1024)/(24+448))


