from copy import deepcopy

import pytest

from operator_graph.analysis.engine import analyze_graph
from operator_graph.validation.replay import validate_replay


def sample():
    tensor = {"shape": [4], "dtype": "float16", "device": "cpu"}
    return {"model_name": "sample", "summary": {"num_nodes": 1}, "nodes": [
        {"op_id": "n0", "op_type": "MUL", "op_name": "mul", "raw_target": "Mul",
         "stage": "prefill", "layer_id": "old_label", "parents": [], "children": [],
         "input_tensors": [tensor, tensor], "output_tensors": [tensor]}]}


def test_replay_exercises_real_backend_without_mutating_source():
    data = sample()
    before = deepcopy(data)
    v = validate_replay(data, analyze_graph(data))
    assert v["passed"]
    assert data == before
    assert v["stages"]["prefill"]["unique_layer_operator_count"] == 1
    assert v["stages"]["prefill"]["layer_total_flops"] == 4
    assert v["upstream_export_tested"] is False
    assert v["source_fidelity_tested"] is False


def test_stale_summary_and_graph_errors_fail_validation():
    data = sample()
    data["summary"]["num_nodes"] = 2
    data["nodes"][0]["children"] = ["missing"]
    v = validate_replay(data, analyze_graph(data))
    assert not v["passed"]
    assert not v["checks"]["summary_node_count"]
    assert not v["checks"]["graph_integrity"]


@pytest.mark.parametrize("field,value,check", [
    ("weighted_flop_span", 5, "weighted_span_reference"),
    ("num_shape_signatures", 2, "shape_signature_partition"),
    ("cast_count", 1, "precision_accounting"),
    ("top10pct_node_flop_share", 2, "workload_fraction_ranges"),
])
def test_replay_rejects_inconsistent_new_metrics(field, value, check):
    data = sample()
    report = analyze_graph(data)
    report["stages"]["prefill"][field] = value
    validation = validate_replay(data, report)
    assert not validation["passed"]
    assert not validation["checks"][f"prefill:{check}"]
