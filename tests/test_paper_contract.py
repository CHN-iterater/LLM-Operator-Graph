import json
import pytest

from operator_graph import parse_onnx
from operator_graph.model.graph import ComputationGraph
from operator_graph.model.records import OperatorNode
from operator_graph.serialization import graph_to_dict, load_graph


def test_unavailable_is_not_zero_in_serialization():
    g = ComputationGraph('unknown')
    g.add_node(OperatorNode(op_id='x', op_type='Unsupported', op_name='x'))
    d = graph_to_dict(g)
    assert d['summary']['total_flops'] is None
    assert d['nodes'][0]['flops'] is None
    assert d['summary']['known_flops'] == 0


def test_cycles_and_dangling_edges_are_explicit():
    g = ComputationGraph('cycle')
    for key in ['a', 'b']:
        g.add_node(OperatorNode(op_id=key, op_type='Mul', op_name=key, flops=0))
    g.add_edge('a', 'b')
    g.add_edge('b', 'a')
    with pytest.raises(ValueError, match='cycle'):
        g.topo_sort()
    with pytest.raises(ValueError, match='Unknown'):
        g.add_edge('a', 'missing')


def test_weighted_path_preserves_zero_and_unknown():
    g = ComputationGraph('zero')
    for key, weight in [('a', 0), ('b', 4), ('c', 0)]:
        g.add_node(OperatorNode(op_id=key, op_type='Mul', op_name=key, flops=weight))
    g.add_edge('a', 'b'); g.add_edge('b', 'c')
    assert g.critical_path('flops')[0] == 4
    g.get_node('b').flops = None
    assert g.critical_path('flops') == (None, [])


def test_onnx_ordered_operands_optional_slots_and_unique_edges(tmp_path):
    import onnx
    from onnx import helper, TensorProto
    info = lambda name, shape: helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape)
    nodes = [helper.make_node('Mul', ['x', 'x'], ['a']), helper.make_node('Gemm', ['a', 'w', ''], ['y'], transB=1)]
    model = helper.make_model(helper.make_graph(nodes, 'example', [info('x', [2, 3]), info('w', [4, 3])], [info('y', [2, 4])], value_info=[info('a', [2, 3])]))
    path = tmp_path/'example.onnx'; onnx.save(model, path)
    g = parse_onnx(path)
    assert g.num_nodes == 2
    assert g.nodes[0].metadata['input_names'] == ['x', 'x']
    assert g.nodes[1].metadata['input_slots'] == ['a', 'w', '']
    assert g.nodes[0].children == ['onnx_op_0001']
    assert g.nodes[1].metadata['onnx_attributes']['transB'] == 1
    assert g.nodes[1].metadata['static_cost']['flops'] == 48
    assert g.nodes[0].stage == 'forward'


def test_unknown_rank_is_distinct_from_scalar_and_zero_dimension(tmp_path):
    import onnx
    from onnx import helper, TensorProto
    info = lambda name, shape: helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
    model = helper.make_model(helper.make_graph([helper.make_node('Identity', ['x'], ['y'])], 'x', [info('x', None)], [info('y', None)]))
    path = tmp_path/'unknown.onnx'; onnx.save(model, path)
    g = parse_onnx(path)
    assert g.nodes[0].output_tensors[0].shape is None
    assert g.nodes[0].metadata['static_cost']['declared_tensor_bytes'] is None
    # The parser can preserve zero dimensions without making a paper-validation claim.
    model.graph.output[0].CopyFrom(info('y', [0, 2]))
    onnx.save(model, path)
    assert parse_onnx(path).nodes[0].output_tensors[0].shape == (0, 2)


def test_legacy_reader_does_not_reclassify_decode_without_provenance(tmp_path):
    d = {'schema_version': '1.0', 'model_name': 'old', 'nodes': [{'op_id': 'x', 'op_type': 'Mul', 'op_name': 'x', 'stage': 'decode', 'parents': [], 'children': [], 'input_tensors': [], 'output_tensors': []}]}
    path = tmp_path/'old.json'; path.write_text(json.dumps(d), encoding='utf-8')
    g = load_graph(path)
    assert g.nodes[0].stage == 'decode'
    assert g.execution.get('decode_stage_semantics') is None
    path2 = tmp_path/'new.json'
    path2.write_text(json.dumps(graph_to_dict(g)), encoding='utf-8')
    assert load_graph(path2).nodes[0].stage == 'decode'


def test_legacy_static_status_overrides_zero_placeholder(tmp_path):
    node = {'op_id': 'x', 'op_type': 'Unsupported', 'op_name': 'x', 'flops': 0,
            'memory_bytes': 0, 'arith_intensity': 0.0,
            'metadata': {'static_cost': {'flops': None, 'logical_io_bytes': None,
                                        'arithmetic_intensity': None, 'flops_status': 'unsupported'}}}
    path = tmp_path/'legacy.json'
    path.write_text(json.dumps({'schema_version': '1.0', 'nodes': [node]}), encoding='utf-8')
    d = graph_to_dict(load_graph(path))
    assert d['nodes'][0]['flops'] is None
    assert d['summary']['total_flops'] is None


def test_replay_accepts_valid_graph_with_unavailable_estimates():
    from operator_graph.analysis.engine import analyze_graph
    from operator_graph.validation.replay import validate_replay
    from operator_graph import annotate_graph
    g = ComputationGraph('unknown')
    g.add_node(OperatorNode(op_id='x', op_type='Unsupported', op_name='x'))
    d = graph_to_dict(annotate_graph(g))
    assert validate_replay(d, analyze_graph(d))['passed']


def test_schema_reader_writer_reject_dangling_edges():
    from operator_graph.serialization import graph_from_dict
    g = ComputationGraph('bad')
    g.add_node(OperatorNode(op_id='x', op_type='Mul', op_name='x', parents=['missing']))
    with pytest.raises(ValueError, match='Dangling'):
        graph_to_dict(g)
    with pytest.raises(ValueError, match='Dangling'):
        graph_from_dict({'schema_version':'2.0', 'nodes':[g.nodes[0].to_dict()]})


def test_custom_domain_does_not_inherit_builtin_rules():
    from operator_graph.estimation.rules import estimate_node_cost
    t = {'shape': [4], 'dtype': 'float32'}
    n = {'op_type':'Mul', 'input_tensors':[t,t], 'output_tensors':[t],
         'metadata':{'onnx_op_type':'Mul', 'domain':'custom.example'}}
    result = estimate_node_cost(n)
    assert result['flops'] is None
    assert result['logical_io_bytes'] is None
    assert result['flops_status'] == 'unsupported_domain'


def test_replay_normalizes_optional_tensor_device():
    from operator_graph.analysis.engine import analyze_graph
    from operator_graph.validation.replay import validate_replay
    t = {'shape':[4], 'dtype':'float32'}
    d = {'nodes':[{'op_id':'x', 'op_name':'x', 'op_type':'Mul', 'stage':'forward',
                   'input_tensors':[t,t], 'output_tensors':[t]}]}
    assert validate_replay(d, analyze_graph(d))['passed']


def test_schema2_cannot_silently_drop_estimation_provenance():
    g = ComputationGraph('provenance')
    g.add_node(OperatorNode(op_id='x', op_type='LAYER_NORM', op_name='x',
                           raw_target='SimplifiedLayerNormalization'))
    with pytest.raises(ValueError, match='provenance'):
        graph_to_dict(g, include_metadata=False)


def test_display_aliases_do_not_erase_raw_onnx_types(tmp_path):
    import onnx
    from onnx import helper, TensorProto
    info = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT, [2])
    node = helper.make_node('GroupQueryAttention', ['x'], ['y'], domain='com.microsoft')
    path = tmp_path/'display.onnx'
    onnx.save(helper.make_model(helper.make_graph([node], 'display', [info('x')], [info('y')])), path)
    call = parse_onnx(path).nodes[0]
    assert call.op_type == 'ATTENTION'
    assert call.raw_target == 'GroupQueryAttention'
    assert call.metadata['domain'] == 'com.microsoft'
    assert call.flops is None


def test_onnx_dependency_error_precedes_unloaded_parser_error(monkeypatch):
    import builtins
    from operator_graph.parsers.onnx import OnnxParser

    parser = OnnxParser()
    original_import = builtins.__import__

    def without_onnx(name, *args, **kwargs):
        if name == 'onnx':
            raise ModuleNotFoundError("No module named 'onnx'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_onnx)
    with pytest.raises(ModuleNotFoundError, match='onnx'):
        parser.parse()
