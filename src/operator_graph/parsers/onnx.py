"""Top-level ONNX calls, ordered operands and deduplicated dependencies."""
from pathlib import Path
from operator_graph.model.graph import ComputationGraph
from operator_graph.model.records import TensorMeta, OperatorNode
from .op_names import ONNX_OP_MAP


class OnnxParser:
    def __init__(self, registry=None):
        from operator_graph.model.registry import OperatorRegistry
        self.registry = registry or OperatorRegistry.get_default()
        self.graph = None
        self.source = None

    def load(self, path):
        import onnx
        self.source = Path(path)
        self.graph = onnx.load(str(self.source), load_external_data=False).graph

    def parse(self, model_name=''):
        import onnx
        if self.graph is None:
            raise RuntimeError('Call load() before parse()')
        tensor_map = {}
        for v in [*self.graph.value_info, *self.graph.input, *self.graph.output]:
            tt = v.type.tensor_type
            shape = tuple(d.dim_value if d.HasField('dim_value') else -1 for d in tt.shape.dim) if tt.HasField('shape') else None
            dtype = onnx.TensorProto.DataType.Name(tt.elem_type).lower() if tt.elem_type else 'unknown'
            dtype = {'float':'float32','double':'float64'}.get(dtype, dtype)
            tensor_map[v.name] = TensorMeta(shape, dtype)
        for v in self.graph.initializer:
            dtype = onnx.TensorProto.DataType.Name(v.data_type).lower()
            tensor_map[v.name] = TensorMeta(tuple(v.dims), {'float':'float32','double':'float64'}.get(dtype, dtype))
        g = ComputationGraph(model_name or self.source.stem)
        producer = {}
        for i, node in enumerate(self.graph.node):
            # Registry labels are presentation metadata. Estimation uses raw ONNX type.
            display = ONNX_OP_MAP.get(node.op_type)
            spec = self.registry.get(display) if display and display != 'UNKNOWN' else None
            spec = spec or self.registry.lookup(node.op_type)
            ins, outs = [n for n in node.input if n], [n for n in node.output if n]
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute
                     if a.type in (onnx.AttributeProto.INT, onnx.AttributeProto.INTS,
                                   onnx.AttributeProto.FLOAT, onnx.AttributeProto.FLOATS)}
            call = OperatorNode(op_id=f'onnx_op_{i:04d}', op_type=spec.name, op_name=node.name or f'{node.op_type}_{i}',
                                category=spec.category, layer_id='root', raw_target=node.op_type,
                                input_tensors=[tensor_map.get(n, TensorMeta(None, 'unknown')) for n in ins],
                                output_tensors=[tensor_map.get(n, TensorMeta(None, 'unknown')) for n in outs],
                                metadata={'onnx_op_type': node.op_type, 'domain': node.domain or '', 'source':'onnx',
                                          'onnx_index': i, 'input_names': ins, 'output_names': outs,
                                          'input_slots': list(node.input), 'output_slots': list(node.output),
                                          'onnx_attributes': attrs})
            g.add_node(call)
            for name in outs:
                if name in producer:
                    raise ValueError(f'Duplicate tensor producer: {name}')
                producer[name] = call.op_id
        for i, node in enumerate(self.graph.node):
            for name in node.input:
                if name and name in producer:
                    g.add_edge(producer[name], f'onnx_op_{i:04d}')
        g.topo_sort()
        g.execution = {'graph_kind': 'onnx_forward', 'source_onnx': str(self.source.resolve()),
                       'subgraphs_expanded': False}
        return g


def annotate_graph(graph):
    from operator_graph.estimation.rules import estimate_node_cost
    for n in graph.nodes:
        cost = estimate_node_cost(n.to_dict())
        n.flops, n.memory_bytes, n.arith_intensity = cost['flops'], cost['logical_io_bytes'], cost['arithmetic_intensity']
        n.metadata['static_cost'] = cost
    return graph


def parse_onnx(path, model_name='', registry=None):
    parser = OnnxParser(registry)
    parser.load(path)
    return annotate_graph(parser.parse(model_name))
