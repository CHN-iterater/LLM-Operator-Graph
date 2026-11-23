"""Top-level ONNX calls, ordered operands and deduplicated dependencies."""
from pathlib import Path
from itertools import chain
from operator_graph.model.graph import ComputationGraph
from operator_graph.model.records import TensorMeta, OperatorNode
from .op_names import ONNX_OP_MAP


def _dtype_name(element_type):
    from onnx import TensorProto

    name = TensorProto.DataType.Name(element_type).lower()
    return {'float': 'float32', 'double': 'float64'}.get(name, name)


def _collect_tensor_metadata(graph):
    tensors = {}
    declarations = chain(graph.value_info, graph.input, graph.output)
    for declaration in declarations:
        tensor_type = declaration.type.tensor_type
        if tensor_type.HasField('shape'):
            shape = tuple(
                dimension.dim_value if dimension.HasField('dim_value') else -1
                for dimension in tensor_type.shape.dim
            )
        else:
            shape = None
        dtype = _dtype_name(tensor_type.elem_type) if tensor_type.elem_type else 'unknown'
        tensors[declaration.name] = TensorMeta(shape, dtype)

    for initializer in graph.initializer:
        tensors[initializer.name] = TensorMeta(
            tuple(initializer.dims), _dtype_name(initializer.data_type)
        )
    return tensors


def _operand_metadata(names, tensors):
    records = []
    for name in names:
        record = tensors.get(name)
        if record is None:
            record = TensorMeta(None, 'unknown')
        records.append(record)
    return records


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

    def _call_record(self, index, node, tensors):
        import onnx

        # Display aliases must not replace the raw type used by estimation.
        display = ONNX_OP_MAP.get(node.op_type)
        spec = self.registry.get(display) if display and display != 'UNKNOWN' else None
        spec = spec or self.registry.lookup(node.op_type)
        inputs = [name for name in node.input if name]
        outputs = [name for name in node.output if name]
        numeric_types = (
            onnx.AttributeProto.INT, onnx.AttributeProto.INTS,
            onnx.AttributeProto.FLOAT, onnx.AttributeProto.FLOATS,
        )
        attributes = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute if attribute.type in numeric_types
        }
        return OperatorNode(
            op_id=f'onnx_op_{index:04d}', op_type=spec.name,
            op_name=node.name or f'{node.op_type}_{index}',
            category=spec.category, layer_id='root', raw_target=node.op_type,
            input_tensors=_operand_metadata(inputs, tensors),
            output_tensors=_operand_metadata(outputs, tensors),
            metadata={
                'onnx_op_type': node.op_type, 'domain': node.domain or '',
                'source': 'onnx', 'onnx_index': index,
                'input_names': inputs, 'output_names': outputs,
                'input_slots': list(node.input), 'output_slots': list(node.output),
                'onnx_attributes': attributes,
            },
        )

    def parse(self, model_name=''):
        import onnx  # Check the optional dependency before validating parser state.

        if self.graph is None:
            raise RuntimeError('Call load() before parse()')

        tensors = _collect_tensor_metadata(self.graph)
        graph = ComputationGraph(model_name or self.source.stem)
        producers = {}
        for index, node in enumerate(self.graph.node):
            call = self._call_record(index, node, tensors)
            graph.add_node(call)
            for name in call.metadata['output_names']:
                if name in producers:
                    raise ValueError(f'Duplicate tensor producer: {name}')
                producers[name] = call.op_id

        for index, node in enumerate(self.graph.node):
            for name in node.input:
                if name and name in producers:
                    graph.add_edge(producers[name], f'onnx_op_{index:04d}')
        graph.topo_sort()
        graph.execution = {
            'graph_kind': 'onnx_forward', 'source_onnx': str(self.source.resolve()),
            'subgraphs_expanded': False,
        }
        return graph


def annotate_graph(graph):
    from operator_graph.estimation.rules import estimate_node_cost
    for node in graph.nodes:
        cost = estimate_node_cost(node.to_dict())
        node.flops = cost['flops']
        node.memory_bytes = cost['logical_io_bytes']
        node.arith_intensity = cost['arithmetic_intensity']
        node.metadata['static_cost'] = cost
    return graph


def parse_onnx(path, model_name='', registry=None):
    parser = OnnxParser(registry)
    parser.load(path)
    return annotate_graph(parser.parse(model_name))
