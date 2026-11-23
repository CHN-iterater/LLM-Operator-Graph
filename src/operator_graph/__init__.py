"""Call-level operator graphs with traceable static estimates."""
from .model.graph import ComputationGraph
from .model.records import OperatorNode, TensorMeta, LayerNode
from .model.registry import OperatorRegistry, OperatorSpec
from .parsers.onnx import parse_onnx, annotate_graph
from .serialization import graph_to_dict, graph_to_json, load_graph

__version__ = '0.1.0'


def parse_model(model, *example_args, **kwargs):
    from .exporters.torch import parse_model as export
    return export(model, *example_args, **kwargs)


__all__ = ['parse_model','parse_onnx','annotate_graph','ComputationGraph','OperatorNode',
           'TensorMeta','LayerNode','OperatorRegistry','OperatorSpec','graph_to_dict','graph_to_json','load_graph']
