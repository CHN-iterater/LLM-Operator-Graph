"""Schema-2 output and lossless reading of structural fields in legacy JSON."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
import json

SCHEMA_VERSION = '2.0'


def graph_to_dict(graph, registry=None, include_metadata=True):
    if not include_metadata:
        raise ValueError('Schema 2 requires raw operator and estimation provenance')
    graph.topo_sort()
    nodes = [n.to_dict(include_metadata) for n in graph.nodes]
    def total(key):
        values = [n[key] for n in nodes]
        return sum(values) if all(v is not None for v in values) else None
    layers = graph.get_layers()
    layer_tree = graph.get_layer_tree()
    result = {'schema_version': SCHEMA_VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
              'model_name': graph.model_name,
              'prompt': {'text': graph.prompt_text, 'tokens': graph.prompt_tokens},
              'execution': deepcopy(graph.execution),
              'summary': {'num_nodes': len(nodes), 'num_layers': len(layers),
                          'layers': layers, 'operator_counts': graph.get_operator_counts(),
                          'total_flops': total('flops'), 'total_memory_bytes': total('memory_bytes'),
                          'known_flops': sum(n['flops'] for n in nodes if n['flops'] is not None),
                          'known_memory_bytes': sum(n['memory_bytes'] for n in nodes if n['memory_bytes'] is not None)},
              'layer_tree': layer_tree.to_dict() if layer_tree else None,
              'nodes': nodes}
    if registry is not None:
        result['registry_info'] = {'num_operator_types': registry.num_operators,
                                  'operator_types': [s.name for s in registry.list_specs()]}
    return result


def graph_to_json(graph, registry=None, output_path=None, include_metadata=True, indent=2):
    if output_path is None:
        raise ValueError('output_path is required')
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph_to_dict(graph, registry, include_metadata),
                              ensure_ascii=False, indent=indent, allow_nan=False), encoding='utf-8')
    return path


def graph_from_dict(data):
    from operator_graph.model.graph import ComputationGraph
    from operator_graph.model.records import OperatorNode, TensorMeta, LayerNode
    if str(data.get('schema_version', '1.0')) not in ('1.0', SCHEMA_VERSION):
        raise ValueError('Unsupported graph schema version')
    g = ComputationGraph(data.get('model_name', ''))
    g.execution = deepcopy(data.get('execution', {}))
    g.prompt_text = data.get('prompt', {}).get('text', '')
    g.prompt_tokens = data.get('prompt', {}).get('tokens', 0)
    allowed = {f.name for f in fields(OperatorNode)}
    for n in data.get('nodes', []):
        values = {k: deepcopy(v) for k, v in n.items() if k in allowed}
        values.setdefault('op_name', n['op_id'])
        for key in ('input_tensors', 'output_tensors'):
            values[key] = [TensorMeta(tuple(t['shape']) if t.get('shape') is not None else None,
                                      t.get('dtype', 'unknown'), t.get('device', 'cpu')) for t in n.get(key, [])]
        # Explicit legacy statuses take precedence over numeric zero placeholders.
        # This reads stored provenance; re-estimation is a separate operation.
        if str(data.get('schema_version', '1.0')) == '1.0':
            cost = values.get('metadata', {}).get('static_cost', {})
            for field_name, cost_key in [('flops','flops'), ('memory_bytes','logical_io_bytes'),
                                         ('arith_intensity','arithmetic_intensity')]:
                if cost_key in cost:
                    values[field_name] = cost[cost_key]
        g.add_node(OperatorNode(**values))
    def layer(d):
        node = LayerNode(d['layer_id'], d.get('layer_type', 'unknown'), d.get('op_ids', []))
        for child in d.get('children', []):
            node.add_child(layer(child))
        return node
    if data.get('layer_tree'):
        g.set_layer_tree(layer(data['layer_tree']))
    g.topo_sort()
    return g


def load_graph(path):
    return graph_from_dict(json.loads(Path(path).read_text(encoding='utf-8-sig')))
