"""Graph storage and explicit structural operations."""
from __future__ import annotations
from collections import Counter, deque
from pathlib import Path


class ComputationGraph:
    def __init__(self, model_name=''):
        self.model_name = model_name
        self.prompt_text = ''
        self.prompt_tokens = 0
        self.execution = {}
        self._nodes = {}
        self._layer_tree = None

    @property
    def nodes(self):
        return list(self._nodes.values())

    @property
    def num_nodes(self):
        return len(self._nodes)

    def add_node(self, node):
        if node.op_id in self._nodes:
            raise ValueError(f'Duplicate call ID: {node.op_id}')
        self._nodes[node.op_id] = node

    def get_node(self, op_id):
        return self._nodes.get(op_id)

    def add_edge(self, parent_id, child_id):
        if parent_id not in self._nodes or child_id not in self._nodes:
            raise ValueError('Unknown call in dependency edge')
        parent, child = self._nodes[parent_id], self._nodes[child_id]
        if child_id not in parent.children:
            parent.children.append(child_id)
        if parent_id not in child.parents:
            child.parents.append(parent_id)

    def topo_sort(self):
        for n in self.nodes:
            if len(set(n.parents)) != len(n.parents) or len(set(n.children)) != len(n.children):
                raise ValueError('Duplicate dependency edge')
            for direction, inverse in [('parents','children'),('children','parents')]:
                for key in getattr(n, direction):
                    other = self.get_node(key)
                    if other is None or n.op_id not in getattr(other, inverse):
                        raise ValueError('Dangling or asymmetric dependency')
        degree = {n.op_id: len(n.parents) for n in self.nodes}
        queue = deque(k for k, value in degree.items() if value == 0)
        order = []
        while queue:
            n = self._nodes[queue.popleft()]
            order.append(n)
            for child in n.children:
                degree[child] -= 1
                if degree[child] == 0:
                    queue.append(child)
        if len(order) != self.num_nodes:
            raise ValueError('Dependency cycle: no valid topological order')
        return order

    def compute_levels(self):
        levels = {}
        for n in self.topo_sort():
            levels[n.op_id] = 1 + max((levels[p] for p in n.parents), default=-1)
        return levels

    def critical_path(self, weight_attr=''):
        order = self.topo_sort()
        if not order:
            return 0, []
        weights = {n.op_id: getattr(n, weight_attr) if weight_attr else 1 for n in order}
        if any(v is None for v in weights.values()):
            return None, []
        if any(v < 0 for v in weights.values()):
            raise ValueError('Path weights must be nonnegative')
        distance, previous = {}, {}
        for n in order:
            parent = max(n.parents, key=distance.get) if n.parents else None
            previous[n.op_id] = parent
            distance[n.op_id] = (distance[parent] if parent else 0) + weights[n.op_id]
        last = max(distance, key=distance.get)
        length, path = distance[last], []
        while last is not None:
            path.append(last)
            last = previous[last]
        return length, path[::-1]

    def get_operator_counts(self):
        return dict(Counter(n.op_type for n in self.nodes))

    def tag_unassigned_as(self, stage):
        for n in self.nodes:
            if n.stage in ('unknown', 'forward', ''):
                n.stage = stage

    def get_layers(self):
        return sorted({n.layer_id for n in self.nodes if n.layer_id})

    def get_layer_nodes(self, layer_id):
        return [n for n in self.nodes if n.layer_id == layer_id]

    def get_layer_tree(self):
        return self._layer_tree

    def set_layer_tree(self, tree):
        self._layer_tree = tree

    def to_dict(self):
        from operator_graph.serialization import graph_to_dict
        return graph_to_dict(self)

    def save_to_json(self, output_dir, name=''):
        from operator_graph.serialization import graph_to_json
        filename = f'{name}_graph.json' if name else 'graph.json'
        return graph_to_json(self, output_path=Path(output_dir)/filename)
