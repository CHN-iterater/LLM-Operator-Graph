"""Call, tensor and optional display-group records; no runtime dependencies."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TensorMeta:
    shape: tuple[int, ...] | None
    dtype: str
    device: str = 'cpu'

    def to_dict(self):
        return {'shape': list(self.shape) if self.shape is not None else None,
                'dtype': self.dtype, 'device': self.device}


@dataclass
class OperatorNode:
    op_id: str
    op_type: str
    op_name: str
    category: str = 'other'
    layer_id: str = ''
    input_tensors: list[TensorMeta] = field(default_factory=list)
    output_tensors: list[TensorMeta] = field(default_factory=list)
    flops: int | None = None
    memory_bytes: int | None = None
    arith_intensity: float | None = None
    stage: str = 'forward'
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    raw_target: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)
    hardware_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_metadata=True):
        result = {k: getattr(self, k) for k in ('op_id','op_type','op_name','category',
                  'layer_id','flops','memory_bytes','arith_intensity','stage')}
        result.update(parents=list(self.parents), children=list(self.children),
                      input_tensors=[t.to_dict() for t in self.input_tensors],
                      output_tensors=[t.to_dict() for t in self.output_tensors])
        if include_metadata:
            from copy import deepcopy
            result.update(raw_target=self.raw_target, metadata=deepcopy(self.metadata),
                          hardware_metrics=deepcopy(self.hardware_metrics))
        return result


@dataclass
class LayerNode:
    """Optional display hierarchy; memberships may include child memberships."""
    layer_id: str
    layer_type: str = 'unknown'
    op_ids: list[str] = field(default_factory=list)
    children: list[LayerNode] = field(default_factory=list)
    parent: LayerNode | None = field(default=None, repr=False, compare=False)

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def operator_ids(self, _visited=None):
        seen = set() if _visited is None else _visited
        if id(self) in seen:
            return set()
        seen.add(id(self))
        ids = set(self.op_ids)
        for child in self.children:
            ids.update(child.operator_ids(seen))
        return ids

    def num_ops(self, _visited=None):
        return len(self.operator_ids(_visited))

    def total_flops(self, graph):
        values = [graph.get_node(k).flops for k in self.operator_ids() if graph.get_node(k)]
        return sum(values) if all(v is not None for v in values) else None

    def to_dict(self):
        return {'layer_id': self.layer_id, 'layer_type': self.layer_type,
                'op_ids': list(self.op_ids), 'num_ops': self.num_ops(),
                'children': [c.to_dict() for c in self.children]}
