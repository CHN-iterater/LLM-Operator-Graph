"""Logical DAG topology and arithmetic weighted paths."""
from __future__ import annotations

from collections import Counter, deque

def _topology(nodes, malformed=False, cost_rows=()):
    ids = {n["op_id"] for n in nodes}
    children = {n["op_id"]: set(n.get("children", [])) & ids for n in nodes}
    indeg = dict.fromkeys(ids, 0)
    for cs in children.values():
        for c in cs:
            indeg[c] += 1
    original_indeg = indeg.copy()
    weights = {r["op_id"]: r["flops"] for r in cost_rows}
    weights_known = bool(nodes) and all(weights.get(k) is not None for k in ids)
    span_to = dict.fromkeys(ids, 0)
    q = deque(sorted(k for k,v in indeg.items() if v == 0))
    level = {k: 0 for k in q}
    visited = 0
    while q:
        k = q.popleft()
        visited += 1
        if weights_known:
            span_to[k] += weights[k]
        for c in sorted(children[k]):
            level[c] = max(level.get(c, 0), level[k] + 1)
            if weights_known:
                span_to[c] = max(span_to[c], span_to[k])
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    valid = visited == len(nodes) and not malformed
    widths = Counter(level.values()) if valid else {}
    edges = sum(len(cs) for cs in children.values())
    span = max(span_to.values()) if valid and weights_known else None
    return {"num_edges": edges,
            "is_dag": None if malformed else visited == len(nodes),
            "critical_path_nodes": max(level.values())+1 if valid and level else None,
            "mean_level_width": len(nodes)/len(widths) if widths else None,
            "max_level_width": max(widths.values()) if widths else None,
            "singleton_level_fraction": sum(v==1 for v in widths.values())/len(widths) if widths else None,
            "weighted_flop_span": span,
            "flop_work_over_span": sum(weights.values())/span if span else None,
            "single_producer_consumer_edge_fraction": (
                sum(len(cs) == 1 and original_indeg[c] == 1 for cs in children.values() for c in cs)/edges
                if valid and edges else None),
            "fanout_node_fraction": sum(len(cs) > 1 for cs in children.values())/len(nodes) if valid and nodes else None,
            "max_fanout": max(map(len, children.values())) if valid and nodes else None,
            "level_widths": dict(sorted(widths.items()))}
