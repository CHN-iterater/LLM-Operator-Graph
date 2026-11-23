"""Aggregate complete and partial estimates without conflating unavailable values with zero."""
from __future__ import annotations

from collections import Counter

def _aggregate(rows):
    n = len(rows)
    fs = [r["flops"] for r in rows if r["flops"] is not None]
    bs = [r["logical_io_bytes"] for r in rows if r["logical_io_bytes"] is not None]
    paired = [r for r in rows if r["flops"] is not None and r["logical_io_bytes"] is not None]
    pair_b = sum(r["logical_io_bytes"] for r in paired)
    total_f = sum(fs) if len(fs) == n else None
    total_b = sum(bs) if len(bs) == n else None
    return {"num_nodes": n, "estimated_flops": total_f, "known_flops": sum(fs),
            "logical_io_bytes": total_b, "known_logical_io_bytes": sum(bs),
            "flop_model_coverage": len(fs)/n if n else None,
            "io_model_coverage": len(bs)/n if n else None,
            "flops_status_counts": dict(Counter(r["flops_status"] for r in rows)),
            "arithmetic_intensity": total_f/total_b if total_f is not None and total_b else None,
            "modeled_arithmetic_intensity": sum(r["flops"] for r in paired)/pair_b if pair_b else None,
            "ai_matched_node_count": len(paired)}
