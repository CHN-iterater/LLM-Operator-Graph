# Paper-to-code alignment

Reference: the English manuscript shortened on 2026-11-21. This migration does not modify the paper or its archived data.

Implementation paths below are relative to `src/operator_graph/`.

| Paper requirement | New implementation | Change from legacy code |
|---|---|---|
| Metadata collection, call records, dependency linking, and derived attributes | Separate model, parser, and estimation modules, orchestrated by the CLI | Reporting and stage inference moved out of mixed entry points and the large graph class |
| Top-level ONNX calls and raw operator types | `raw_target`, `metadata.onnx_op_type`, and `domain` stored separately | Display names do not override raw types |
| Ordered operands and dependency edges have different granularity | Raw `input_slots` retain empty optional positions; `input_names` and tensor lists retain nonempty uses in order | Repeated inputs are preserved; dependency edges are deduplicated only by call pair |
| Formulas, assumptions, availability, and aggregation basis | `metadata.static_cost` and offline per-call reports | `static-v2.0` formulas retained; legacy `utils` is not a second active rule source |
| Known zero differs from unavailable | Node values use `None`; aggregation reports complete and known-part totals separately | Unavailable values are no longer filled with zero; explicit legacy `static_cost` states override zero placeholders |
| Completeness and availability are not construction accuracy | Validation reports state their scope | Replay, small-graph checks, and archive agreement are not labeled independent accuracy measurements |
| Two independent cache-disabled inputs | Export uses the precise `full_prompt`/`single_token_probe` conditions | Legacy labels remain readable and are not interpreted as cached decoding |
| Logical access is a static estimate under declared rules | Dtype payload, view aliasing, materialization, Gather, and repeated-use rules retained | `memory_bytes` is not described as peak GPU memory or measured DRAM traffic |
| AI is the ratio of complete totals | Completeness and a nonzero denominator remain required | Node AI values are not averaged; no actual bottleneck classification is made for a specific GPU |
| Topological levels and logical paths | Cycles and dangling dependencies raise explicit errors; weighted paths retain zero | Failed sorting no longer returns insertion order; zero weights are no longer raised to one |

## Added robustness and compatibility handling

- Unknown rank, scalars, and zero dimensions remain distinct. The parser preserves true zero dimensions rather than converting them to unknown values. The paper's experiments still cover positive static dimensions or scalars; small code tests do not expand that experimental scope.
- Consumers are identified by traversal index, avoiding omission of nodes with no nonempty output when looking up nodes by output.
- Nonstandard custom domains do not inherit standard estimation rules merely because an operator is named `Mul` or `Reshape`. Standard domains and legacy records without a domain are supported by default; other domains need explicit rule implementations. This namespace guard does not change the corpus formulas: all seven archives use standard domains.
- Schema 2.0 reading and writing reject invalid dependencies. To diagnose existing damaged JSON, dictionaries can still be passed directly to `analyze_graph`.
- When estimates are unavailable, layer-aggregation checks compare complete totals rather than comparing `None` with a known-part total.
- Omitted optional defaults such as `device` in legacy tensor records are normalized semantically, avoiding failures caused by irrelevant default-field differences.

## Features outside the default core

| Original module or feature | Disposition |
|---|---|
| Two calculators in `utils` | Historical interfaces remain in the original repository; the new project has one rule source |
| `PhaseSplitter`, legacy KV analysis, and Roofline reports | Actual execution phases are not inferred from input-size heuristics; extensions require separate design |
| FX `OperatorParser`, `OperatorHook`, `TensorRecorder` | Potential optional backends; the initial ONNX workflow works independently |
| `ModuleParser`, `ModuleHook` | Module-level analysis extensions, separate from call-level ONNX construction |
| Hardware parameters and legacy profiling interfaces | Downstream projects or future optional extensions |
| Text generation and answer display | Not required for constructing graphs from the paper's two forwards; the default CLI does not generate text |
| `LayerPartitioner` | Retained as an explicit extension; heuristic labels are not evidence of construction accuracy |

The current project makes no claim to cover cached-decoding experiments, nested-subgraph expansion, numerical inference-equivalence validation, or measured energy-consumption improvements. Model export support is bounded by actual export tests and the software versions used.
