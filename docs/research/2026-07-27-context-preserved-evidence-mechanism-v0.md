# Context-preserved evidence mechanism: candidate design v0

This is a mechanism design entry, not an architecture lock.

## Evidence behind the design

- Crop replacement increased disease/damage false positives (Task 11C.1).
- Local features added conditional taxonomy information beyond a same-width
  global control on two disjoint exploration batches (Tasks 12A and 12B).
- Therefore global context must remain the anchor, while local evidence may be
  useful as a conditional residual rather than a replacement.

## Candidate mechanism family

Let `g` be a full-image representation and `l` an evidence representation.
A context-preserved family has the form

`z = global_anchor(g) + evidence_gate(g, l) * local_residual(l)`.

The gate represents evidence presence/reliability, not taxonomy confidence.
The diagnosis head sees `z`; abstention and evidence presence are supervised
separately. The exact projection, gate, and token implementation remain open.

## Required competing explanations

- `M0`: global-only baseline.
- `M1`: crop replacement, already falsified for reliability.
- `M2`: ungated `G+L` fusion; establishes complementarity but is not the proposed
  final method.
- `M3`: global-anchored gated residual candidate.
- `H2-control`: full-frame token selection without a second crop encoder.
- `H3-control`: separate presence and taxonomy heads without feature fusion.

## Falsifiable predictions

1. Local information should improve positive taxonomy conditional on the global
   feature.
2. Removing the global anchor should increase real-null confidence.
3. A genuine presence gate should reduce real-null acceptance without erasing
   the conditional positive gain.
4. Improvements must survive prompt changes and fresh null sources; otherwise
   the gate is another dataset shortcut.

## Next smallest discriminator

Before implementing a gated multimodal adapter, Task 13A should test whether a
separate frozen-feature evidence-presence head trained without PlantSeg can
generalize to fresh positive, PlantDoc healthy-null, and untouched PlantSeg
damage-null. This isolates H3 from H1. Failure blocks a learned gate and sends
the project toward H2 token selection; success permits a tiny gated-fusion
prototype. No QLoRA or confirmatory set is authorized.

