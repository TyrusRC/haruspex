# Decision-row format (JSONL)

One decision per line. Three types. `state` may be a string or a JSON object
(serialized compactly). Lines beginning with `#` are comments.

## boolean (Noul)
```json
{"type":"boolean","state":"...","question":"...",
 "criteria":{"true":"what true means","false":"what false means"},
 "target":true}
```
- `target`: `true` / `false`, or a float probability in `[0,1]` (soft label).
- Readout: `P(yes) = sigmoid(score / T)`, then optional isotonic.

## choice
```json
{"type":"choice","state":"...","question":"...",
 "candidates":{"optA":"description","optB":"description"},
 "target":"optA"}
```
- `target`: the correct option name, or a dict of `{option: probability}` (soft).
- Readout: `softmax(scores / T)` over candidates.

## score (ordered rubric)
```json
{"type":"score","state":"...","question":"...",
 "levels":["low","medium","high"],
 "target":2}
```
- `target`: the level index (0-based), or a float.
- Readout: `softmax(scores / T)` over levels; reported `score = Σ i · pᵢ`.

## Labeling sources (bootstrapping, no manual grind)
1. The deterministic heuristic (tag/tech overlap) as weak labels to cold-start.
2. Real outcomes from a pentest harness (confirmed = positive, false-positive = negative).
3. Distillation: label a few hundred rows once with a big model (`scripts/distill.py`),
   then train the small specialist and run it offline forever.

Keep near-duplicates (same document / trajectory / counterfactual family) out of
both train and eval splits so the numbers are honest.
