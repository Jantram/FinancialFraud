# Fin-JEPA on AMLSim

Adaptation of [cedricwyh/fin-jepa](https://github.com/cedricwyh/fin-jepa) (originally built for
daily stock price sequences) to the IBM AMLSim synthetic anti-money-laundering dataset.

## Why AMLSim, and why this required adaptation
The original repo assumes one entity (a stock) tracked over consecutive daily rows. AMLSim
provides an analogous structure: each **account** has a transaction history over time, which we
treat as the "daily sequence" the model expects.

## Changes from the original public repo
- `model.py`: fixed a syntax error in the original file (`class Fin-JEPA` is invalid Python —
  hyphens aren't allowed in identifiers). Renamed to `FinJEPA`. No architectural changes.
- The original repo's data pipeline (`data.py`, referenced by `compare_arch2.py` /
  `benchmark.py`) is not included in the public release (imported from the author's private
  `~/dev/chan-jepa` path) and is specific to Chinese equities data. We wrote an entirely new
  data pipeline (`data_pipeline.py`) for AMLSim from scratch, following the same
  `(context, target)` windowing contract the model expects.
- `experiment_e.py` (predictor-error-as-signal idea) also depends on private HuggingFace
  datasets/checkpoints. We reused the *idea* (prediction error as an anomaly signal) in our own
  evaluation code rather than the original script.

See `REPORT.md` for full methodology and results.
