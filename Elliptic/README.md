# T-JEPA on Elliptic

Adaptation of [jose-melo/t-jepa](https://github.com/jose-melo/t-jepa) (a tabular JEPA that
treats each COLUMN/feature as a token and learns via feature-masking) to the Elliptic Bitcoin
fraud dataset.

## Changes from the public repo
- Added `elliptic_dataset.py` (`Elliptic(BaseDataset)`) — see file docstring for details.
- Registered the dataset in the framework's dataset map.
- Added a dataset-specific branch in `TorchDataset.__init__` implementing Elliptic's official
  temporal train/val/test split (see below), instead of the framework's default random split.
- Self-supervised context/target encoder: standard repo `Encoder` class, `hidden_dim=64,
  num_layers=4, num_heads=8`, **trained from scratch (randomly initialized), 1 epoch** of
  self-supervised JEPA pretraining on Elliptic (feature-masking + prediction), before generating
  embeddings — see RESULTS.md for why this matters.
- Downstream classifier: repo's built-in `src.models.mlp.MLP`, `encoder_type="linear_flatten"`,
  `input_embed_dim=64`, 4 hidden layers x 256 units, trained via PyTorch Lightning.

## Official split used
Train: time_step 1-27 · Validation: 28-34 · Test: 35-49 (matches the field-standard Elliptic
benchmark split).

See `RESULTS.md` for metrics.
