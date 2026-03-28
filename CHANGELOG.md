# Changelog

## 0.1.0 — Unreleased

- Initial release
- Schema: TrainConfig, KitManifest, Recipe, TrainResult, EvalResult
- Backbone: SentenceEmbedder (frozen sentence-transformers wrapper)
- Models: ClassifierHead (Linear + MLP)
- Training: train_model() with early stopping, metrics tracking
- Export: ONNX, Core ML (updatable), .kit.zip packaging
- Eval: Run pack eval protocol with per-class metrics
- CLI: train, eval, export, kit, info, list-models
