# Agent instructions

- Work only on `main`; do not create branches, force-push, or rewrite published history.
- Read this file and the relevant architecture/research notes before changing project direction.
- Keep Rust as the primary implementation language. Do not add Python application code, Python scripts, or mandatory Python runtime dependencies.
- Do not fabricate tests, benchmarks, citations, hardware support, or capabilities. Mark unimplemented commands and experimental behavior accurately.
- Treat RMQ, MAP, QMF, and other named directions as proposals until supported by sourced research and reproducible experiments.
- Preserve user/model data; never commit model weights, credentials, tokens, private datasets, or personal information.
- Validate changes with `cargo fmt --all -- --check`, `cargo test --workspace`, and relevant CLI checks before committing.
- Keep commits focused and descriptive. Commit meaningful milestones to `main` and push without force.
- Reconcile this scaffold with the complete master directive when its missing continuation arrives.
