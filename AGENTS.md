# Agent instructions

Before changing project direction, read this file, [`MASTER_DEVELOPMENT_DIRECTIVE.md`](MASTER_DEVELOPMENT_DIRECTIVE.md), [`HANDOFF.md`](HANDOFF.md), [`ROADMAP.md`](ROADMAP.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), and relevant research/decision notes. These documents preserve the user's requirements; do not restart or duplicate completed work.

## Project philosophy

QML is a local-first computational research laboratory, not a chatbot or a thin wrapper around an inference engine. “Quantum-inspired” refers to classical mathematics only and must never imply physical quantum computation or quantum advantage. Treat RMQ, MAP, QMF, and other QML methods as proposals until mathematical definitions, comparison baselines, and reproducible evidence exist. Report failed experiments and limitations as carefully as successes.

## Technical constraints

- Rust is the primary implementation language. Do not add Python application code, scripts, or mandatory Python development/runtime dependencies. Introduce C++/CUDA/WGSL only with a justified, independently validated need.
- Keep the application local-first: no required account, cloud service, telemetry, API subscription, or remote transmission of weights/private experiment data.
- Preserve imported model files by default. Never commit weights, credentials, tokens, private datasets, or personal information.
- Keep `unsafe` forbidden at workspace level unless the owner explicitly revises that policy. Validate sizes, offsets, integer arithmetic, and resource budgets before allocations or accesses.
- Introduce crates only for cohesive responsibilities. Do not add placeholder crates or pretend unfinished commands work.
- Evaluate every dependency for purpose, license, security status, maintenance, and transitive Python independence. Keep `Cargo.lock` checked in. Test-only oracles must remain dev dependencies.
- The repository is licensed under the **MIT License** (`LICENSE`, approved by the repository owner); respect all third-party dependency licenses verified via `cargo-deny`.
- **Permanent Memorial Dedication:** The memorial dedication to **Ashlan Haley Rapponotti** at the end of `README.md` must remain intact throughout all future documentation updates, repository reorganizations, and releases. Never remove, alter, or relocate it.
- **Releases:** Update `RELEASE_NOTES.md` with clear, detailed release notes for each milestone push and publish versioned `v*` GitHub Releases via `.github/workflows/release.yml`.

## Architecture and coding conventions

- Keep parsers and numerical kernels independently testable and free of CLI presentation concerns. Return descriptive errors for untrusted input; do not panic on malformed files.
- Bound input and allocation sizes before reading/allocating. Prefer checked arithmetic for dimensions, byte counts, and offsets. Never allocate proportional to a model payload for metadata-only operations.
- Make output deterministic where applicable. Preserve user input files; inspection commands are read-only.
- Document supported formats precisely. Unknown dtypes, fields, or semantics must fail explicitly rather than be guessed. A test oracle is evidence for fixtures, not proof of universal compatibility.
- Use synthetic or legally appropriate fixtures. Do not include downloaded weights or private datasets in tests or Git.

## Git workflow

- Work only on `main`. Do not create branches, force-push, or rewrite published history.
- Inspect the working tree and existing changes before editing; do not overwrite work from another agent.
- Commit and push each meaningful validated milestone with a descriptive, focused message. Update `HANDOFF.md` before each milestone push and keep it accurate as status changes.

## Engineering and testing

For applicable changes, run:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --workspace
cargo deny check
```

Add meaningful unit/integration/math/reference/property tests where appropriate, plus malformed-input, serialization, determinism, numerical-stability, resource-budget, regression, and reproducibility coverage. Use explicit tolerances with documented justification for approximate methods. Do not add meaningless tests, fake results, or unnecessary skips. Keep CPU-only CI functional; GPU validation is optional and must state whether it ran.

CI must cover Windows and Linux and run applicable formatting, lint, build, tests, and dependency security/license checks. Document any toolchain or external-service limitation rather than claiming a check passed.

## Research, security, and documentation

Keep measured results, established research, untested hypotheses, and QML contributions distinct. Cite primary or reputable sources beside claims. Do not claim novelty, patentability, quantum advantage, or performance superiority without evidence.

Update architecture, roadmap, changelog, experiment manifests, decision records, and handoff notes when relevant. README and CLI help must separate implemented, experimental, proposed, and unavailable capabilities. Never advance a milestone version to imply unfinished work is complete.
