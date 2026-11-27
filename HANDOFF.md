# Handoff

## PROJECT COLOSSUS — authorized experiment in progress

Owner selected only Qwen/Qwen2.5-Coder-14B-Instruct, superseding the earlier
22B–24B choice and authorizing this one original BF16 download after SENTINEL.
Pinned revision: aedcc2d42b622764e023cf882b6652e646b95671.
All experiment storage is D:\QML-COLOSSUS, with dedicated original-model,
download-cache, scratch, calibration, reference-evaluation, compressed-qmf
and experiment-results directories. Preflight free D: space was
585,672,253,440 bytes; 200 GiB was reserved as the planning floor.
No other models are downloaded; no existing files or models are deleted.

The only application correction counts emitted generation tokens over total
generation-call time including prefill, and separates ladder evaluation rate
from optional sample-generation rate. Prompt tokens no longer inflate the
generation numerator. Inference, quantizers, graphics and architecture are
unchanged. Targeted runtime metric and real-generation regression tests pass;
the ladder separation regression passes; formatting and strict Clippy for the
runtime/CLI and all their targets pass. Full final workspace release validation
has not yet run for COLOSSUS.

Frozen protocol revision 2 defines 14 tasks across all seven requested categories,
17 disjoint calibration passages (1,340 tokens, including nine actual code
passages) and seven numerical-reference passages (125 predictions).
The external Rust experiment driver builds offline using existing workspace
APIs/dependencies. Positive oracle self-tests passed 14/14; deliberately wrong
negative controls were rejected 14/14. These are synthetic harness tests,
not model scores. All six original shards now pass official SHA-256 verification;
native bounded inspection counts 579 BF16 tensors, 14,770,033,664 parameters,
29,540,067,328 payload bytes and 29,540,133,960 shard-file bytes. No 14B
compression, generation-quality, performance or fit result is claimed yet.
Original BF16 generation may be constrained by available RAM; do not claim
task-quality retention without an actually measured original baseline.
See docs/PROJECT_COLOSSUS.md and experiments/exp-007-project-colossus.toml.

The initial 283-token prose-only calibration setup was revised before any
quantization or candidate output. Its preliminary reference pass was deliberately
stopped after 43.806 seconds; no cache completed. All v1 protocol, provenance,
partial logs and measurements remain on D:. Revision 2 adds code calibration
coverage while preserving every held-out task/assertion, numerical-reference
passage and generation setting. Current cache identity:
40c4a4351ba8056a15974ce9b9a88d4183d220fd7f9d0eaad1dcee83f96af2c2.
The full protocol, external Rust driver and objective oracle are documented in
docs/COLOSSUS_REPRODUCTION.md. Active pipeline is
D:\QML-COLOSSUS\scratch\run-experiment.ps1, with evidence in
experiment-results\experiment-pipeline-v2.log and reference-v2.stderr.log.
It uses the same existing algorithms, ATLAS publication and verified cache APIs;
the untied embedding retains the existing non-calibrated quantizer when no
embedding calibration statistic is supplied.

Application checkpoint 7d9ca53dd715ddcfd30b3440c4dc13fd131803f9 is pushed to main.
CI run https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/actions/runs/45999301187
passed all jobs: 142 Windows tests, 143 Linux tests, zero failures/ignored, plus
formatting, documentation/memorial checks, builds, strict Clippy and dependency
policy. This is an intermediate validated checkpoint, not a completed COLOSSUS
release. Original autoregressive generation was not attempted after resource
preflight: 29,540,133,960 checkpoint bytes exceed 17,563,394,048 available
physical RAM bytes before runtime overhead; source BF16 streaming remains valid.

Next: finish and verify only the pinned source; run streaming calibration and
reference; forge/evaluate INT3, INT2 and QMR-200 using existing algorithms;
record every outcome/resource failure, update all required scientific docs,
validate and publish v0.6.5 through the native Releases-only six-asset protocol,
independently verify it, report exact total D: storage, and STOP. QMR-175 is
optional if resources permit. Do not download any additional model, commit
weights/QMF, change algorithms/Prism/distribution/MIT/memorial or begin another
milestone. v0.6.4 remains the latest completed verified release.

## PROJECT SENTINEL — v0.6.4 COMPLETE AND INDEPENDENTLY VERIFIED — STOP

Verified November 27, 2026. The public native GitHub Releases section contains v0.6.4 | PROJECT SENTINEL, draft=false, prerelease=false and latest=true. All six required assets are present with comprehensive release notes: standalone Windows desktop EXE, Windows portable desktop/CLI ZIP, Linux desktop/CLI archive, source archive, RELEASE_MANIFEST.json and SHA256SUMS.txt. Distribution remains exclusively native GitHub Releases; the README has only the approved informational navigation and now displays verified Windows/Linux results. Signing is not configured and no signature is claimed.

Immutable annotated tag v0.6.4 identifies application source 5a6bfe2aa37411efe009a12d5748567827c2702d. Main CI passed at https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/actions/runs/45990440439. Exact-tag validation, publication and automatic independent public verification passed at https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/actions/runs/45990448588. The formerly skipped downstream-verification case is resolved: optional signing was skipped, while verify-public actually ran and succeeded. Windows passed 140 tests and Linux 141, zero failures/ignored, plus doc-tests; all five required checks passed in their platform/policy jobs. Existing tests remain intact.

The unauthenticated public job downloaded all six assets, verified complete checksum coverage and exact source identity, checked public/latest/maturity flags, confirmed standalone/portable desktop byte equality, validated both platform packages' artwork/documentation/MIT/memorial and executed the downloaded Linux CLI. A separate local retrieval of the actual public Windows portable ZIP matched its public digest, passed packaged documentation/memorial checks and executed qml 0.6.4 natively. No GUI launch/performance benchmark or real-model experiment is claimed. Only the README/results record and this handoff change after the immutable application tag; release assets and historical releases are not overwritten.

| Public asset | Bytes | SHA-256 |
|---|---:|---|
| Windows standalone EXE | 13,871,104 | 8676707a0dd1fabcd806dea6b79b7157188b4eb8d11bab64ed2f1445255e8c4e |
| Windows portable ZIP | 12,374,029 | 7d556d7ce902337a2b5fa20e988ad16ee12ecb640dc1b8bcbe14e92ba7ee0cee |
| Linux archive | 15,204,281 | 2582eae803eb16ce93591c51e79d07913dcb93a3a95e17691c395286d81f0bbb |
| Source archive | 3,622,263 | 98fb118a958ddef52f543263a5e7074752253e8be114836b1e2ba70dc1944a48 |
| RELEASE_MANIFEST.json | 842 | 0b9bc2a7d1f32aaf153882eb04c8700a99277536640bb7f604f61c31b3521c71 |
| SHA256SUMS.txt | 552 | f5412a983250aefea326b46b83ae85c125e277470954ffdb3d74dbbff01d4c9a |

Completion audit: architecture/activation/attention/RoPE/window admission, harmless inactive metadata, finite constants, checked dimensions, optional-bias presence/shape/byte length/finite/error behavior across all three loaders, unchanged AEGIS output-projection errors, runtime-bound cache provenance migration, 19 new synthetic regressions, all validation gates, six native release assets and independent public verification are complete. MIT, Project Prism/hardware kernels, original artwork, historical experiments, existing model files, dependency graph and permanent distribution rules are preserved. The exact memorial remains the final README section. No model was downloaded or deleted. The following candidate-era notes are historical; this verified status supersedes their pending statements.

Remaining boundaries: only the documented unscaled existing Llama/Qwen2 subset is admitted; no scaled/hybrid or new architecture implementation is claimed. Old caches require a fresh output directory because corrected effective defaults/semantics change provenance. Dimension representability is not a physical RAM/VRAM reservation. Safetensors cannot identify a finite structurally valid value change without independently expected identity; unsigned checksums do not authenticate a producer. No new pretrained-model quality, independent reference-runtime agreement or performance finding follows from the synthetic tests. SENTINEL is complete. Stop now and await explicit owner authorization for any subsequent milestone.

Owner-authorized starting checkpoint: 65d1491764303582846792fa406f3e408a3d36a1. CONTINUUM is complete and independently verified. SENTINEL is limited to runtime architecture/configuration admission, optional attention-bias correctness, synthetic regression validation and public v0.6.4 publication. Stop after SENTINEL is complete; do not begin another milestone, download/delete models, add architectures, change compression mathematics or graphics, or change the permanent distribution policy. Older CONTINUUM active/pending/continue statements below are historical and superseded by this scope and stop condition.

Implementation: a private HF admission module validates supported Llama/Qwen2 causal identity, SiLU/swish, full-head attention geometry, unscaled split-half RoPE (including identity-only settings), actual sliding-window/layer-type behavior, supported family defaults and classified metadata. Unsupported active controls, conflicting identities/rotary settings and unclassified fields return explicit errors. TransformerConfig validates positive finite CPU/GPU constants, rotary angle representability, shader scalar dimensions and checked matrix/sequence/trace/KV/directory byte products. Runtime directories grow as real layers load rather than reserving from an unbacked count. The public config/model APIs, existing supported computation and backends remain available.

Optional bias handling now uses manifest/source metadata to distinguish absence from failure in memory QMF, file-backed QMF and streaming BF16. Present q/k/v vectors require exact rank/length, finite values and successful bounded decoding with tensor-named errors. AEGIS required-output-projection logic is unchanged. No mathematical kernels, format algorithms, graphical files, license, historical experiments, model files, dependencies or distribution rules were changed. The version badge alone updates its informational version; artwork is otherwise unchanged.

Focused validation: 18 new synthetic SENTINEL tests pass; all 22 runtime tests and doc-tests pass, including the unchanged AEGIS and GPU/CPU calibration/KV parity tests. New cases cover identity/defaults, representative metadata without weight allocations, unsupported activation/features, attention dimensions, unscaled/unsupported/conflicting RoPE, inactive versus active windows, finite constants, byte arithmetic, configuration size, unbacked layer count, absent/valid/wrong-rank/wrong-length q/k/v vectors, valid-checksum non-finite F32/F16/BF16, decoded-length mismatch, corrupt CRC, injected I/O and streaming truncation. Positive q/k/v cases affect multi-token logits and preserve memory/file CPU equality. An initial new-test compile used a nonexistent cache convenience method; the fixture was corrected to the existing KvCache constructor before successful validation. Initial 15 cases, extended 16 cases and the final 22-test runtime run passed. Strict runtime Clippy passed before the final fixture extension; full workspace validation follows at v0.6.4.

Release status: candidate only until all five required local checks, Windows/Linux CI and exact-tag release/public verification pass. The established six-asset native Releases-only protocol is unchanged, including the CONTINUUM fix allowing automatic verification after optional signing is skipped. Full results and immutable source identity will be recorded before completion.

Review follow-up: pinned HF v4.43.1 sources confirmed omitted theta 10000 for both families, distinct from an explicitly supplied Qwen2.5 checkpoint theta 1000000; the initial draft fallback/test expectation was corrected before publication. Typed rotary fields and a duplicate-checking metadata map reject duplicated active controls before a later inactive value can hide them. All 22 runtime tests passed after this parser refinement and after extending non-finite half-vector cases through streaming as well as both QMF paths. Final workspace checks are rerun on the frozen source. Earlier intermediate full workspace format/lint/tests/release-build/deny runs passed; only the final frozen-source results establish release readiness.

Limitations: admission covers the documented existing unscaled Llama/Qwen2 subset, not every variant using those names. Explicitly supported family defaults correct earlier omitted-field values, so omitted-value behavior can change. Unknown controls must be classified instead of silently ignored. Addressable arithmetic is not a RAM/VRAM reservation. Safetensors cannot detect a finite structurally valid value change without independent expected identity. Synthetic tests do not establish pretrained-model quality, independent reference agreement, new hardware performance or graphical behavior. No model download or real-model experiment is authorized here. See docs/RUNTIME_ADMISSION.md.

Runtime cache correctness follow-up: the CLI cache fingerprint previously bound raw config bytes but not runtime semantics/effective defaults. A small execution_identity method now binds a SENTINEL runtime domain and validated serialized settings, and the existing CLI hash incorporates it. This is required to avoid reusing pre-correction defaults or ignored bias failures as current evidence. No cache format, serialization, publication algorithm or existing cache file changed. Older identities must regenerate in a new output directory; all historical files are preserved. One additional focused test verifies stable identities, sensitivity to effective epsilon/context, and rejection of invalid configurations: 19 new SENTINEL tests in total. Final full workspace validation includes this migration.

Final local validation on the frozen v0.6.4 source: cargo fmt --all -- --check passed; cargo clippy --workspace --all-targets -- -D warnings passed; cargo test --workspace passed all 140 Windows tests (23 runtime tests, including 19 new cases), zero failures/ignored, plus doc-tests; cargo build --workspace --release passed in 38.76 seconds; cargo deny check passed advisories/bans/licenses/sources under the unchanged policy, with existing allowed duplicate-version warnings. All 44 documentation files, local artwork/document paths and the exact final memorial passed the permanent Rust policy guard. The release contract compiled with warnings denied and accepted packaged documentation at the source root. All three canonical README remote references returned HTTP 200. Native actionlint passed all three unchanged workflows. A Git diff check confirms AGENTS/distribution rules, MIT, historical experiments, original artwork/diagram, Project Prism and hardware/shader source are unchanged. The lockfile changes only the 13 workspace package versions. Platform package smoke and exact-tag CI/public verification follow before completion.

Local Windows packaging passed: actual standalone desktop and portable desktop/CLI package created; extracted package retains complete offline README artwork/docs, MIT and the exact memorial; packaged CLI reports qml 0.6.4; standalone and portable desktop bytes match. Older private v0.6.3 build staging was moved to verified owned target storage without deletion or overwrite. Local packages are not offered as a distribution alternative. Implementation is ready for main-branch CI, followed by immutable exact-tag release validation and independent public verification.

Pre-tag corruption review adds exact QMF dense-vector encoding/byte-length checks before decoding, closing a potential large allocation from a small vector declaration with an inconsistent blob length. The existing length/CRC regression now injects an I/O failure at the bias payload and proves byte-length rejection happens without any payload read. The same 19 new tests remain; no extra format or allocation policy was introduced. This refinement is validated before tagging; b9e4e743065b4cdeda23dbe678ce59a3b95fa607 is an intermediate implementation commit, not the final release source.

The final predecode refinement passed all five local checks again: formatting, strict workspace Clippy, all 140 Windows tests/doc-tests, optimized workspace build (28.60 seconds) and cargo-deny with the unchanged allowed duplicate-version warnings. The standalone policy and release-contract checks also passed. This supersedes the earlier local candidate build for publication.

The final pinned-default review distinguishes omitted Qwen2 KV heads (HF default 32) from explicit null (query-head count); Llama omissions still use query heads. The existing family-default regression now checks omission, explicit null and explicit KV values separately. This affects only omitted Qwen2 settings; explicit existing checkpoint values remain unchanged. The runtime fingerprint protects prior cache evidence from this correction too. Validation and exact-tag publication use the final source containing this distinction; 52243bd90600a6e4da80a88b1c12798e603d297e remains an intermediate candidate, not a published tag.

Final publication source: all five checks passed after the omitted/null KV-head correction, including all 140 Windows tests and 23 runtime tests (19 new regressions), zero failures/ignored, and doc-tests. Optimized workspace build completed in 28.23 seconds; cargo-deny and both standalone documentation/release checks passed under unchanged policies. README now shows these latest engineering results with no unsupported scientific claim and retains only native Releases navigation. At the owner's usage-limit steering, proceed directly with the validated main push and immutable v0.6.4 tag; the existing exact-tag workflow independently repeats Windows/Linux gates before publication and public verification. No further scope expansion is authorized.

## PROJECT CONTINUUM — v0.6.3 VERIFIED RESTORATION CHECKPOINT

Public verification completed November 27, 2026. The native GitHub Releases section contains v0.6.3 | PROJECT CONTINUUM with draft=false, prerelease=false and latest=true. All six mandatory assets are present: standalone Windows desktop executable, Windows portable desktop/CLI package, Linux desktop/CLI archive, source archive, RELEASE_MANIFEST.json and SHA256SUMS.txt. Comprehensive source release notes are the native release body. Software distribution remains exclusively native GitHub Releases; no direct asset URLs are used here.

Immutable annotated tag v0.6.3 points to application source 5e76f96819ac68de3f0d4aad853efb6316df7581. Main CI passed at https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/actions/runs/45978456766. Exact-tag release validation and publication passed at https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/actions/runs/45979159833. Separate read-only public verification passed at https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/actions/runs/45980675785. It downloaded all six native public assets without authentication, verified complete SHA-256 coverage and exact source provenance, checked public/latest/maturity flags, confirmed standalone/portable desktop byte equality, checked packaged artwork/license/memorial in both platforms and executed the downloaded Linux qml 0.6.3 CLI. A separate local retrieval of the actual public Windows portable package matched its SHA-256 and passed documentation/memorial validation and native qml 0.6.3 execution. No graphical launch or new renderer benchmark is claimed.

| Public asset | Bytes | SHA-256 |
|---|---:|---|
| Windows standalone executable | 13,778,944 | 5df075e0b2effc5f78e28f4970dd6b2f9ddf588562ed8fa01e646499d2327d25 |
| Windows portable package | 12,292,118 | c010b129af90ec4fe186b14e2d8aa692d3beefb309c3aa489df3245a96c563fe |
| Linux archive | 15,127,677 | 2ee5cc763bed1e6b010357b2fa47db5ad97d51404373c0277534861976ab3bfb |
| Source archive | 3,597,242 | 8a5d6467abb36c04791432b308afb612d0ffa54fc87d3b14c907798017d3fd7e |
| RELEASE_MANIFEST.json | 842 | 8843934dc17840a3765011b2feee28bdedf4b11eef8efff595361e2611e0aa0e |
| SHA256SUMS.txt | 552 | 562e33b2de5c2af2cd64eca6bd1e308c4be92627d88420174d610c0d792fefe4 |

Final engineering results: cargo fmt --all -- --check, cargo clippy --workspace --all-targets -- -D warnings, cargo test --workspace, cargo build --workspace --release and cargo deny check passed locally and in the exact-tag platform/policy jobs. The final source has 121 passing Windows tests and 122 passing Linux tests, zero failures/ignored, plus passing doc-tests. All 14 policy/release-contract tests passed locally. Native actionlint passed all three workflows; both PowerShell scripts parsed successfully; all canonical README remote references returned 200 and all committed image/document paths and the exact memorial passed offline policy validation. Existing tests were not weakened. No normal/dev/build dependencies or mathematical algorithms were upgraded or replaced.

Signing is explicitly not-configured; no signature authenticity is claimed. The original automatic verification job was skipped through GitHub's disabled-signing dependency propagation, so recovered publication used the independent read-only workflow. Main now has an explicit always()/successful-publication guard for future automatic runs; that guard passed workflow lint, while the current immutable tag retains its original workflow. No historical tag, release or asset was moved, deleted or overwritten. Subsequent main changes are workflow scheduling, the read-only recovery verifier and documentation/provenance evidence, not different application binaries. The continuous campaign remains active; runtime admission, malformed optional-bias handling and independent real-model inference comparison are still outstanding. The metadata/provenance reassessment below is completed groundwork, not completion of those gates.

Publication follow-up: main CI 45978456766 passed 121 Windows and 122 Linux tests, builds and strict lint, with dependency policy successful. Release run 45979159833 passed exact-tag Windows/Linux formatting, Clippy, tests, optimized workspace builds, packaged CLI/documentation checks, dependency policy, source assembly and publication. Optional signing was correctly skipped (not configured). Its downstream independent verification was also skipped by GitHub's default dependency condition despite publication success. The future workflow now explicitly requires always() and successful publication for that verification job. A separate read-only manual verification workflow is added to verify existing immutable tags without changing releases; v0.6.3 remains unverified until this runs successfully. No existing tag or release asset is overwritten.

Continued baseline reassessment: docs/VALIDATION_BASELINE.md records the existing checkpoint's pinned upstream revision, matching hashes for all four immutable shards and config/tokenizer/index identities, native metadata inspection of 339 unique BF16 tensors, and complete tensor-shape/shard-map consistency. The measured source total is 15,231,271,888 bytes; exp-006's individual shard sizes agree, but its historical aggregate field is 400,000 bytes smaller. The historical record is preserved and the discrepancy is documented separately. Native inspect accepts individual shards; an index JSON is explicitly rejected by the bounded-header parser. This is metadata/provenance evidence only, not a new inference or quality benchmark. Local anonymous API polling reached its rate limit; authenticated connector monitoring continued, and hosted public verification remains deliberately unauthenticated.

This checkpoint restores scientific presentation and establishes permanent native-Releases-only distribution and release infrastructure. The campaign continues beyond this checkpoint; it does not claim completion of runtime admission, independent real-model reference validation, new compression research or large-model experiments.

Changes: exact tag/workspace-version/source-commit and release-note validation; bounded native Rust manifest/checksum contract and exclusive metadata creation; source archive; complete offline documentation/artwork packaging; exact packaged CLI version smoke; default read-only workflow permissions; separate optional OIDC signing with exact tagged-workflow identity; non-overwriting draft-first publication; independent unauthenticated public asset/provenance/checksum/latest/maturity verification and public package smoke checks. Manifest signing remains not-configured unless QML_SIGN_RELEASES=true; no signature is claimed for the default path. README validation additionally rejects opaque reference/HTML links and repository-prefix lookalikes. Source remains Rust-only with no dependency upgrades or application algorithm, renderer, shader, model or benchmark changes.

Local validation: formatting and strict workspace Clippy passed; the full workspace run passed 118 tests plus doc-tests before the final added policy/package regression cases; the optimized workspace build passed in 48.08 seconds; cargo deny passed advisories, bans, licenses and sources with unchanged allowed duplicate-version warnings. New policy/package cases receive a targeted rerun before commit. Native actionlint passed both workflows; PowerShell syntax validation passed both packaging/verification scripts. All three canonical remote README references returned HTTP 200. The initial locked check correctly required the version-only workspace lockfile update; offline regeneration changed only the 13 workspace package versions. The initial compliance commit 7982aa0d9906d54d82ab5d561e16e92aeabca557 passed Windows/Linux/dependency-policy CI, run 45975945236.

The candidate-era status and publication recovery notes below are historical validation records; the verified status above supersedes their pending statements. Preserve all historical tags/releases. Public software distribution remains exclusively the native GitHub Releases section; no direct asset links are permitted in this or any other documentation.

Final local follow-up: all 14 policy/release-contract integration tests passed (5 standalone policy and 9 release-contract tests, including shared policy and NIST SHA tests). Formatting and strict workspace Clippy were rerun successfully after those final Rust changes. The package checker initially rejected an omitted README-linked experiment record; packaging now includes committed experiment records and the retry passed. Windows standalone and portable staging passed complete documentation/license/memorial checks and exact qml 0.6.3 CLI smoke. Failed owned staging was preserved under target for investigation rather than deleting existing artifacts. No local package is offered as an alternative distribution location. Native public publication and Linux execution remain pending CI.

Remaining correctness findings from read-only reassessment: HF configuration parsing does not yet reject all unsupported architecture/activation/positional semantics; present-but-malformed optional q/k/v bias loads can still be swallowed in the memory, file-backed and streaming paths. These are prioritized next correctness work, not fixed or benchmarked by this release. Existing AEGIS required-output-projection correctness and ATLAS bounded-payload construction remain intact. No new model was downloaded or independently validated. The source-bound manifest and checksums detect corruption but an unsigned manifest does not authenticate a producer. GUI launch/rendering performance was not newly tested; package smoke executes the CLI.

## PROJECT CONTINUUM — active, permanent distribution override

Interim compliance restoration: prohibited artifact URLs were removed from README and source release-note documentation; the native Releases page is the only distribution navigation. Original committed artwork/screenshots are restored in a structured scientific README with a current classical architecture diagram and local informational badges. The exact final memorial is checked automatically. The always-applied rule and AGENTS/master overrides prevent historical prompts from reintroducing prohibited distribution. A native, bounded Rust policy checker is included as a workspace test and invoked in CI and release validation. Four policy regression tests and strict workspace Clippy passed; the first policy run correctly caught a missing capability-inventory file, which was then created and rechecked. Full CONTINUUM publication-infrastructure work and runtime admission correctness remain in progress; this interim commit does not claim a new completed release.

GitHub access is restored; ATLAS release assets and final main CI passed. The owner now authorizes continued CONTINUUM work and permanently supersedes all historical direct-download instructions. Native GitHub Releases is the exclusive software distribution section. Documentation may use only the approved neutral wording and an ordinary informational Releases page link. Follow docs/GITHUB_DISTRIBUTION_POLICY.md and automated repository policy validation. Preserve all earlier releases, source work, scientific records and the exact memorial. Do not weaken this rule without explicit owner authorization AND clarification from GitHub Support.


## PROJECT ATLAS — v0.6.2 COMPLETE

Starting checkpoint: 1c80f69477c483750dd1162f8ba68698aa918e5e. Work is exclusively on main. This owner-authorized assignment supersedes the prior AEGIS stop only for bounded-memory artifact construction, associated Q-Forge/CLI configuration, synthetic validation and v0.6.2 publication.

### Implementation

- QmfV1Builder retains its existing in-memory API and shares all encoder implementations with a new disk payload store. new_disk and finish_disk return a DiskArtifact. Scalar conversion streams through a fixed 64 KiB buffer; packed bytes are chunked. No complete compressed model payload is retained by the disk builder.
- The manifest, blob offsets/lengths/CRCs, deterministic ordering, shared latent references and QMF v1 version are preserved. The legacy small-model builder remains available. Existing reader defaults remain unchanged.
- DiskArtifact streams the complete artifact into an AEGIS sibling stage with incremental CRC-32 and SHA-256, reopens it through QmfFileReader, checks the whole-file hash and every blob without loading operators, checks cancellation and all protected source paths again, and uses AEGIS authorized/no-clobber publication.
- ArtifactLimits separates maximum artifact file, builder resident accounting, combined owned temporary file bytes, individual tensor allocation and manifest size. Defaults: 8 GiB / 256 MiB / 16 GiB / 8 GiB / 16 MiB respectively. Explicit reader_budget enables larger file-backed reads without raising default RAM allocations.
- DiskBuildOptions configures scratch storage. Free space is checked before construction, periodically during payload output and before final staging. It is advisory rather than a reservation; actual write failures propagate. NamedTempFile guards exclusively create and remove only owned scratch/stage files.
- Existing forge_path_to_file_atomic now uses disk construction. forge_path_to_file_atomic_with_options adds limits and explicit overwrite authorization. ForgeControl remains authoritative for cancellation in the forge API; standalone builders use DiskBuildOptions.cancel. Decoded tensor allocation is clamped by PayloadBudget and the artifact's tensor limit. Every backing shard and the original index/input file are protected again immediately before publication.
- Morph adds --scratch-dir, --max-artifact-bytes, --max-builder-memory-bytes, --max-temp-bytes, --max-tensor-bytes, --max-manifest-bytes and explicit --overwrite. Existing transformation strategies and calibration branches are unchanged.
- Added QmfError cancellation/resource errors and cooperative file-integrity verification. Checked arithmetic protects lengths, offsets, serialized/temporary totals, parameter counts and allocation conversions. Bounded manifest counting and conservative reservations guard builder metadata; legacy infallible metadata insertion latches any disk-mode resource failure instead of silently publishing.
- Normal dependencies tempfile and sysinfo are already in the workspace dependency graph; sysinfo is requested with only its disk feature for the new preflight responsibility. tempfile/serde_json test dependencies support isolated synthetic fixtures. No dependency versions were upgraded.
- Updated README downloads, release notes, changelog, architecture and roadmap. MIT license, memorial text, Prism renderer/shaders/identity, GPU kernels, compression/research algorithms and historical experiments were preserved.

### Synthetic validation and measured storage

- Byte-identical memory/disk artifacts across 17 transformation strategies, plus calibrated GPTQ parity. Existing file readers and operators reproduce numerical outputs, shared latent caches reuse the same entries, and sizes, offsets, complete bpp and SHA-256 match.
- Regression coverage includes independent resource bounds, checked overflow, physical free-space preflight rejection, logical scratch exhaustion, source allocation rejection before decoding, corrupt scratch CRCs, malformed dimensions, metadata limits, failed publication, cancellation during transformation and before publication, late aliases to a second shard, index protection, temporary file cleanup and unrelated scratch-file preservation. A CLI integration test exercises resource flags and explicit overwrite.
- Instrumented eight-tensor fixture, 1,048,576 encoded payload bytes: resident payload buffer 0 bytes; encoding buffer 65,536 bytes; conservative manifest/buffer accounting 151,280 bytes.
- Same eight tensor entries with longer arrays, 33,554,432 encoded payload bytes: resident payload buffer 0 bytes; encoding buffer 65,536 bytes; conservative manifest/buffer accounting 151,728 bytes.
- These counters measure builder-owned payload/buffers and conservative metadata accounting, not process RSS. Borrowed fixture arrays, input tensors, quantizer workspace, OS page cache and other application allocations are excluded. Fixtures write only modest synthetic data; no pretrained model was downloaded or transformed.
- cargo fmt --all -- --check: passed.
- cargo clippy --workspace --all-targets -- -D warnings: passed.
- cargo test --workspace: all 107 Windows tests passed, zero failures/ignored; doc-tests passed. Linux CI passed all 108 tests, including Unix-only link protection.
- cargo build --workspace --release: passed (45.52 seconds with the existing local dependency cache).
- cargo deny check: advisories/bans/licenses/sources passed under the unchanged policy; existing duplicate-version warnings remain allowed.
- Initial strict Clippy identified a cloned-reference test slice; corrected before final successful validation. Existing tests were not weakened or removed.

### Remaining memory bottlenecks and limitations

The 256 MiB resident limit is conservative builder-owned-state accounting (serialized manifest × 16 plus two 64 KiB I/O buffers), not an OS RSS cap. Existing quantizers still allocate full dense tensors and workspace; RMQ may retain a complete dense cluster. Calibration profiles, reports, source metadata and caller-owned operators require additional memory. Therefore this is bounded-payload-memory artifact construction, not complete end-to-end bounded-memory transformation.

Temporary disk limits count logical bytes of payload scratch plus final stage, so peak overlap is approximately twice the artifact payload plus manifest/header. Existing artifacts and unrelated files are not included in that owned-space budget. Disk preflight cannot reserve space against concurrent processes. Cleanup preserves user scratch directories and unrelated files. As in AEGIS, publication assumes user-controlled ancestors rather than hostile concurrent directory replacement.

Cancellation is checked between numerical operations and during bounded encoding, copying and verification; an individual running quantizer/kernel is not preempted. Some existing convenience readers and CLI compare still materialize full artifacts; file-backed APIs with explicit reader budgets are needed for larger files. Default CLI inference reader budgets remain conservative. Checksums detect corruption, not malicious producer authentication.

No 14B, 27B or 32B support has been scientifically validated. No sampled MAP-B optimization, persistent/resumable scheduling, new quantizers, GPU changes, architecture extensions or new research experiments were undertaken.

### Exact files changed from the ATLAS checkpoint

ARCHITECTURE.md; CHANGELOG.md; Cargo.lock; Cargo.toml; HANDOFF.md; README.md; RELEASE_NOTES.md; ROADMAP.md; crates/qml-cli/Cargo.toml; crates/qml-cli/src/main.rs; crates/qml-cli/tests/atlas_cli.rs; crates/qml-forge/Cargo.toml; crates/qml-forge/src/lib.rs; crates/qml-format/Cargo.toml; crates/qml-format/src/lib.rs; crates/qml-format/src/v1.rs; crates/qml-format/src/v1/disk.rs.
### Commit and CI status

Implementation commit: d426d91f6df53a2906612f65467346f0e4fb393a. Main-branch Windows/Linux/dependency-policy CI passed: https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/actions/runs/45812230204. Annotated tag v0.6.2 points to this validated commit. Release workflow was started at: https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/actions/runs/45812861029.

### Public release verified — November 27, 2026

Git and the GitHub plugin were reauthenticated successfully for ShamelesAbyss. The public release is v0.6.2 | PROJECT ATLAS: https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/tag/v0.6.2. The latest-release endpoint returns v0.6.2; draft=false and prerelease=false. The release workflow sets make_latest=true. All four mandatory assets were downloaded without authentication. All three binary hashes match the public SHA256SUMS.txt.

- Quantum-Morph-Laboratory-v0.6.2-Windows-x64.exe: 13778944 bytes; SHA-256 b488cacd6ded4125dbc6f36e465c9a6ca2ae21a17891f2ef030de45c342f5a7d.
- Quantum-Morph-Laboratory-v0.6.2-Windows-x64-Portable.zip: 9038341 bytes; SHA-256 8ce1e99cde45f5105b4cc4d74a58d4048ce17e5aeeea74be6d06cf89d851dd15.
- Quantum-Morph-Laboratory-v0.6.2-Linux-x64.tar.gz: 11914842 bytes; SHA-256 0adb4fa746517bb305fdfe219b5b28d58da00ea0762e6cb7d41400be9958ce89.

The standalone Windows EXE is byte-identical to the portable desktop binary. The downloaded portable CLI executed as qml 0.6.2, and both portable archives contain the expected applications and MIT license. Previous source CI was observed green on Windows (107 tests), Linux (108 tests) and dependency policy. Historical Actions run records currently return 404; they cannot be re-fetched. Source code and the release tag remain unchanged. All five required local validation commands passed as recorded above.

The access failure after the initial push blocked final verification temporarily; it is resolved. A final documentation-only local commit f831e0a0860720792229f130d0846f932ea25007 and the newer remote README edit 4ed1575fa2577cff4c58982b58dc5d29b23ee068 are joined with a regular merge on main, preserving both histories. The newer README edit is retained, with the exact permanent memorial restored. Its former direct-link instruction is superseded by the current compliance override. No source code, mathematical algorithms, protected subsystem or experiment measurements were changed during this reconciliation.

The earlier local completion report and Windows packages remain historical local backups. The public release and checksums are now verified; a refreshed completion report supersedes the earlier blocked status. No ATLAS implementation issues remain beyond the documented resource limitations. No 14B/27B/32B scientific validation is claimed.

**STRICT STOP: PROJECT ATLAS is complete. Await further instructions; do not begin another assignment.**

## PROJECT AEGIS — v0.6.1 COMPLETE

- Scope: artifact publication, required untied output projections, calibration/reference cache safety, terminology and release only. Do not start Wave IV or any subsequent milestone without authorization.
- Work performed on `main` in the Codex workspace; no downloaded models or new live-model experiments. The original v0.6.0 checkout was clean.
- `qml-core::publication` uses `tempfile` exclusive randomized sibling staging, `same-file` filesystem identities, destination symbolic-link rejection, and explicit overwrite authorization. Default publication is no-clobber. Authorized persistence never deletes the destination first; failures preserve prior bytes. Q-Forge checks every backing shard against the destination.
- `qml-format` exposes `write_to_file_atomic_with_overwrite`; its existing API defaults to refusing replacement. File-size and manifest validation remain in force. Original weights remain read-only.
- `qml-runtime` propagates required `lm_head.weight` loading errors in memory and lazy file-backed loading. Streaming uses embedding weights only for explicitly tied models and checks projection dimensions and finite BF16 values.
- `qml-core::cache` supplies versioned SHA-256 envelopes and bounded binary access. QCAL2/QREF2 validate dimensions, counts, checked byte lengths, finite f32/f64 values, duplicate calibration tensors, exact reference sequence/vocabulary dimensions and trailing bytes. Disk cache size is limited to 512 MiB.
- CLI cache identity hashes all backing weights, config, tokenizer, optional reference QMF, calibration/evaluation tokens, backend and reference/calibration modes. QREF2 additionally binds labels/categories/tokens. Existing QCAL1/QREF1 caches are unverified and rejected without modification: regenerate with a new output directory. New cache publication refuses overwrite and reports errors.
- Composite score renamed to `composite_numerical_fidelity_pct`; historical JSON's old field name remains a deserialization alias. Formula and measurements are preserved. README/CLI/release notes explain that it does not establish retained intelligence or benchmark capability.
- Updated workspace version, lockfile, README download links, release notes, architecture, roadmap and changelog. MIT license, history, Prism renderer and permanent memorial dedication are preserved.
- Added regression tests for failed/no-clobber publication, hard links, distinct staging names, authorized replacement, Unix symbolic/dangling links, required projection failures in all three execution paths, corrupted lazy output tensors, non-finite streaming projections, cache round trips, provenance/dataset mismatches, oversized lengths, non-finite values and malformed dimensions. Existing tests were retained.

### Validation state

- Rust 1.100.0 selected with a directory-local rustup override; cargo-deny 0.21.0.
- `cargo fmt --all -- --check`: passed.
- `cargo clippy --workspace --all-targets -- -D warnings`: passed on final v0.6.1 source.
- `cargo test --workspace`: all 95 tests passed on Windows, zero failed/ignored; all doc-tests passed. This includes the final projection/cache/publication regressions and existing desktop/GPU parity tests. Linux CI passed all 96 tests, including Unix symbolic/dangling-link coverage.
- `cargo deny check`: passed on v0.6.1 (`advisories ok, bans ok, licenses ok, sources ok`); existing duplicate-version warnings remain allowed by the unchanged policy.
- `cargo build --workspace --release`: passed (optimized workspace build, 5m 05s). Release CLI smoke check passed (`qml 0.6.1`). All five requested commands passed locally.

Initial compile/lint checks during implementation found a missing direct qml-core dependency, a serialization test API mismatch, and Clippy formatting/conditional suggestions; these were corrected before the successful final validation. No existing tests were weakened or removed.

### Limits and remaining work

Cache checksums detect corruption rather than authenticate a malicious producer. Hashing source provenance reads every backing weight byte once. Decoded f64 values and collection overhead require additional bounded RAM beyond the 512 MiB file ceiling. Path-based publication assumes output ancestors are user-controlled, not concurrently replaced by a hostile process. No claim of new live-model benchmark validation is made. Baseline commit: `1631b824ce9daa8f256d46c45497f1e804442dd8`. Implementation commit: `7c669db9e369bcd92ec08f4cb0cc4fefc663680a`, pushed to `main`; annotated release tag `v0.6.1` points to this commit. [Windows/Linux/dependency-policy CI](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/actions/runs/45805000853) passed all jobs. [Release workflow](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/actions/runs/45805910710) passed both platform builds and public publication. Public asset verification is complete below. No scoped implementation issues remain; stop and await authorization for further work.

### Public release verified (November 26, 2026)

- Latest completed milestone: **v0.6.1 | PROJECT AEGIS**. [Public release](https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory/releases/tag/v0.6.1); GitHub's public `releases/latest` endpoint returns `v0.6.1`. Confirmed `draft: false`, `prerelease: false`; release workflow sets `make_latest: true`.
- Downloaded all four mandatory public release assets without authentication. All three binary package hashes matched the downloaded `SHA256SUMS.txt`. The Windows standalone EXE is byte-identical to the portable archive's `qml-desktop.exe`; the packaged CLI executed successfully as `qml 0.6.1`. The portable ZIP includes both executables, documentation and MIT LICENSE. The Linux archive lists both binaries and LICENSE; the Linux workflow also smoke-tested its CLI.
- Verified Windows standalone EXE: 13,719,040 bytes; SHA-256 `4b75143261d56150ce660d0cac2f5e27029c0d599c28150b177e90a5f43f7a59`.
- Verified Windows portable ZIP: 8,964,903 bytes; SHA-256 `ca1d4e88678e7d2929ab024ad33461f03b2e77a7d91dd6d94827b4eccbd2d906`.
- Verified Linux archive: 11,819,180 bytes; SHA-256 `2a080e0af708743ef7417e0f9da5baee8bae9776c77301d18c2fe55ebd1452fa`.
- Implementation and release tag: `7c669db9e369bcd92ec08f4cb0cc4fefc663680a`. A final documentation-only follow-up records these verification results on `main`; the release tag and repository history remain immutable.
- Permanent memorial text verified verbatim against the baseline README. MIT license, experiment manifests, mathematical algorithms and Project Prism renderer were not modified. No models were downloaded and no later milestone was begun.
- **STRICT STOP: PROJECT AEGIS is complete. Await explicit further instructions.** Future roadmap entries below are historical planning, not authorization.
## Current state

- Public repository: <https://github.com/ShamelesAbyss/Quantum-Morph-Laboratory>
- Development branch: `main` only.
- Local workspace: `C:\Users\Shame\.gemini\antigravity\scratch\Quantum-Morph-Laboratory` (Windows 11, Intel Core i9-14900HX, RTX 4070 Laptop GPU 8 GB VRAM, 32 GB RAM, 2560×1440 144 Hz display).
- Local pretrained model caches (NEVER commit to Git):
  - `C:\Users\Shame\.gemini\antigravity\scratch\qml-models-cache` (`smollm2-135m`, `smollm2-360m`, `smollm2-1.7b`, `qwen2.5-1.5b`, `qwen2.5-3b`)
  - `D:\qml-models-cache\qwen2.5-7b` (`Qwen/Qwen2.5-7B-Instruct`, 4 shards, `14.19 GiB` BF16)
  - `D:\qml-work\qwen25-7b` (cached `_cal_profile.qcal`, `_ref_logits.qref`, and forged `int4/int3/int2/qmr-200/qmr-175/ternary` QMF v1 artifacts)
- Previous completed milestone: **`v0.6.0` — PROJECT EVENT HORIZON Wave III** (Layer-by-Layer Streaming BF16 Calibration & Reference Engine `< 450 MiB` peak VRAM for `7B–32B`, GPU-Resident Per-Head INT8 KV-Cache `~3.88×` smaller VRAM, 10th Persistent GPU Kernel `MATVEC_VQ_WGSL`, `QCAL1`/`QREF1` binary disk caches, activation-weighted `fit_vq_calibrated`, and `exp-006` Live Stage D **`Qwen2.5-7B-Instruct`** 8 GB GPU Breakthrough breaking **`2.251 bpp` (`1.85 GiB`, `PPLx = 1.43×`, `65.7%` Top-1)** and **exact `2.000 bpp` (`1.63 GiB`, `PPLx = 2.13×`, `52.6%` Top-1, `8.00×` smaller than BF16)**!).
- **Permanent Release Distribution Policy:** All user-facing releases use clean semantic versions (`v0.3.0`, `v0.4.0`, `v0.5.1`, `v0.6.0`, ...) and MUST be published as full, visible public GitHub Releases (`draft: false`, `prerelease: false`, `make_latest: true`) with all 4 compiled assets (`Quantum-Morph-Laboratory-<tag>-Windows-x64.exe`, `Quantum-Morph-Laboratory-<tag>-Windows-x64-Portable.zip`, `Quantum-Morph-Laboratory-<tag>-Linux-x64.tar.gz`, `SHA256SUMS.txt`), full `RELEASE_NOTES.md`, and only an ordinary informational native Releases link in README. Never delete or overwrite historical tags/releases.
- The workspace contains **13 cohesive safe-Rust crates** with `#![forbid(unsafe_code)]` and zero Python dependencies.
- **License:** MIT ([LICENSE](LICENSE)). All workspace crates inherit `license.workspace = true`.
- **Permanent Memorial Dedication:** The memorial dedication to Ashlan Haley Rapponotti at the end of `README.md` (and `RELEASE_NOTES.md`) MUST be preserved verbatim across all future updates and releases.

## Implemented capabilities

- `qml-core`: Dense linear algebra, SplitMix64 PRNG, deterministic one-sided Jacobi exact thin SVD, HMT randomized SVD, spectral metrics, Grassmannian subspace affinity, and pure safe-Rust FIPS 180-4 SHA-256.
- `qml-hardware`: Cross-platform CPU/SIMD/RAM telemetry, hardware-only WGPU compute selection (skips `DeviceType::Cpu` software emulators), **10 persistent VRAM WGSL matrix kernels** (`FP32`, `FP16`, `BF16`, `INT8_SYM`, `Q4_0`, `Q4_1`, `INT3_ASYM`, `INT2_ASYM`, `TERNARY`, **`VECTOR_QUANT`**), **8 full-transformer GPU-resident WGSL kernels** (`RMSNORM`, uniform-barrier `HADAMARD_FWD`, `ROPE_AND_KV_APPEND` FP32 & INT8, `GQA_ATTN` FP32 & INT8, `SWIGLU`, `VEC_ADD`) in `gpu_full.rs`, and `GPU_SCRATCH_MAX_ELEMS = 262_144`.
- `qml-model`: Unified single + multi-shard `SafetensorsSource` (`PayloadBudget` up to `8 GiB/tensor`, `64 GiB/total`, direct row-seek `read_tensor_rows_f64`, zero-`f64` `read_tensor_bf16_bits`) and GGUF v2/v3 readers with exact safe-Rust codecs (`F64..FP8`, `Q8_0`, `Q4_0`, `Q4_1`, `Q2_K..Q6_K`) and HF `config.json` parser.
- `qml-scope`: Two-pass streaming Welford statistics, outliers, histograms, SVD spectra, role classification, and cross-layer Grassmannian affinity.
- `qml-quant`: Conventional baselines, RMQ Shared-Basis/Core, `MAP`/`MAP-B`, PROJECT RESONANCE (`HadamardSpec`, `Int3`/`Int2`/`Ternary`/`VectorQuant`/`Echo`), Blockwise Cholesky GPTQ (`gptq.rs`), `fit_vq_calibrated`, and compact binary `QCAL1` Hessian disk persistence.
- `qml-format`: Experimental QMF v0 & multi-GB QMF v1 (`QmfFileReader`, `write_to_file_atomic`, streaming CRC-32 + SHA-256, `Arc` shared-latent deduplication, sub-3-bit operators, `decode_row`).
- `qml-forge`: Streaming single & multi-shard Safetensors → QMF v1 pipeline (`forge_path_to_file_atomic`) with calibrated GPTQ & RVQ, cancellation, and verification.
- `qml-tokenizer`: Pure safe-Rust Hugging Face `tokenizer.json` Byte-Level BPE tokenizer with ChatML formatting and stateful UTF-8 `StreamingDecoder`.
- `qml-runtime`: Native Llama & Qwen2 Transformer (`q/k/v` biases, FP32/FP64 & GPU-resident INT8 KV-cache, `Cpu`, `GpuHybrid`, single-submission `GpuFull`), plus **`streaming.rs`** Layer-by-Layer Streaming BF16 Calibration & Reference Evaluator (`< 450 MiB` peak VRAM on 7B).
- `qml-behavior`: Layer-by-layer hidden tracing, RSA, reversible causal ablations, `ReferenceSuiteCache` with binary `QREF1` disk caching, and 4-category held-out Capability Retention Suite.
- `qml-genesis`: Q-Genesis Extreme role/depth-stratified multi-objective NSGA-II search + Genome-to-QMF v1 export.
- `qml-cli`: 15 laboratory commands supporting sharded Safetensors, `--streaming-ref`, `--kv-cache int8`, `--calibrate`, `rvq-200/175/150`, and `GpuFull`.
- `qml-desktop`: PROJECT PRISM native WGPU 3D scientific observatory (`eframe` WGPU + `wgpu 24`, 7 custom WGSL shaders, `Rgba16Float` HDR bloom, 11 real-engine views, `GpuFull` default GPU inference, Consumer VRAM Fit Matrix).

## Validation actually executed on `v0.6.0`

All of the following passed locally on Windows 11 with Rust `1.100.0` (`cargo +1.100.0`) and `cargo-deny 0.21.0`:

- `cargo +1.100.0 fmt --all -- --check`
- `cargo +1.100.0 clippy --workspace --all-targets -- -D warnings`
- `cargo +1.100.0 test --workspace` — all **88 automated tests** across **13 crates** passed (including all 18 WGSL GPU compute shaders, `MATVEC_VQ_WGSL` parity, GPU-resident INT8 KV-cache parity + VRAM reduction, Layer-by-Layer Streaming BF16 calibration parity, Cholesky GPTQ + Hadamard rotation proofs, GGUF K-quants, and PRISM WGPU rendering pipelines on NVIDIA RTX 4070).
- `cargo +1.100.0 build --release -p qml-desktop -p qml-cli` + smoke-test launch of `qml.exe --version` (`qml 0.6.0`) and `qml-desktop.exe`.
- `cargo deny check` — `advisories ok, bans ok, licenses ok, sources ok` (zero ignored vulnerabilities).
- **Live Real-Model GPU Experiments (`exp-006`, RTX 4070):** 4-shard `14.19 GiB` BF16 `Qwen2.5-7B-Instruct` streamed at `444.5 MiB` peak VRAM (`196.8s`) and evaluated on `GpuFull` + GPU INT8 KV-cache across `INT4` (`0.98×` PPLx / `90.9%` Top-1), `INT3` (`1.07×` / `86.3%`), **`INT2` (`2.251 bpp`, `1.85 GiB`, `1.43×` / `65.7%`, `9.7 tok/s`)**, **`QMR-200` (`2.000 bpp`, `1.63 GiB`, `2.13×` / `52.6%`, `9.1 tok/s`)**, plus honest `1.75/1.73 bpp` cliff boundaries and `SmolLM2-360M` GPU `MATVEC_VQ_WGSL` negative control (`experiments/exp-006-event-horizon-wave3-7b-sub2bit.toml`).

## Known limits and blockers

- QMF v0/v1 remain experimental research formats and are not yet compatibility-stable across major milestones.
- Both `QMR-175_GPTQ` (`1.750 bpp`) and `TERNARY_HADAMARD_GPTQ` (`1.727 bpp`) still cliff at 7.6B; uncompensated 2-stage RVQ also cliffs without Cholesky `H⁻¹` compensation. Breaking `< 2.00 bpp` requires Hessian-compensated lattice/trellis quantization (QuIP#/QTIP-style E8 + block-128 GPTQ).
- `MAP-B` Pass 1 currently evaluates 8 full quantizers across all rows (~15 min on 7.6B); sampled-row screening will speed up `14B–32B` allocation ~10×.
- Safe cross-platform WGPU intentionally does not expose GPU temperature, board power draw, or OS-wide VRAM usage.

## Future roadmap — NOT AUTHORIZED by PROJECT AEGIS (Wave IV — `v0.7.0`)

1. **Stage E (`14B`, e.g., `Qwen2.5-14B-Instruct`) & Stage F (`27B–32B`, e.g., `Qwen2.5-32B-Instruct`) on 8 GB RTX 4070:** Implement sampled-row `MAP-B` Pass 1 screening and resumable layer-checkpointed Q-Forge disk streaming, then run `14B` (`~3.6 GiB` at `2.00 bpp`) and `32B` (`~7.5 GiB` at `1.75–2.00 bpp`) live on 8 GB VRAM!
2. **Hessian-Compensated Lattice/Trellis Quantization (`< 1.75 bpp` Breakthrough):** Pair our GPU-resident `MATVEC_VQ_WGSL` / E8 lattice decoder with blockwise Cholesky `H⁻¹` compensation and PV-Tuning coordinate refinement to push `7B–32B` cleanly into `1.75–1.50 bpp`.
3. **Extended Architectures & Calibrated Q-Genesis:** Add Mistral/Gemma/MoE support and connect `qml-genesis` to cached `QCAL1`/`QREF1` profiles.
