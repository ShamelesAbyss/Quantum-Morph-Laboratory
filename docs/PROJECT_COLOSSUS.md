# PROJECT COLOSSUS — experiment 007

Status: in progress; no 14B compression or task-quality result is established yet.
The owner authorized this experiment after SENTINEL. This does not authorize new
quantizers, model architectures, graphical changes or another milestone.

## Source and resource boundary

Only [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct)
is selected, pinned to revision
`aedcc2d42b622764e023cf882b6652e646b95671`. Its original BF16 Safetensors,
configuration, tokenizer, model card and Apache-2.0 license are kept under
`D:\QML-COLOSSUS\original-model`. Six shard sizes and SHA-256 identities come
from the pinned official repository inventory; every shard must verify before use.
No weights, calibration logits or generated QMF files are committed.

The preflight recorded 585,672,253,440 free bytes on D: and a 200 GiB reservation
floor. Dedicated directories contain download-cache, scratch, calibration,
reference-evaluation, compressed-qmf and experiment-results. Failed attempts,
partial downloads and logs are preserved. Existing model caches are untouched.
All builds and temporary experiment storage use the dedicated D: scratch tree.

The [official configuration](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct/blob/main/config.json)
has 48 Qwen2 layers, hidden width 5120, intermediate width 13824, 40 query heads,
8 KV heads, vocabulary 152064, SiLU, unscaled full-head RoPE and inactive sliding
window attention. The experiment uses the existing SENTINEL admission parser.
Original configuration bytes remain immutable; the effective test context is
1024 tokens and is included in cache provenance.

## Frozen evaluation

The protocol is fixed before any candidate output. Greedy generation uses seed
42, temperature 0, a 384-token output limit, EOS 151645 and the same ChatML
system/user inputs and INT8 KV setting across candidates.

Fourteen custom tasks cover two examples per requested category:

| Category | Fixed tasks | Objective checks |
|---|---|---|
| Rust generation | Consecutive run encoding; first unique Unicode character | rustc compilation and assertions, including empty/Unicode/repeated inputs |
| Python generation | Merge closed intervals; signed list rotation | Assertions for boundaries, duplicates, empty input and nonmutation |
| JavaScript generation | Stable typed deduplication; lower bound | Isolated VM assertions for duplicates, types and insertion boundaries |
| Debugging | Rust reverse-index bounds; Python mutable default | Compilation/execution and regression assertions |
| Algorithmic reasoning | Weighted shortest path; Levenshtein distance | Exact JSON path/cost/distance and specified DP complexity |
| Code explanation | Python inner-list aliasing; JavaScript map behavior | Exact predicted states and predeclared explanation fields |
| Instruction following | Sorted-name JSON; exact three-line output | Exact keys/values and formatting |

Code checks distinguish a single fenced answer from literal format compliance.
Compilation, runtime errors, truncation and incorrect answers remain in the
denominator. Pure generated snippets run with timeouts, without external
packages, network APIs or application access. Python is an optional test oracle
for the explicitly requested generated Python snippets, never an application
dependency or orchestration implementation. Synthetic known-correct controls
passed all 14 checks; deliberately wrong controls were rejected by all 14.
These control results are harness tests, not model performance.

Calibration uses eight separate programming/CS passages, disjoint from all task
prompts and targets. A separate seven-passage teacher-forced corpus supplies
streamed BF16 reference logits. It measures numerical fidelity, independently
of task pass rates. Calibration Hessian block 64, Hadamard block 64, quantization
group 128 and seed 42 preserve existing calibrated GPTQ mathematics.
QCAL2 profiles are sharded by tensor using the existing bounded verified cache
serializer; QREF2 references are sharded by example. Every shard binds pinned
source inventory, verified file hashes, original config/tokenizer bytes,
datasets, effective runtime settings and execution-identity domain.

## Representations and interpretation

Primary trials are INT3 GPTQ, INT2 GPTQ and QMR-200 using existing MAP-B, ATLAS
publication and streaming reference implementations. QMR-175 is optional only
if resources permit. No quantizer shortcut or new compression algorithm is
authorized. Failures and resource limits must be recorded explicitly.

The [published Ollama Q4_K_M listing](https://ollama.com/library/qwen2.5-coder:14b-instruct-q4_K_M)
reports approximately 9.0 GB. This is a practical published storage comparator,
not a locally measured artifact or quality comparison. No Ollama weights are
downloaded or evaluated.

Fourteen custom tasks are not a standardized benchmark and cannot establish
general programming competence. Passing functional cases does not establish
asymptotic complexity. Numerical fidelity cannot establish task correctness.
Original BF16 full-model generation exceeds currently available physical RAM;
unless an original autoregressive baseline actually runs, compressed task scores
must not be described as measured task-quality retention against BF16.

## Throughput correction

Generation throughput now counts emitted tokens over the entire generation
call, including prefill and callback work. Prompt tokens are excluded from the
numerator. This is explicitly an end-to-end generation rate, not a decode-only
measurement. Ladder evaluation throughput always counts evaluated predictions
over evaluation wall time; sample-generation rate is a separate optional JSON
field and is absent if generation did not run or failed. Historical measurements
are preserved and are not retroactively recalculated without raw timing data.

## Results and reproducibility

Pending the actual runs. The local frozen protocol, Rust driver, objective
oracle, command logs and resource measurements are under
`D:\QML-COLOSSUS`. Final summaries belong in experiment 007, README,
CHANGELOG, release notes and HANDOFF. A public v0.6.5 release is not complete
until exact-tag validation and independent native public-release verification
pass with all six mandatory assets.
