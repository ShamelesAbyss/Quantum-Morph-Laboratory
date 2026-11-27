# PROJECT COLOSSUS — experiment 007

Status: INT3 construction and evaluation completed; INT2 and QMR-200 remain
in progress. Results apply only to this small fixed experiment.
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

All six original shards passed their official SHA-256 identities on November
27, 2026. Native bounded metadata inspection counted **14,770,033,664 parameters**,
all BF16. Shards total 29,540,133,960 bytes including headers; tensor payloads
total 29,540,067,328 bytes.

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
Before any candidate output, three format regressions also verify that the
strict three-line task accepts one terminal line ending but rejects extra blank
lines and leading/trailing spaces. The previous oracle is preserved; both
14-case positive and negative controls pass again with the corrected guard.
Calibration/reference data and task expected values remain unchanged.

Calibration uses 17 separate programming/CS passages, including nine actual
Rust/Python/JavaScript code passages, disjoint from all task prompts and targets:
**1,340 native-tokenizer tokens**. A separate seven-passage teacher-forced corpus
contains **125 next-token predictions** and supplies
streamed BF16 reference logits. It measures numerical fidelity, independently
of task pass rates. Calibration Hessian block 64, Hadamard block 64, quantization
group 128 and seed 42 preserve existing calibrated GPTQ mathematics.
QCAL2 profiles are sharded by tensor using the existing bounded verified cache
serializer; QREF2 references are sharded by example. Every shard binds pinned
source inventory, verified file hashes, original config/tokenizer bytes,
datasets, effective runtime settings and execution-identity domain.

Protocol revision 1's eight prose passages produced only 283 calibration tokens.
Before quantization or any candidate output, its preliminary streaming reference
pass was deliberately stopped after 43.806 seconds (six layer progress entries);
no completed cache or candidate result was produced. Its protocol, provenance,
partial logs and sampled process measurements remain preserved as v1 evidence.
Revision 2 adds code calibration coverage. All 14 held-out tasks/assertions,
generation settings and seven numerical reference passages are unchanged.
The [complete frozen inputs and research harness](COLOSSUS_REPRODUCTION.md)
are published for inspection. These tasks are held out from calibration, with
no claim that their general concepts are absent from model pretraining.

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

The native streaming BF16-weight reference and candidate GPU execution paths
use their existing accumulation, attention and KV implementations. No independent
Hugging Face runtime/tokenizer parity check is claimed. Any numerical difference
may include backend arithmetic as well as compression. The 1,340-token
calibration set and 125-prediction reference corpus remain small.

Existing Q-Forge fast paths populate some Frobenius-named diagnostic fields
with matvec-probe surrogates. Those fields are not used as independently
measured dense Frobenius errors or as quality conclusions here. Correcting
that separate legacy reporting issue is outside the authorized application
change, which remains limited to throughput reporting.

## Throughput correction

Generation throughput now counts emitted tokens over the entire generation
call, including prefill and callback work. Prompt tokens are excluded from the
numerator. This is explicitly an end-to-end generation rate, not a decode-only
measurement. Ladder evaluation throughput always counts evaluated predictions
over evaluation wall time; sample-generation rate is a separate optional JSON
field and is null if generation did not run or failed. Historical measurements
are preserved and are not retroactively recalculated without raw timing data.

## Results and reproducibility

The first completed phase is the original BF16-weight streaming reference.
Computation took **1,148.589287 seconds** with GPU use reported. The existing
layer-weight GPU counter peaked at **550,502,512 bytes (525.0 MiB)**.
Externally sampled process working set peaked at **3,708,489,728 bytes
(3.45 GiB)** at one-second sampling; this is not a guaranteed OS high-water mark.
The full process took **1,881.6290715 seconds**, including the existing safe
cache serialization/publication overhead. These timing scopes are distinct.

The collected profile accounts for 1,114,480,640 bytes of Hessian/RMS values in
memory. Its 337 verified tensor shards total 557,461,943 bytes; seven verified
reference-cache files total 76,034,012 bytes. This is a completed native
reference/calibration result for the specified checkpoint and corpus, not a
measured original-model autoregressive coding baseline or a general 14B claim.

Remaining candidate results are pending the actual runs. The local frozen protocol, Rust driver, objective
oracle, command logs and resource measurements are under
`D:\QML-COLOSSUS`. Final summaries belong in experiment 007, README,
CHANGELOG, release notes and HANDOFF. A public v0.6.5 release is not complete
until exact-tag validation and independent native public-release verification
pass with all six mandatory assets.

INT3 GPTQ construction completed successfully. The complete QMF is
**6,003,510,569 bytes**, **3.2517247857777125 bpp**, SHA-256
`181e5c6afa8cf1af9399b44212b9d8bc1e4be6e2c6a278d88616fed5e475a5c3`.
The measured forge call took 1,180.6526261 seconds, including safe publication;
its process took 1,191.0513974 seconds including setup/cache loading.
One-second process sampling recorded a 14,042,476,544-byte working-set peak
(13.08 GiB). This illustrates quantizer workspace beyond ATLAS builder-owned
limits. All seven numerical evaluations and fourteen task checks are complete.

**INT3 passed 11/14 tasks.** Category counts are Rust 1/2, Python 2/2,
JavaScript 2/2, debugging 2/2, algorithmic reasoning 1/2, code explanation 2/2
and instruction following 1/2. The run-encoding function compiles but never
updates its initial run value; the mixed-run assertion fails. The graph answer
has the correct path but reports weight 7 instead of 4. The JSON-only answer's
values are correct after extraction, but its Markdown fence violates the
explicit instruction. Outputs are unchanged in the
[raw answer record](COLOSSUS_TASK_OUTPUTS.md).

All fourteen outputs reached EOS, with zero output-limit truncations. Only
1/14 outputs had literal format compliance; single fenced code/JSON blocks
remain eligible for functional testing except when an instruction explicitly
forbids Markdown. This distinction is fixed in the published oracle.

Generation emitted 827 tokens over 1,549.205938 seconds of generation-call time:
**0.533821863 emitted tokens/s including prefill**, not decode-only throughput.
The short numerical corpus measured reference perplexity 36.5527760797859,
candidate perplexity 36.93154912153539, ratio 1.010362360465392, mean KL
0.0550587877294491 nats, and 107/125 matching top-1 predictions (85.6%).
The existing composite numerical-fidelity formula yields 92.28719583830207%;
this is not task capability or retained intelligence. Numerical evaluation
took 101.0093159 seconds (1.237509619 predictions/s), a separate metric.
The full evaluation process took 1,892.3775918 seconds and sampled peak
working set was 13,741,355,008 bytes (12.80 GiB).

These are objective observations on fourteen custom tasks. The errors and
latency limit practical conclusions; no original BF16 task baseline, broad
benchmark capability, generally useful assistant or measured quality-retention
claim follows.

The artifact loaded as WGPU Full-Resident on the RTX 4070 Laptop GPU/Vulkan:
337 reported GPU operators, zero reported CPU fallback operators and
5,687,212,304 bytes of GPU weight accounting (5.30 GiB). Whole-file integrity
revalidation took 148.9443314 seconds; model loading then took 89.6171724 seconds.
The reported counters exclude host embedding-row access, tokenization and
other CPU work, and are not total device allocation.

## Native use at the measured context

A separate read-only `compressed-qmf/runtime-config-1024.json` changes only
the copied configuration's maximum context to 1024. Structural comparison
against the original passed with exactly that one intentional field change.
Original configuration/tokenizer bytes remain unchanged. The native runtime
requires the QMF, this 661-byte configuration and the 7,031,645-byte tokenizer:
INT3's minimum three-file inference set is 6,010,542,875 bytes. Calibration
caches and original weights are retained experimental data, not inference
inputs. The published Ollama figure is approximate storage context, with
no measured quality or speed comparison.

For exploratory native CLI use, keep prompt plus requested output within the
measured 1024-token window. The original advertised 32768-token context is
not evaluated. This command uses the application's default chat formatting;
the frozen Rust harness above supplies the exact benchmark system/prompt:

```powershell
qml generate D:\QML-COLOSSUS\compressed-qmf\int3.qmf --config D:\QML-COLOSSUS\compressed-qmf\runtime-config-1024.json --tokenizer D:\QML-COLOSSUS\original-model\tokenizer.json --backend gpu-full --kv-cache int8 --chat --prompt "Your short coding question" --max-tokens 256 --temperature 0
```

No additional generation is run or scored by documenting this example.
