# Whisper Base quantization

How small can a saved Whisper Base model become while keeping its word error
rate within 5% of the original model?

I tested 8-, 4-, 3-, and 2-bit weight precision, then searched for allocations
that give different parts of the same model different precision. The final
search exported and evaluated 96 models. **INT8 at 76.43 MB was the smallest
validated result.** A 55.66 MB candidate passed development and the first
independent test, but failed the second.

## Hypothesis

Some attention and feed-forward matrices might tolerate low precision while
others need 8 bits. Allocating precision selectively could therefore save more
space than uniform INT8 without materially increasing transcription errors.

The checkpoint stayed fixed: `openai/whisper-base`. This was post-training,
**weight-only** compression using OpenVINO and NNCF, with no fine-tuning, pruning,
or smaller architecture. Packed weights do not imply equally low-bit activations
or CPU arithmetic. FP32 names the stored baseline weights; execution used the
OpenVINO runtime defaults.

The objective was minimum saved size, subject to `WER <= 1.05 × FP32 WER` on
**each** independent clean set. A 5.00% baseline allows 5.25%, not 10.00%.
Sizes sum saved model files, including graphs and quantization parameters but
excluding the experiment's separate `variant.json` record. MB is decimal;
displayed sizes are rounded.

## How I searched

Early INT8 allocations produced similar scores, so I moved below 8 bits.
Uniform 2-bit compression severely damaged transcription. Earlier 4-bit and
gradient-guided allocations looked better on development than on new clips.
That motivated a search driven by measured word errors, with sensitivity
proxies used only to choose the next experiment.

The final search varied 96 attention/feed-forward matrices. The two convolutions,
token embedding, and output projection stayed INT8 to focus the search on the
transformer matrices. This restriction was not proof that those operations
must always stay at 8 bits.

| Trials | Exports | Reasoning |
|---|---:|---|
| Component ablations | 20 | Flip each of ten groups between 4 and 8 bits from opposite anchors, measuring both compression damage and the benefit of restoring precision. |
| Component allocations | 30 | Fit a regularized WER regression with pairwise group interactions. Rank all 1,024 possible group masks cheaply, propose six size targets from 57–72 MB, export them, and refit after each batch. Swaps let precision move between groups. |
| Individual matrix flips | 18 | Refine three promising allocations. Rank changes by activation-weighted reconstruction error per estimated byte, then measure actual WER. |
| Quantizer variants | 16 | Compare group sizes 16/32/64, symmetric/asymmetric INT4, and supported calibration corrections. Smaller groups cost more scale storage but may reduce rounding error. |
| Isolated low-bit probes | 12 | Lower selected matrices to 3 or 2 bits, instead of lowering an entire stage. |

The reconstruction proxy was `sum((W_quantized − W)² × 2 E[x²])`, using two
development clips. This approximates a diagonal Hessian of local output
reconstruction, **not** the task-loss Hessian or the full
[HAWQ-V2 algorithm](https://arxiv.org/abs/1911.03852). The component regression
guided limited exports; it did not establish a global optimum.

NNCF performed quantization and packing. The project-specific work was the graph
inventory, allocation search, export verification, and evaluation. For graphs
containing both INT2 and INT3, NNCF could not process an already packed low-bit
graph. I generated the two branches from the same 4/8-bit graph and replaced
the relevant weight inputs. Every candidate was saved, reopened, checked for
the requested packed types, and measured in actual bytes.

AWQ and GPTQ first had to export and transcribe two clips. AWQ produced payloads
identical to plain INT4 and was recorded as skipped; symmetric INT4 filled its
slots. GPTQ changed weights and passed generation, but all four tested variants
increased screen errors against their matched plain parents. Two calibration
clips are insufficient to conclude that GPTQ is generally worse. These options
use [NNCF's OpenVINO compression workflow](https://docs.openvino.ai/2026/openvino-workflow/model-optimization-guide/weight-compression/4-bit-weight-quantization.html).

## Separating search from testing

All candidates used the same 72-clip development screen. Sixteen candidates
spanning the measured size–WER frontier and promising alternatives were scored
on the full 180-clip pool. Earlier test clips were explicitly repurposed as
development and were no longer independent evidence.

Before searching, I reserved two disjoint 150-clip LibriSpeech `test-clean` sets.
Neither overlapped development or earlier test clips. A contained 60/75/15
short/medium/long clips; B contained 60/78/12. The bands were under 10, 10–20,
and 20–30 seconds. Speakers could recur: these were unseen clips, not necessarily
unseen speakers.

Selection chose the smallest model within **2%** of development FP32 WER, with
a 5% fallback. The tighter margin was a heuristic buffer for sampling variation.
The primary and ordered runner-up were frozen before either new set's predictions
were read. Only candidates passing A proceeded to B; both sets had to pass.
Neither failure was used to revise the allocation.

All models used English, greedy decoding, temperature zero, no previous-clip
conditioning, and at most 128 new tokens on mono 16 kHz audio. WER sums word
substitutions, insertions, and deletions over the corpus, divided by reference
words. NFKC, case, punctuation, and whitespace normalization happen first;
punctuation differences alone do not increase the score.

## Results

| Model | Saved MB | Development WER | Set A WER | Set B WER | Both within 5%? |
|---|---:|---:|---:|---:|---|
| FP32 control | 290.95 | 5.587% | 6.185% | 4.848% | Reference |
| Uniform INT8 | 76.43 | 5.512% | 6.116% | 4.825% | Yes |
| Primary: 95 INT4 matrices + one INT2 | 55.66 | 5.677% | 6.231% | 5.439% | No |
| Runner-up: 95 INT4 matrices + one INT3 | 55.80 | 5.632% | 6.507% | Not evaluated | No: failed A |

Both finalists used asymmetric INT4, group size 64, and no calibration correction.
The primary lowered decoder layer 1's feed-forward output matrix (`fc2`) to
INT2; the runner-up lowered layer 0's `fc2` to INT3. The four boundary operations
stayed INT8; other fixed tensors retained floating point. Exact identities and
allocations are recorded in [selection.json](results/provenance/selection.json) and the
measurement bundle.

A's FP32 control made **269 edits / 4,349 words**, allowing at most 282 edits.
The primary made 271 (+0.74% relative WER); the runner-up made 283 (+5.20%),
missing by one edit. B's control made **213 edits / 4,394 words**, allowing at
most 223. The primary made 239 (+12.21%) and failed.

On B, the primary gained errors on 27 clips and lost errors on 10: 26 extra
edits overall. Its 95% paired speaker-bootstrap interval for the absolute WER
difference was −0.021 to +1.163 percentage points. This wide interval reflects
uncertainty; it does not override the failed point-estimate criterion. All
2,000-resample intervals are in [confirmation.csv](results/tables/confirmation.csv).

**INT8 reduced saved size by 73.7% from FP32 and passed both confirmations.**
The primary was another 27.2% smaller than INT8, but missed the accuracy goal.
Only the two frozen finalists received independent testing; other allocations
have development evidence only.

### Secondary checks

| Model | Harder speech WER | 10 dB white-noise WER | Median CPU generation / clip |
|---|---:|---:|---:|
| FP32 | 12.870% | 10.439% | 0.619 s |
| INT8 | 12.772% | 10.485% | 0.540 s |
| Primary | 13.167% | 12.486% | 0.489 s |
| Runner-up | 13.562% | 11.382% | 0.561 s |

Harder speech used 120 fresh `test-other` clips under 20 seconds. White noise
was added to A and saved as float audio to avoid clipping. Timing used 12 matched
clips, one warmup per clip and three measured repeats, on Windows 11 with an
Intel Family 6 Model 183 CPU. It excludes features, loading, and compilation;
it is not an end-to-end or peak-memory benchmark. See [environment.json](results/provenance/environment.json).

## Conclusion

The development margin did not predict both independent outcomes. Eighty-two
of the 96 candidates were within 5% on the short screen; that was weak evidence
for deployment. The experiment demonstrates the difference between finding a
promising allocation and validating one, including the failures that exposed it.

This is a bounded search on audiobook speech up to 30 seconds, not proof of an
optimal allocation or population-wide equivalence. Real environmental noise,
other accents, power, peak RAM, and NPU execution were not measured. A follow-up
would use broader development speakers and more calibration clips, then reserve
new test data before selecting another allocation.

## Checking the work

```text
.github/workflows/         automated evidence checks on GitHub
configs/search.json        pinned reproduction settings
requirements/experiment.txt  pinned inference and test dependencies
src/whisper_precision/     Python package
    cli.py                 search, freeze, confirm, and diagnose
    prepare.py             development inputs and control exports
    verify.py              audit recorded results without inference
    quantization.py        per-matrix packing and export verification
    search.py              allocation proposals and resumable evaluation
tests/                     scoring, packed types, selection, and cache checks
results/
    tables/                readable candidate and confirmation CSVs
    raw/                   compressed original records and development inputs
    provenance/            frozen selection, environment, checksums, source
pyproject.toml             package metadata and command-line entry points
```

Using Python 3.13, install the repository tools once, then verify the evidence:

```console
python -m pip install --no-deps -e .
python -m whisper_precision.verify
```

The verifier uses Python's standard library, with no ML dependencies. It
recomputes transcript WER, checks clip separation, frozen selection and source
hashes, packed-type records, and ordered A/B decisions. The full search is listed
in [candidates.csv](results/tables/candidates.csv); blank development cells mean a model
was screened but not scored on all 180 clips.

`results/raw/measurements.json.gz` contains the original JSON records and predictions.
`results/raw/inputs.json.gz` contains only development inputs and exclusions. A small
numerical source snapshot preserves the exact experiment code through this
cleanup. Checksums detect accidental changes, not independent certification.

For inference and tests, the tested setup is Python 3.13 on Windows:

```powershell
py -3.13 -m venv .venv313
.\.venv313\Scripts\python.exe -m pip install -r requirements/experiment.txt
.\.venv313\Scripts\python.exe -m pip install --no-deps -e .
.\.venv313\Scripts\python.exe -m unittest discover -s tests
.\.venv313\Scripts\python.exe -m whisper_precision.prepare --check
```

To repeat the expensive search, run `python -m whisper_precision.prepare` without
`--check`, then `python -m whisper_precision.cli all`. The recorded candidate models total 6.02 GB; allow additional
space for controls, downloads, and caches. Outputs go to ignored
`artifacts/reproduction/`. Individual commands
are `prepare`, `preflight`, `search`, `select`, `final`, and `diagnostics`.
The cleanup passed 28 tests and a new two-clip export/transcription check that
matched the recorded predictions. The cleaned runner has not repeated the entire
search; the published numbers come from the preserved original run. Reusing
these published clips reproduces that experiment and is not a new independent
validation. Model files, audio,
local archives, and caches are excluded from Git.

The GitHub workflow runs scoring/evidence tests and verifies the published records.
The full 28-test suite additionally needs the experiment dependencies. Line-ending
rules preserve the checksummed evidence bytes when cloning on Windows or Linux.
