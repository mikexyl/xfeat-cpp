# JIST Frame Refinement and VPR Evaluation

This note describes the frame-selection refinement used by
`jist_mixvpr_recall_example`, how it differs from the other benchmark variants,
and how it is scored. The current refinement deliberately selects exactly one
image from each matched sequence.

## Sequence construction

For each dataset, the benchmark first keeps every `n_skip`-th image. It then
forms non-overlapping logical groups of `n_seq` retained images. Groups never
cross dataset boundaries.

The JIST TensorRT engine has a fixed input length, `T` (currently 5). When a
logical group contains more than `T` images, the benchmark selects `T` uniformly
spaced offsets:

```text
offset(t) = round(t * (n_seq - 1) / (T - 1)),  t = 0, ..., T - 1
```

This always includes the first and last image. JIST processes those images once
and returns:

- one L2-normalized 512-D sequence descriptor, and
- `T` L2-normalized 512-D per-image descriptors from the model's internal
  image branch.

MixVPR uses its ResNet-50, 512-D descriptor on the final image of each logical
group for the grouped baseline.

## Retrieval and argmax refinement

JIST retrieval is still controlled by the sequence descriptor. For a pair of
logical sequences `A` and `B`, the retrieval score is cosine similarity:

```text
sequence_score(A, B) = dot(sequence_descriptor(A), sequence_descriptor(B))
```

The pair is discarded when this score is below the active JIST threshold. For
every retained pair, frame refinement constructs the complete `T x T`
similarity matrix from the per-image descriptors:

```text
frame_score(i, j) = dot(frame_descriptor(A, i), frame_descriptor(B, j))
(i*, j*) = argmax frame_score(i, j)
```

The detection is then assigned to image `i*` in sequence `A` and image `j*` in
sequence `B`. Their timestamps, poses, and file paths replace the two sequence
endpoints for evaluation and optional geometric verification.

The frame score does **not** replace or modify the sequence retrieval score.
Threshold sweeps continue to filter on `sequence_score`; argmax only chooses
which single frame pair represents an accepted sequence match. Consequently,
the unrefined and refined JIST variants have the same number of retrieval
detections at a given threshold, but their correctness and geometric-
verification outcomes can differ.

The implementation is intentionally simple and deterministic:

- it evaluates all `T^2` frame pairs;
- it chooses one global maximum, with the first maximum winning ties;
- it does not enforce diagonal motion, monotonic alignment, or a temporal
  offset shared with neighboring sequence matches; and
- it never returns multiple frames from either sequence.

This makes `frame-argmax` an isolated test of whether JIST's per-image
descriptors can recover a better representative frame without changing the
sequence-level retriever.

## Benchmark variants

| CSV mode | Retrieval descriptor | Reported image pair |
| --- | --- | --- |
| `JIST,last-frame` | JIST sequence descriptor | Last image in each logical group |
| `JIST,frame-argmax` | JIST sequence descriptor | Highest-similarity pair among JIST's `T x T` per-image descriptors |
| `MixVPR,last-frame` | MixVPR descriptor | Last image in each logical group |
| `MixVPR,every-frame` | MixVPR descriptor | Every retained `n_skip` image independently (`n_seq=0`) |

`n_seq=0` is a special MixVPR-only reference. It does not construct JIST
sequences or load the JIST engine. It measures the best-case framewise coverage
available at the chosen `n_skip`, at substantially higher pair-search and
optional verification cost.

## Ground truth and metrics

Ground-truth loops are extracted from trajectories before any VPR model runs.
The default ground-sequence configuration samples the trajectories every 1 s
and creates a loop query when an earlier pose is within 5 m and 30 degrees of
yaw. Same-run references must be at least 30 s old; cross-run references do not
need that exclusion. Thus the GT loop inventory is identical across JIST,
refined JIST, MixVPR, and all values of `n_seq`.

For precision, a selected detection is correct when its two selected poses pass
the same translation, yaw, and same-run temporal-separation rules. For recall,
each method-independent GT loop is detected when a predicted pair with matching
dataset endpoints is assigned close enough in time to one of its valid
query/reference pairs. In the PR sweep, the assignment tolerance is the GT
sample period (1 s by default).

The fixed-threshold report also contains `JIST,temporal-any`. That is a separate
evaluation rule: it credits an unrefined JIST detection when its sequence
endpoints lie within `--temporal-match-tolerance` (5 s by default) of a GT loop.
It does not select a concrete image pair and is not the argmax refinement used
in the PR curves.

## Optional geometric verification

With `--verification`, the selected image pair from each variant is passed to:

1. XFeat local feature extraction,
2. LightGlue matching, and
3. fundamental-matrix RANSAC filtering.

This stage can only remove retrieval detections. For `frame-argmax`, it verifies
the two argmax-selected images rather than the sequence-ending images. Without
`--verification`, none of the XFeat, LightGlue, or RANSAC engines are loaded.

## Running the curves

Run the grouped, verification-off sweep:

```bash
python3 python/plot_jist_mixvpr_pr.py
```

Append the framewise MixVPR reference without geometric verification:

```bash
python3 python/plot_jist_mixvpr_pr.py --n-seq 0 --jobs 1 --append
```

Regenerate and display the consolidated figure without rerunning inference:

```bash
python3 python/plot_jist_mixvpr_pr.py --plot-only --show
```

For a server-side verified framewise run, use a separate verified output set:

```bash
python3 python/plot_jist_mixvpr_pr.py --verification
python3 python/plot_jist_mixvpr_pr.py --verification --n-seq 0 --jobs 1 --append
```

The framewise verified run may be very expensive because every detection above
the minimum MixVPR threshold is sent to geometric verification.

## Dual-output artifacts

The official endpoint/refined ablation uses one FP32 dual-output model; endpoint
mode ignores the frame-descriptor output. The generated artifacts are kept out
of Git and identified by SHA-256:

| Artifact | SHA-256 |
| --- | --- |
| Source `JIST_r18_512_seqgem_simplified.onnx` | `7a97d2ce8bd05b2a817c313ef60cc1bc184cec66d33ed1286de3ba0f65e73d4e` |
| Generated `JIST_r18_512_seqgem_frames.onnx` | `f8e3ddd7df7a33e9d203b1ab2d3c795020b401c46a49ad88a3a97edac6d38218` |
| RTX 5080 `JIST_r18_512_seqgem_frames_fp32.engine` | `e8d197bde685eeac0a111df1d48b9770dc74f386c450cac88f978cdaaa115bae` |

The engine was built independently on the RTX 5080 (compute capability 12.0)
with TensorRT 10.13.2.6 and CUDA 13.0. It is machine-specific and must not be
copied between hosts.
