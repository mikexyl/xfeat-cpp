#!/usr/bin/env python3
"""Sweep VPR group sizes on all GRACO ground runs and plot loop-event metrics."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("build/examples/jist_mixvpr_recall_example"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/graco"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "ground-01",
            "ground-02",
            "ground-03",
            "ground-04",
            "ground-05",
            "ground-06",
        ],
        help="Dataset names resolved below --dataset-root",
    )
    parser.add_argument(
        "--jist-engine",
        type=Path,
        default=Path("onnx_model/trt/JIST_r18_512_seqgem_frames_fp16.engine"),
    )
    parser.add_argument(
        "--mixvpr-engine",
        type=Path,
        default=Path("onnx_model/trt/mixvpr_resnet50_512d_fp16.engine"),
    )
    parser.add_argument(
        "--xfeat-engine",
        type=Path,
        default=Path("onnx_model/trt/xfeat_320x224_fp16.engine"),
    )
    parser.add_argument(
        "--lighterglue-engine",
        type=Path,
        default=Path("onnx_model/trt/lg_320x224_dyn_fp16.engine"),
    )
    parser.add_argument("--xfeat-top-k", type=int, default=500)
    parser.add_argument("--verification-min-matches", type=int, default=20)
    parser.add_argument("--ransac-min-inliers", type=int, default=15)
    parser.add_argument("--ransac-min-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--ransac-reprojection-threshold", type=float, default=2.0)
    parser.add_argument("--ransac-confidence", type=float, default=0.999)
    parser.add_argument("--ransac-max-iterations", type=int, default=2000)
    parser.add_argument("--n-skip", type=int, default=5)
    parser.add_argument("--n-seq", type=int, nargs="+", default=[5, 10, 15, 30])
    parser.add_argument("--positive-distance", type=float, default=5.0)
    parser.add_argument("--positive-yaw-degrees", type=float, default=30.0)
    parser.add_argument("--min-time-separation", type=float, default=30.0)
    parser.add_argument("--gt-sample-period", type=float, default=1.0)
    parser.add_argument("--temporal-match-tolerance", type=float, default=5.0)
    parser.add_argument("--pose-max-dt", type=float, default=0.02)
    parser.add_argument("--jist-threshold", type=float, default=0.9)
    parser.add_argument("--mixvpr-threshold", type=float, default=0.6)
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Number of concurrent C++ TensorRT processes",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="Limit each run for testing; 0 processes every complete group",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/jist_mixvpr_nseq_sweep"),
    )
    parser.add_argument("--verbose-engine", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from output-dir/sweep_results.csv without inference",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the Matplotlib figure interactively after saving it",
    )
    args = parser.parse_args()

    if args.n_skip <= 0 or any(value < 5 for value in args.n_seq):
        parser.error("--n-skip must be positive and every --n-seq value must be >= 5")
    if len(set(args.n_seq)) != len(args.n_seq):
        parser.error("--n-seq values must be unique")
    if not args.datasets or len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets must be non-empty and unique")
    if args.gt_sample_period <= 0:
        parser.error("--gt-sample-period must be positive")
    if not 0 <= args.positive_yaw_degrees <= 180:
        parser.error("--positive-yaw-degrees must be in [0, 180]")
    if args.temporal_match_tolerance < args.gt_sample_period:
        parser.error(
            "--temporal-match-tolerance must be at least --gt-sample-period"
        )
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if (
        args.xfeat_top_k <= 0
        or args.verification_min_matches < 8
        or args.ransac_min_inliers < 8
        or not 0 <= args.ransac_min_inlier_ratio <= 1
        or args.ransac_reprojection_threshold <= 0
        or not 0 < args.ransac_confidence < 1
        or args.ransac_max_iterations <= 0
    ):
        parser.error("XFeat/LightGlue/RANSAC verification parameters are invalid")
    return args


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_benchmark(args: argparse.Namespace, n_seq: int, output_dir: Path) -> list[dict[str, str]]:
    metrics_path = output_dir / f"metrics_nseq_{n_seq}.csv"
    log_path = output_dir / f"benchmark_nseq_{n_seq}.log"
    gt_loops_path = output_dir / f"ground_truth_loops_nseq_{n_seq}.csv"
    executable = project_path(args.executable)
    command = [
        str(executable),
        "--dataset-root",
        str(args.dataset_root),
        "--datasets",
        *args.datasets,
        "--jist-engine",
        str(args.jist_engine),
        "--mixvpr-engine",
        str(args.mixvpr_engine),
        "--xfeat-engine",
        str(args.xfeat_engine),
        "--lighterglue-engine",
        str(args.lighterglue_engine),
        "--xfeat-top-k",
        str(args.xfeat_top_k),
        "--verification-min-matches",
        str(args.verification_min_matches),
        "--ransac-min-inliers",
        str(args.ransac_min_inliers),
        "--ransac-min-inlier-ratio",
        str(args.ransac_min_inlier_ratio),
        "--ransac-reprojection-threshold",
        str(args.ransac_reprojection_threshold),
        "--ransac-confidence",
        str(args.ransac_confidence),
        "--ransac-max-iterations",
        str(args.ransac_max_iterations),
        "--n-skip",
        str(args.n_skip),
        "--n-seq",
        str(n_seq),
        "--positive-distance",
        str(args.positive_distance),
        "--positive-yaw-degrees",
        str(args.positive_yaw_degrees),
        "--min-time-separation",
        str(args.min_time_separation),
        "--gt-sample-period",
        str(args.gt_sample_period),
        "--temporal-match-tolerance",
        str(args.temporal_match_tolerance),
        "--pose-max-dt",
        str(args.pose_max_dt),
        "--jist-threshold",
        str(args.jist_threshold),
        "--mixvpr-threshold",
        str(args.mixvpr_threshold),
        "--max-sequences",
        str(args.max_sequences),
        "--metrics-csv",
        str(metrics_path),
        "--gt-loops-csv",
        str(gt_loops_path),
    ]
    if args.verbose_engine:
        command.append("--verbose")

    print(f"Starting n_seq={n_seq}; log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        log_tail = "\n".join(log_lines[-40:])
        raise RuntimeError(
            f"Benchmark failed for n_seq={n_seq} with exit code "
            f"{process.returncode}; see {log_path}\n{log_tail}"
        )

    with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        rows = list(csv.DictReader(metrics_file))
    expected_metrics = {
        (method, mode, stage)
        for method, mode in (
            ("JIST", "last-frame"),
            ("JIST", "temporal-any"),
            ("JIST", "frame-argmax"),
            ("MixVPR", "last-frame"),
        )
        for stage in ("retrieval", "verified")
    }
    actual_metrics = {
        (row["method"], row["evaluation_mode"], row["verification_stage"])
        for row in rows
    }
    if actual_metrics != expected_metrics:
        raise RuntimeError(f"Unexpected metrics in {metrics_path}")
    print(f"Completed n_seq={n_seq}", flush=True)
    return rows


def consolidate_ground_truth(args: argparse.Namespace, output_dir: Path) -> Path:
    gt_paths = [
        output_dir / f"ground_truth_loops_nseq_{n_seq}.csv" for n_seq in args.n_seq
    ]
    reference_bytes = gt_paths[0].read_bytes()
    for path in gt_paths[1:]:
        if path.read_bytes() != reference_bytes:
            raise RuntimeError(
                "Independently extracted GT loops differ between concurrent runs: "
                f"{gt_paths[0]} and {path}"
            )
    consolidated_path = output_dir / "ground_truth_loops.csv"
    shutil.copyfile(gt_paths[0], consolidated_path)
    return consolidated_path


def write_consolidated_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        raise RuntimeError("No benchmark rows were produced")
    rows = sorted(
        rows,
        key=lambda row: (
            int(row["n_seq"]),
            row["method"],
            row["evaluation_mode"],
            row["verification_stage"],
        ),
    )
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_metrics(rows: list[dict[str, str]]) -> None:
    gt_counts = {int(row["ground_truth_loops"]) for row in rows}
    if len(gt_counts) != 1:
        raise RuntimeError(
            "GT-loop denominator changed across methods or n_seq values: "
            f"{sorted(gt_counts)}"
        )

    for n_seq in {int(row["n_seq"]) for row in rows}:
        jist_rows = {
            (row["evaluation_mode"], row["verification_stage"]): row
            for row in rows
            if row["method"] == "JIST" and int(row["n_seq"]) == n_seq
        }
        for stage in ("retrieval", "verified"):
            if not {
                ("last-frame", stage),
                ("temporal-any", stage),
            } <= jist_rows.keys():
                raise RuntimeError(f"Missing JIST {stage} metrics for n_seq={n_seq}")
            for metric in ("recall", "precision"):
                if float(jist_rows[("temporal-any", stage)][metric]) + 1e-12 < float(
                    jist_rows[("last-frame", stage)][metric]
                ):
                    raise RuntimeError(
                        f"JIST temporal-any {stage} {metric} is below last-frame "
                        f"for n_seq={n_seq}"
                    )

        variants = {
            (row["method"], row["evaluation_mode"])
            for row in rows
            if int(row["n_seq"]) == n_seq
        }
        for method, mode in variants:
            stage_rows = {
                row["verification_stage"]: row
                for row in rows
                if int(row["n_seq"]) == n_seq
                and row["method"] == method
                and row["evaluation_mode"] == mode
            }
            if int(stage_rows["verified"]["predicted_positives"]) > int(
                stage_rows["retrieval"]["predicted_positives"]
            ):
                raise RuntimeError(
                    f"Verification added detections for {method}/{mode}, n_seq={n_seq}"
                )
            if float(stage_rows["verified"]["recall"]) > float(
                stage_rows["retrieval"]["recall"]
            ) + 1e-12:
                raise RuntimeError(
                    f"Verification increased recall for {method}/{mode}, n_seq={n_seq}"
                )


def plot_metrics(
    rows: list[dict[str, str]], args: argparse.Namespace, output_dir: Path
) -> tuple[Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    try:
        import matplotlib

        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required for plotting; install python3-matplotlib or "
            "run this script in an environment that provides it"
        ) from error

    series_specs = (
        ("JIST", "last-frame", "JIST (last frame)", "#2563eb", "o", "-"),
        (
            "JIST",
            "temporal-any",
            f"JIST (any frame, +/-{args.temporal_match_tolerance:g} s)",
            "#059669",
            "^",
            "--",
        ),
        (
            "JIST",
            "frame-argmax",
            "JIST (frame argmax)",
            "#7c3aed",
            "D",
            "-.",
        ),
        ("MixVPR", "last-frame", "MixVPR (last frame)", "#dc2626", "s", "-"),
    )
    precision_values = [100.0 * float(row["precision"]) for row in rows]
    precision_ceiling = 5.0 * math.ceil(max(precision_values) * 1.1 / 5.0)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), sharex=True)
    for stage_index, (stage, stage_title) in enumerate(
        (("retrieval", "Before geometric verification"), ("verified", "After XFeat + LightGlue + RANSAC"))
    ):
        for metric_index, (metric, metric_title) in enumerate(
            (("recall", "Recall"), ("precision", "Precision"))
        ):
            axis = axes[stage_index][metric_index]
            for method, evaluation_mode, label, color, marker, line_style in series_specs:
                series_rows = sorted(
                    (
                        row
                        for row in rows
                        if row["method"] == method
                        and row["evaluation_mode"] == evaluation_mode
                        and row["verification_stage"] == stage
                    ),
                    key=lambda row: int(row["n_seq"]),
                )
                x_values = [int(row["n_seq"]) for row in series_rows]
                y_values = [100.0 * float(row[metric]) for row in series_rows]
                axis.plot(
                    x_values,
                    y_values,
                    color=color,
                    marker=marker,
                    linestyle=line_style,
                    linewidth=2,
                    markersize=7,
                    label=label,
                )
            axis.set_title(f"{stage_title}: {metric_title}")
            axis.set_xlabel("Logical group size $n_{seq}$")
            axis.set_ylabel(f"{metric_title} (%)")
            axis.set_xticks(sorted(args.n_seq))
            axis.set_xlim(min(args.n_seq) - 1, max(args.n_seq) + 1)
            axis.set_ylim(
                0.0,
                105.0 if metric == "recall" else max(5.0, precision_ceiling),
            )
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=8.0)

    fig.suptitle(
        "GRACO ground runs 01-06: fixed trajectory-derived GT loop queries\n"
        f"n_skip={args.n_skip}, GT={args.positive_distance:g} m/{args.positive_yaw_degrees:g} deg yaw, "
        f"temporal exclusion={args.min_time_separation:g} s, "
        f"thresholds={args.jist_threshold:g}/{args.mixvpr_threshold:g}"
    )
    fig.text(
        0.5,
        0.01,
        f"Verification: top-k={args.xfeat_top_k}, LightGlue matches>={args.verification_min_matches}, "
        f"RANSAC inliers>={args.ransac_min_inliers}, ratio>={args.ransac_min_inlier_ratio:g}, "
        f"threshold={args.ransac_reprojection_threshold:g} px. Temporal-any uses +/-"
        f"{args.temporal_match_tolerance:g} s.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.07, right=0.98, bottom=0.11, top=0.84, wspace=0.22, hspace=0.34
    )

    png_path = output_dir / "recall_precision_vs_nseq.png"
    pdf_path = output_dir / "recall_precision_vs_nseq.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    if args.show:
        plt.show()
    plt.close(fig)
    return png_path, pdf_path


def print_summary(rows: list[dict[str, str]]) -> None:
    print("\nSummary")
    print(
        f"{'n_seq':>6}  {'method':<7}  {'mode':<12}  {'stage':<9}  {'GT loops':>8}  "
        f"{'recall':>8}  {'precision':>10}  {'F1':>8}"
    )
    for row in sorted(
        rows,
        key=lambda item: (
            int(item["n_seq"]),
            item["method"],
            item["evaluation_mode"],
            item["verification_stage"],
        ),
    ):
        print(
            f"{int(row['n_seq']):6d}  {row['method']:<7}  "
            f"{row['evaluation_mode']:<12}  "
            f"{row['verification_stage']:<9}  "
            f"{int(row['ground_truth_loops']):8d}  "
            f"{100.0 * float(row['recall']):7.2f}%  "
            f"{100.0 * float(row['precision']):9.2f}%  "
            f"{100.0 * float(row['f1']):7.2f}%"
        )


def main() -> int:
    args = parse_args()
    executable = project_path(args.executable)
    if not executable.is_file():
        raise SystemExit(f"Benchmark executable does not exist: {executable}")

    output_dir = project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    consolidated_path = output_dir / "sweep_results.csv"
    if args.plot_only:
        if not consolidated_path.is_file():
            raise SystemExit(f"Consolidated CSV does not exist: {consolidated_path}")
        with consolidated_path.open(newline="", encoding="utf-8") as metrics_file:
            rows = list(csv.DictReader(metrics_file))
    else:
        rows: list[dict[str, str]] = []
        worker_count = min(args.jobs, len(args.n_seq))
        print(
            f"Launching {len(args.n_seq)} TensorRT benchmarks with "
            f"{worker_count} concurrent processes",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(run_benchmark, args, n_seq, output_dir): n_seq
                for n_seq in args.n_seq
            }
            for future in as_completed(futures):
                rows.extend(future.result())
        gt_path = consolidate_ground_truth(args, output_dir)
        print(f"Verified identical GT extraction: {gt_path}", flush=True)
        write_consolidated_csv(rows, consolidated_path)
    validate_metrics(rows)
    png_path, pdf_path = plot_metrics(rows, args, output_dir)
    print_summary(rows)
    print(f"\nResults CSV: {consolidated_path}")
    print(f"Plot PNG:    {png_path}")
    print(f"Plot PDF:    {pdf_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
