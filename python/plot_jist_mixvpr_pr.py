#!/usr/bin/env python3
"""Sweep VPR thresholds and plot verification-off temporal-loop PR curves."""

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
    parser.add_argument("--verification", action="store_true")
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
    parser.add_argument("--threshold-count", type=int, default=40)
    parser.add_argument("--jist-threshold-min", type=float, default=0.8)
    parser.add_argument("--jist-threshold-max", type=float, default=0.99)
    parser.add_argument("--mixvpr-threshold-min", type=float, default=0.4)
    parser.add_argument("--mixvpr-threshold-max", type=float, default=0.9)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="Limit each dataset for testing; 0 processes all groups",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to output/jist_mixvpr_pr[_verified] according to --verification",
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Replace requested n_seq rows in an existing consolidated CSV and retain the others",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.n_skip <= 0 or any(value != 0 and value < 5 for value in args.n_seq):
        parser.error("--n-skip must be positive; --n-seq values must be 0 or >= 5")
    if len(set(args.n_seq)) != len(args.n_seq) or args.jobs <= 0:
        parser.error("--n-seq must be unique and --jobs must be positive")
    if args.threshold_count < 2:
        parser.error("--threshold-count must be at least 2")
    for minimum, maximum, label in (
        (args.jist_threshold_min, args.jist_threshold_max, "JIST"),
        (args.mixvpr_threshold_min, args.mixvpr_threshold_max, "MixVPR"),
    ):
        if not -1 <= minimum < maximum <= 1:
            parser.error(f"Invalid {label} threshold range")
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


def run_sweep(args: argparse.Namespace, n_seq: int, output_dir: Path) -> list[dict[str, str]]:
    csv_path = output_dir / f"pr_nseq_{n_seq}.csv"
    log_path = output_dir / f"pr_nseq_{n_seq}.log"
    gt_path = output_dir / f"ground_truth_loops_nseq_{n_seq}.csv"
    command = [
        str(project_path(args.executable)),
        "--dataset-root",
        str(args.dataset_root),
        "--datasets",
        *args.datasets,
        "--jist-engine",
        str(args.jist_engine),
        "--mixvpr-engine",
        str(args.mixvpr_engine),
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
        "--max-sequences",
        str(args.max_sequences),
        "--pr-curve-csv",
        str(csv_path),
        "--pr-threshold-count",
        str(args.threshold_count),
        "--pr-jist-threshold-min",
        str(args.jist_threshold_min),
        "--pr-jist-threshold-max",
        str(args.jist_threshold_max),
        "--pr-mixvpr-threshold-min",
        str(args.mixvpr_threshold_min),
        "--pr-mixvpr-threshold-max",
        str(args.mixvpr_threshold_max),
        "--gt-loops-csv",
        str(gt_path),
    ]
    if args.verification:
        command.extend(
            [
                "--pr-enable-verification",
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
            ]
        )
    print(f"Starting PR sweep n_seq={n_seq}; log: {log_path}", flush=True)
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
        tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        )
        raise RuntimeError(
            f"PR sweep n_seq={n_seq} failed with exit code {process.returncode}; "
            f"see {log_path}\n{tail}"
        )
    with csv_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    print(f"Completed PR sweep n_seq={n_seq}", flush=True)
    return rows


def validate(rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    expected_variants = {
        ("JIST", "last-frame"),
        ("JIST", "frame-argmax"),
        ("MixVPR", "last-frame"),
    }
    gt_counts = {int(row["ground_truth_loops"]) for row in rows}
    if len(gt_counts) != 1:
        raise RuntimeError(f"GT denominator changed: {sorted(gt_counts)}")
    for n_seq in args.n_seq:
        subset = [row for row in rows if int(row["n_seq"]) == n_seq]
        variants = {(row["method"], row["evaluation_mode"]) for row in subset}
        expected_for_group = (
            {("MixVPR", "every-frame")} if n_seq == 0 else expected_variants
        )
        expected_stage = "verified" if args.verification else "off"
        if variants != expected_for_group or any(
            row["verification_stage"] != expected_stage for row in subset
        ):
            raise RuntimeError(f"Unexpected PR variants for n_seq={n_seq}")
        for variant in expected_for_group:
            curve = [
                row
                for row in subset
                if (row["method"], row["evaluation_mode"]) == variant
            ]
            if len(curve) != args.threshold_count:
                raise RuntimeError(f"Wrong threshold count for {variant}, n_seq={n_seq}")


def write_consolidated(rows: list[dict[str, str]], path: Path) -> None:
    rows = sorted(
        rows,
        key=lambda row: (
            int(row["n_seq"]),
            row["method"],
            row["evaluation_mode"],
            -float(row["threshold"]),
        ),
    )
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def consolidate_gt(args: argparse.Namespace, output_dir: Path) -> Path:
    paths = [
        output_dir / f"ground_truth_loops_nseq_{n_seq}.csv" for n_seq in args.n_seq
    ]
    reference = paths[0].read_bytes()
    for path in paths[1:]:
        if path.read_bytes() != reference:
            raise RuntimeError(f"GT extraction differs between {paths[0]} and {path}")
    destination = output_dir / "ground_truth_loops.csv"
    shutil.copyfile(paths[0], destination)
    return destination


def plot(rows: list[dict[str, str]], args: argparse.Namespace, output_dir: Path) -> tuple[Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = (
        ("JIST", "last-frame", "JIST", "#2563eb", "o"),
        ("JIST", "frame-argmax", "JIST refined (argmax)", "#7c3aed", "D"),
        ("MixVPR", "last-frame", "MixVPR", "#dc2626", "s"),
    )
    group_sizes = sorted({int(row["n_seq"]) for row in rows if int(row["n_seq"]) > 0})
    panel_sizes = group_sizes or [0]
    reference_curve = sorted(
        (row for row in rows if int(row["n_seq"]) == 0),
        key=lambda row: float(row["threshold"]),
        reverse=True,
    )
    columns = 2
    panel_count = len(panel_sizes)
    panel_rows = math.ceil(panel_count / columns)
    fig, axes = plt.subplots(panel_rows, columns, figsize=(12.5, 5.2 * panel_rows), squeeze=False)
    for axis, n_seq in zip(axes.flat, panel_sizes, strict=False):
        for method, mode, label, color, marker in variants if n_seq > 0 else ():
            curve = sorted(
                (
                    row
                    for row in rows
                    if int(row["n_seq"]) == n_seq
                    and row["method"] == method
                    and row["evaluation_mode"] == mode
                ),
                key=lambda row: float(row["threshold"]),
                reverse=True,
            )
            points = [
                (100.0 * float(row["recall"]), 100.0 * float(row["precision"]))
                for row in curve
                if math.isfinite(float(row["precision"]))
            ]
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=color,
                marker=marker,
                markevery=max(1, len(points) // 8),
                markersize=5,
                linewidth=2,
                label=label,
            )
        if reference_curve:
            reference_points = [
                (100.0 * float(row["recall"]), 100.0 * float(row["precision"]))
                for row in reference_curve
                if math.isfinite(float(row["precision"]))
            ]
            axis.plot(
                [point[0] for point in reference_points],
                [point[1] for point in reference_points],
                color="#111827",
                linestyle="--",
                linewidth=2.2,
                label="MixVPR every sampled frame (reference)",
            )
        axis.set_title(
            f"$n_{{seq}}={n_seq}$" if n_seq > 0 else "Framewise MixVPR reference ($n_{seq}=0$)"
        )
        axis.set_xlabel("Temporal-loop recall (%)")
        axis.set_ylabel("Detection precision (%)")
        axis.set_xlim(0, 105)
        axis.set_ylim(0, 105)
        axis.grid(True, alpha=0.3)
        axis.legend()
    for axis in list(axes.flat)[panel_count:]:
        axis.set_visible(False)

    verification_title = (
        "XFeat + LightGlue + RANSAC verified" if args.verification else "verification-off"
    )
    fig.suptitle(
        f"GRACO ground 01-06: {verification_title} precision–recall curves\n"
        f"fixed 195 trajectory-derived GT loops, n_skip={args.n_skip}, "
        f"GT={args.positive_distance:g} m/{args.positive_yaw_degrees:g} deg yaw"
    )
    fig.text(
        0.5,
        0.015,
        f"JIST thresholds {args.jist_threshold_min:g}–{args.jist_threshold_max:g}; "
        f"MixVPR thresholds {args.mixvpr_threshold_min:g}–{args.mixvpr_threshold_max:g}; "
        + (
            f"verification top-k={args.xfeat_top_k}, matches>={args.verification_min_matches}, "
            f"inliers>={args.ransac_min_inliers}, ratio>={args.ransac_min_inlier_ratio:g}, "
            f"threshold={args.ransac_reprojection_threshold:g} px."
            if args.verification
            else "XFeat/LightGlue/RANSAC disabled."
        ),
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.88, hspace=0.3, wspace=0.23)
    filename_stem = "pr_curves_verified" if args.verification else "pr_curves_verification_off"
    png_path = output_dir / f"{filename_stem}.png"
    pdf_path = output_dir / f"{filename_stem}.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    if args.show:
        plt.show()
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    args = parse_args()
    default_output = Path(
        "output/jist_mixvpr_pr_verified" if args.verification else "output/jist_mixvpr_pr"
    )
    output_dir = project_path(args.output_dir or default_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    consolidated_path = output_dir / "pr_sweep_results.csv"
    if args.plot_only:
        with consolidated_path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
    else:
        executable = project_path(args.executable)
        if not executable.is_file():
            raise RuntimeError(f"Benchmark executable does not exist: {executable}")
        rows: list[dict[str, str]] = []
        workers = min(args.jobs, len(args.n_seq))
        print(
            f"Launching {len(args.n_seq)} "
            f"{'verified' if args.verification else 'verification-off'} PR sweeps with "
            f"{workers} concurrent processes",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_sweep, args, n_seq, output_dir): n_seq
                for n_seq in args.n_seq
            }
            for future in as_completed(futures):
                rows.extend(future.result())
        print(f"Verified identical GT extraction: {consolidate_gt(args, output_dir)}")
        validate(rows, args)
        if args.append and consolidated_path.is_file():
            with consolidated_path.open(newline="", encoding="utf-8") as source:
                existing_rows = list(csv.DictReader(source))
            replaced_n_seq = set(args.n_seq)
            rows = [
                row for row in existing_rows if int(row["n_seq"]) not in replaced_n_seq
            ] + rows
        write_consolidated(rows, consolidated_path)
    validate(rows, args)
    png_path, pdf_path = plot(rows, args, output_dir)
    print(f"PR CSV:  {consolidated_path}")
    print(f"Plot PNG: {png_path}")
    print(f"Plot PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
