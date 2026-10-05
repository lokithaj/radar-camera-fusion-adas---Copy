"""
Memory Benchmark Script for Radar-Camera Fusion ADAS (CPU Peak RAM / RSS Profiler)

Measures peak physical RAM (Resident Set Size / Working Set) of the inference process
using psutil, comparing baseline vs. camera-guided sparsification.
"""

import argparse
import csv
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
import psutil

DEFAULT_RECORDING = Path(r"C:\Users\lokit\Desktop\RADIal_data\RECORD@2020-11-21_13.44.44")


def stream_reader(pipe, output_lines):
    """Continuously read lines from child process pipe to prevent pipe buffer deadlock."""
    try:
        for line in iter(pipe.readline, ""):
            output_lines.append(line)
            # Optionally print progress
            if "Processing sample" in line:
                print(f"    {line.strip()}", flush=True)
    except Exception:
        pass
    finally:
        pipe.close()


def profile_process_memory(cmd, cwd=None, sample_interval_s=0.05):
    """
    Spawns cmd as a child process and tracks peak physical RSS RAM using psutil.

    Returns
    -------
    peak_rss_bytes : int
    duration_s : float
    returncode : int
    output_lines : list of str
    """
    output_lines = []

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    reader_thread = threading.Thread(
        target=stream_reader,
        args=(proc.stdout, output_lines),
        daemon=True,
    )
    reader_thread.start()

    peak_rss = 0
    start_time = time.perf_counter()

    try:
        parent_process = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                current_rss = parent_process.memory_info().rss
                # Add RSS of any child processes spawned by inference
                for child in parent_process.children(recursive=True):
                    try:
                        current_rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                if current_rss > peak_rss:
                    peak_rss = current_rss

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

            time.sleep(sample_interval_s)

        # Final check if still running
        try:
            if parent_process.is_running():
                current_rss = parent_process.memory_info().rss
                for child in parent_process.children(recursive=True):
                    try:
                        current_rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                if current_rss > peak_rss:
                    peak_rss = current_rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    proc.wait()
    duration_s = time.perf_counter() - start_time
    reader_thread.join(timeout=3.0)

    return peak_rss, duration_s, proc.returncode, output_lines


def parse_frames_processed(output_lines):
    """Extract sample indices and frame count from inference stdout."""
    sample_indices = []
    sample_regex = re.compile(r"Processing sample\s+(\d+)\.\.\.")
    for line in output_lines:
        match = sample_regex.search(line)
        if match:
            sample_indices.append(int(match.group(1)))
    return len(sample_indices), sample_indices


def benchmark_run(
    recording_path: Path,
    mode: str,
    retention_ratio: float,
    project_root: Path,
    python_exe: str = sys.executable,
):
    """Run a single benchmark configuration and print its results."""
    cmd = [
        python_exe,
        "-m",
        "fusion.inference",
        "--recording",
        str(recording_path),
        "--sparsification-mode",
        mode,
        "--retention-ratio",
        str(retention_ratio),
    ]

    print(f"\n========================================================")
    print(f"Running Benchmark: Mode = '{mode}' | Retention Ratio = {retention_ratio}")
    print(f"Command: {' '.join(cmd)}")
    print(f"========================================================")

    peak_rss_bytes, duration_s, returncode, output_lines = profile_process_memory(
        cmd=cmd,
        cwd=str(project_root),
        sample_interval_s=0.05,
    )

    peak_ram_mb = peak_rss_bytes / (1024.0 * 1024.0)
    peak_ram_gb = peak_rss_bytes / (1024.0 * 1024.0 * 1024.0)
    num_frames, sample_indices = parse_frames_processed(output_lines)
    success = (returncode == 0)

    print(f"\n--- Run Summary ---")
    print(f"Configuration/Mode:       {mode}")
    print(f"Retention Ratio:          {retention_ratio:.2f}")
    print(f"Completed Successfully:   {success} (exit code: {returncode})")
    print(f"Frames Processed:         {num_frames} {sample_indices if num_frames > 0 else ''}")
    print(f"Execution Duration:       {duration_s:.2f} s")
    print(f"Peak CPU RAM (RSS):       {peak_ram_mb:.2f} MB ({peak_ram_gb:.4f} GB)")

    if not success:
        print("\n[ERROR] Output from failed run:")
        print("".join(output_lines[-30:]))

    return {
        "mode": mode,
        "retention_ratio": retention_ratio,
        "peak_ram_mb": peak_ram_mb,
        "peak_ram_gb": peak_ram_gb,
        "peak_rss_bytes": peak_rss_bytes,
        "duration_s": duration_s,
        "success": success,
        "returncode": returncode,
        "num_frames": num_frames,
        "sample_indices": sample_indices,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark peak CPU RAM (RSS) for Radar-Camera Fusion ADAS."
    )
    parser.add_argument(
        "--recording",
        type=Path,
        default=DEFAULT_RECORDING,
        help=f"Path to RADIal recording (default: {DEFAULT_RECORDING})",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    recording_path = args.recording.resolve()

    if not recording_path.exists():
        print(f"[ERROR] Recording directory does not exist: {recording_path}")
        sys.exit(1)

    print("================================================================")
    print(" RADAR-CAMERA FUSION ADAS - CPU MEMORY (RAM) BENCHMARK")
    print("================================================================")
    print(f"Environment Info:")
    print(f"  Python executable:    {sys.executable}")
    print(f"  psutil version:       {psutil.__version__}")
    print(f"  Recording path:       {recording_path}")
    print(f"  Project root:         {project_root}")
    print(f"  Compute target:       CPU-only (No CUDA/GPU allocated)")
    print("================================================================")

    # 1. Benchmark baseline (none, 1.0)
    baseline_result = benchmark_run(
        recording_path=recording_path,
        mode="none",
        retention_ratio=1.0,
        project_root=project_root,
    )

    # Cooldown pause
    time.sleep(2.0)

    # 2. Benchmark camera-guided (camera, 0.50)
    camera_result = benchmark_run(
        recording_path=recording_path,
        mode="camera",
        retention_ratio=0.50,
        project_root=project_root,
    )

    # Comparative analysis
    b_ram = baseline_result["peak_ram_mb"]
    c_ram = camera_result["peak_ram_mb"]
    diff_mb = c_ram - b_ram
    pct_change = (diff_mb / b_ram * 100.0) if b_ram > 0 else 0.0

    print("\n" + "=" * 64)
    print(" COMPARATIVE MEMORY SUMMARY")
    print("=" * 64)
    print(f"Baseline (mode=none, ratio=1.00):")
    print(f"  Peak RAM = {b_ram:.2f} MB ({baseline_result['peak_ram_gb']:.4f} GB)")
    print(f"  Success  = {baseline_result['success']}")
    print(f"  Frames   = {baseline_result['num_frames']}")
    print()
    print(f"Camera-guided 50% (mode=camera, ratio=0.50):")
    print(f"  Peak RAM = {c_ram:.2f} MB ({camera_result['peak_ram_gb']:.4f} GB)")
    print(f"  Success  = {camera_result['success']}")
    print(f"  Frames   = {camera_result['num_frames']}")
    print()
    print(f"Memory difference = {diff_mb:+.2f} MB")
    print(f"Memory change     = {pct_change:+.2f} %")
    print("=" * 64)

    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    mem_csv_path = output_dir / "memory_benchmark.csv"
    with open(mem_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "retention_ratio", "frames_processed", "peak_cpu_ram_mb", "peak_cpu_ram_gb"])
        writer.writerow([
            baseline_result["mode"],
            f"{baseline_result['retention_ratio']:.2f}",
            baseline_result["num_frames"],
            f"{baseline_result['peak_ram_mb']:.2f}",
            f"{baseline_result['peak_ram_gb']:.4f}",
        ])
        writer.writerow([
            camera_result["mode"],
            f"{camera_result['retention_ratio']:.2f}",
            camera_result["num_frames"],
            f"{camera_result['peak_ram_mb']:.2f}",
            f"{camera_result['peak_ram_gb']:.4f}",
        ])
    print(f"Saved memory benchmark results to: {mem_csv_path}")

    # Detailed engineering rationale for presentation
    print("\n--- Presentation & Slide Recommendation ---")
    if abs(pct_change) < 5.0:
        print(
            "Verdict: Memory consumption is ESSENTIALLY UNCHANGED between baseline\n"
            "and camera-guided sparsification (< 5% variance, within OS working set noise).\n"
            "\n"
            "Why this is expected:\n"
            "1. Process memory footprint in this CPU environment is dominated by:\n"
            "   - PyTorch runtime and Ultralytics YOLO11n model weights (~2.6M parameters)\n"
            "   - OpenCV display and image buffers (1920x1080 camera frames in memory)\n"
            "   - RADIal DBReader dataset sensor buffers (FMCW raw ADC / 16-RX chirp cubes)\n"
            "2. Adaptive/Camera-guided sparsification operates on lightweight metadata lists\n"
            "   (filtering ~2-15 radar object dictionary entries before ML association).\n"
            "   Pruning a handful of 3D point dictionaries saves only negligible bytes,\n"
            "   so it alters algorithmic complexity and latency (association runtime), NOT\n"
            "   the deep neural network architecture or overall process RAM allocation."
        )
    else:
        print(
            f"Observed memory difference: {diff_mb:+.2f} MB ({pct_change:+.2f}%)."
        )
    print("================================================================\n")


if __name__ == "__main__":
    main()
