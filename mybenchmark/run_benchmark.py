import os
import sys
import subprocess
from pathlib import Path

# SEQ_LABELS = ["128", "256", "512", "1k", "2k", "4k", "8k", "16k", "32k", "64k"]
SEQ_LABELS = ["128", "256", "512", "1k", "2k", "4k", "8k", "16k", "32k", "64k"]
BATCH_SIZES = [1]
SEQ_MAP = {
    "128": 128,
    "256": 256,
    "512": 512,
    "1k": 1024,
    "2k": 2048,
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
    "64k": 65536,
}


BASE = Path("/root/autodl-tmp/TensorRT-LLM/mybenchmark")
DATA_DIR = BASE / "prepare_data_json"
RESULTS_DIR = BASE / "results"
FIGURES_DIR = BASE / "figures"
PREPARE_DATASET = Path("/root/autodl-tmp/TensorRT-LLM/benchmarks/cpp/prepare_dataset.py")
BENCH_BIN = Path("/root/autodl-tmp/TensorRT-LLM/cpp/build/benchmarks/gptManagerBenchmark")
TOKENIZER_DIR = Path("/root/autodl-tmp/tmp/Qwen/0.6B")
ENGINE_DIR = Path("/root/autodl-tmp/tmp/qwen/0.6B/trt_engines/fp16/1-gpu/")


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_pydeps():
    try:
        import pandas  # noqa: F401
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "pandas"], check=False)
    try:
        import matplotlib  # noqa: F401
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib"], check=False)


def generate_dataset(seq_label, batch_size):
    input_mean = SEQ_MAP[str(seq_label)]
    out_file = DATA_DIR / f"token_norm_dist_{seq_label}_{batch_size}.json"
    cmd = [
        sys.executable,
        str(PREPARE_DATASET),
        "--output",
        str(out_file),
        "--tokenizer",
        str(TOKENIZER_DIR),
        "token-norm-dist",
        "--num-requests",
        "100",
        "--input-mean",
        str(input_mean),
        "--input-stdev",
        "0",
        "--output-mean",
        "50",
        "--output-stdev",
        "0",
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def run_benchmark(seq_label, batch_size):
    dataset_path = DATA_DIR / f"token_norm_dist_{seq_label}_{batch_size}.json"
    out_csv = RESULTS_DIR / f"result_{batch_size}_{seq_label}.csv"
    cmd = [
        str(BENCH_BIN),
        "--engine_dir",
        str(ENGINE_DIR),
        "--request_rate",
        "-1",
        "--streaming",
        "--static_emulated_batch_size",
        str(batch_size),
        "--dataset",
        str(dataset_path),
        "--output_csv",
        str(out_csv),
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def merge_csv():
    import pandas as pd
    frames = []
    for p in RESULTS_DIR.glob("result_*.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")] if hasattr(df.columns, "str") else df
        name = p.stem
        try:
            parts = name.split("_")
            b = int(parts[1])
            s = "_".join(parts[2:])
        except Exception:
            continue
        df.insert(0, "batch_size", b)
        df.insert(1, "sequence_length", s)
        df.insert(2, "sequence_length_num", SEQ_MAP.get(s, None))
        frames.append(df)
    if not frames:
        return None
    all_df = pd.concat(frames, ignore_index=True)
    out = RESULTS_DIR / "all_results.csv"
    all_df.to_csv(out, index=False)
    return out


def plot_graphs(all_csv_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv(all_csv_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")] if hasattr(df.columns, "str") else df
    df = df.dropna(subset=["sequence_length_num"]) if "sequence_length_num" in df.columns else df
    if "batch_size" not in df.columns or "sequence_length" not in df.columns:
        return
    df = df.sort_values(["batch_size", "sequence_length_num"]) if "sequence_length_num" in df.columns else df
    for b in sorted(df["batch_size"].unique()):
        g = df[df["batch_size"] == b]
        x = g["sequence_length_num"].tolist()
        y_comm = (g["token_throughput(token/sec)"] * (g["total_latency(ms)"] / 1000.0)).tolist() if "token_throughput(token/sec)" in g.columns and "total_latency(ms)" in g.columns else []
        y_total = g["total_latency(ms)"].tolist() if "total_latency(ms)" in g.columns else []
        y_ttft = g["avg_time_to_first_token(ms)"].tolist() if "avg_time_to_first_token(ms)" in g.columns else []
        y_inter = g["avg_inter_token_latency(ms)"].tolist() if "avg_inter_token_latency(ms)" in g.columns else []
        if x and y_comm:
            plt.figure()
            print(y_comm)
            plt.plot(x, y_comm, marker="o")
            plt.xlabel("sequence_length")
            plt.ylabel("Total tokens (token_throughput × total_latency_s)")
            plt.grid(True)
            plt.tight_layout()
            plt.ylim(bottom=0)
            plt.savefig(FIGURES_DIR / f"batch{b}_communication.png")
            plt.close()
        if x and y_total:
            plt.figure()
            plt.plot(x, y_total, marker="o")
            plt.xlabel("sequence_length")
            plt.ylabel("total_latency(ms)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"batch{b}_total_latency.png")
            plt.close()
        if x and y_ttft:
            plt.figure()
            plt.plot(x, y_ttft, marker="o")
            plt.xlabel("sequence_length")
            plt.ylabel("avg_time_to_first_token(ms)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"batch{b}_ttft.png")
            plt.close()
        if x and y_inter:
            plt.figure()
            plt.plot(x, y_inter, marker="o")
            plt.xlabel("sequence_length")
            plt.ylabel("avg_inter_token_latency(ms)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"batch{b}_inter_token.png")
            plt.close()


def main():
    ensure_dirs()
    ensure_pydeps()
    for b in BATCH_SIZES:
        for s in SEQ_LABELS:
            ok_gen = generate_dataset(s, b)
            if not ok_gen:
                continue
            run_benchmark(s, b)
    merged = merge_csv()
    if merged:
        plot_graphs(merged)


if __name__ == "__main__":
    main()
