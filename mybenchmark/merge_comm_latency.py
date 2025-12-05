#!/usr/bin/env python3
import os
import csv
import argparse
from collections import defaultdict


REQUIRED_COMM_COLS = {"Algorithm", "Batch_size", "Sequence_length", "TP_Size", "CudaDevice", "MPI_Rank", "Communication"}
REQUIRED_LAT_COLS = {"Algorithm", "Batch_size", "Sequence_length", "TP_Size", "CudaDevice", "Latency"}


def read_csv_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip() for h in header]
        idx = {h: i for i, h in enumerate(header)}
        for row in reader:
            rows.append((header, idx, row))
    return rows


def ensure_columns(header_set, required_set, label):
    missing = required_set - header_set
    if missing:
        raise RuntimeError(f"{label} CSV is missing required columns: {', '.join(sorted(missing))}")


def make_key(idx, row):
    return (
        row[idx["Algorithm"]].strip(),
        row[idx["Batch_size"]].strip(),
        row[idx["Sequence_length"]].strip(),
        row[idx["TP_Size"]].strip(),
        row[idx["CudaDevice"]].strip(),
    )


def load_comm(comm_file):
    comm_map = defaultdict(list)
    for header, idx, row in read_csv_rows(comm_file):
        ensure_columns(set(header), REQUIRED_COMM_COLS, "Communication")
        key = make_key(idx, row)
        comm_map[key].append({
            "Algorithm": row[idx["Algorithm"]].strip(),
            "Batch_size": row[idx["Batch_size"]].strip(),
            "Sequence_length": row[idx["Sequence_length"]].strip(),
            "TP_Size": row[idx["TP_Size"]].strip(),
            "CudaDevice": row[idx["CudaDevice"]].strip(),
            "MPI_Rank": row[idx["MPI_Rank"]].strip(),
            "Communication": row[idx["Communication"]].strip(),
        })
    return comm_map


def load_latency(lat_file):
    lat_map = defaultdict(list)
    for header, idx, row in read_csv_rows(lat_file):
        ensure_columns(set(header), REQUIRED_LAT_COLS, "Latency")
        key = make_key(idx, row)
        lat_map[key].append({
            "Algorithm": row[idx["Algorithm"]].strip(),
            "Batch_size": row[idx["Batch_size"]].strip(),
            "Sequence_length": row[idx["Sequence_length"]].strip(),
            "TP_Size": row[idx["TP_Size"]].strip(),
            "CudaDevice": row[idx["CudaDevice"]].strip(),
            "Latency": row[idx["Latency"]].strip(),
        })
    return lat_map


def write_merged(out_path, merged_rows):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Batch_size", "Sequence_length", "TP_Size", "CudaDevice", "MPI_Rank", "Latency", "Communication"])
        writer.writerows(merged_rows)


def merge_files(comm_file, lat_file, output_dir):
    comm_map = load_comm(comm_file)
    lat_map = load_latency(lat_file)

    merged_rows = []
    for key in comm_map.keys() & lat_map.keys():
        comm_rows = comm_map[key]
        lat_rows = lat_map[key]
        for c in comm_rows:
            for l in lat_rows:
                merged_rows.append([
                    c["Algorithm"], c["Batch_size"], c["Sequence_length"], c["TP_Size"], c["CudaDevice"],
                    c["MPI_Rank"], l["Latency"], c["Communication"],
                ])

    algo, bs, sl, tp, dev = (comm_map or lat_map) and next(iter((comm_map or lat_map).keys())) or ("algo","bs","sl","tp","dev")
    out_name = f"merge_bs{bs}_sl{sl}_tp{tp}_algo{algo}.csv"
    out_path = os.path.join(output_dir, out_name)
    write_merged(out_path, merged_rows)
    return out_path, len(merged_rows)


def main():
    parser = argparse.ArgumentParser(description="Merge communication CSV with latency CSV by key columns")
    parser.add_argument("--comm-file", required=True, help="Path to communication CSV file")
    parser.add_argument("--latency-file", required=True, help="Path to latency CSV file")
    parser.add_argument("--output-dir", default=os.path.join(os.getcwd(), "merged"), help="Directory to write merged CSV")
    args = parser.parse_args()

    out_path, nrows = merge_files(args.comm_file, args.latency_file, args.output_dir)
    print(f"Merged {nrows} rows -> {out_path}")


if __name__ == "__main__":
    main()

