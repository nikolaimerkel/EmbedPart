import csv
import glob
import os
from typing import Optional


# where we have the input files
input_dir = "/data/sdb/nikolai/gnn-partitioner/gnn-partitioning/results/partitioning-metrics/cuttana"

# input files look like:
# "{graph}.xxx.P{partition_number}.tmp.csv"
#
# - first line: ignore
# - second line: run_time in seconds

graphs = ["ogbn-arxiv", "ogbn-products", "reddit", "ogbn-papers100M"]
partitions = [2, 4, 8, 16, 32]

output_csv = "/data/sdb/nikolai/gnn-partitioner/gnn-partitioning/results/partitioning-metrics/cuttana/cuttana_partitioning_runtime.csv"


def read_second_line_as_float(path: str) -> Optional[float]:
    """Return the 2nd line as float if possible, else None."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 2:
            return None
        value_str = lines[1].strip()
        if not value_str:
            return None
        return float(value_str)
    except (OSError, ValueError):
        return None


def find_file_for_graph_and_part(graph: str, p: int) -> Optional[str]:
    """Find the first file matching '{graph}.*.P{p}.tmp.csv' in input_dir."""
    pattern = os.path.join(input_dir, f"{graph}.*.P{p}.tmp.csv")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def main() -> None:
    rows = []
    missing = []
    invalid = []

    for graph in graphs:
        for p in partitions:
            path = find_file_for_graph_and_part(graph, p)
            if path is None:
                missing.append((graph, p))
                continue

            t = read_second_line_as_float(path)
            if t is None:
                invalid.append((graph, p, path))
                continue

            rows.append({
                "graph": graph,
                "cuttana": "cuttana",
                "num_parts": p,
                "partitioning_time": t,
            })

    # write results
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["graph", "cuttana", "num_parts", "partitioning_time"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")

    if missing:
        print("\nMissing files:")
        for graph, p in missing:
            print(f"  - {graph}, P{p}")

    if invalid:
        print("\nFiles with unreadable/invalid 2nd line:")
        for graph, p, path in invalid:
            print(f"  - {graph}, P{p}: {path}")


if __name__ == "__main__":
    main()