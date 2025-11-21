import argparse
import contextlib
from pathlib import Path

import llm

from main import (
    DEFAULT_TOP_K,
    determine_embedding_dim,
    load_collection,
    run_query,
)


DEMOS = [
    ("Query 1 – Color & Style", "royal blue sharara suit"),
    ("Query 2 – Fabric / Occasion", "formal wear for a mehndi"),
    ("Query 3 – Cross-modal imagery", "dress that looks like a flower garden"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the three demo queries and save the output to a text file."
    )
    parser.add_argument(
        "--output",
        default="demo_queries_output.txt",
        help="Destination path for the saved query transcript.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)

    clip_model = llm.get_embedding_model("clip")
    collection = load_collection()
    dim = determine_embedding_dim(collection)

    with open(output_path, "w", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(handle):
            for title, text in DEMOS:
                print(f"\n================ {title} ================")
                run_query(collection, clip_model, dim, text, DEFAULT_TOP_K)

    print(f"Demo query output saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()

