import argparse
import json
import sys
from textwrap import shorten

import chromadb
import llm
import numpy as np


CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "products"
DEFAULT_TOP_K = 3
MAX_VECTOR_PRINT = 1024  # safety cap in case future dimensions grow


def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load collection '{COLLECTION_NAME}' from {CHROMA_DB_PATH}"
        ) from exc
    return collection


def determine_embedding_dim(collection) -> int:
    peek = collection.peek(limit=1)
    sample_ids = peek.get("ids", [])
    if not sample_ids:
        raise RuntimeError("Collection is empty; cannot determine embedding dimension.")
    sample = collection.get(ids=[sample_ids[0]], include=["embeddings"])
    embeddings = sample.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        raise RuntimeError("No embeddings returned from sample document.")
    return len(embeddings[0])


def build_query_vector(model, collection_dim: int, query_text: str) -> np.ndarray:
    """
    Build a fused query vector by repeating the CLIP text embedding
    until it matches the stored dimension (self-concatenation).
    """
    text_vec = np.array(model.embed(query_text), dtype=np.float32)
    text_dim = len(text_vec)

    if text_dim == 0:
        raise RuntimeError("Embedding model returned an empty vector.")

    if collection_dim <= text_dim:
        fused = text_vec[:collection_dim]
    else:
        repeats = int(np.ceil(collection_dim / text_dim))
        fused = np.tile(text_vec, repeats)[:collection_dim]

    norm = np.linalg.norm(fused)
    if norm:
        fused = fused / norm
    return fused


def parse_metadata(raw_meta: dict) -> dict:
    """Convert stringified metadata payloads into dictionaries."""
    if "metadata" in raw_meta and isinstance(raw_meta["metadata"], str):
        try:
            return json.loads(raw_meta["metadata"])
        except json.JSONDecodeError:
            pass
    return raw_meta


def run_query(collection, clip_model, collection_dim: int, query_text: str, top_k: int):
    query_vector = build_query_vector(clip_model, collection_dim, query_text)
    print("\n=== Query ===")
    print(f"Text: {query_text}")
    print(f"Vector dim: {len(query_vector)}")
    pretty_vector = np.array2string(
        query_vector[:MAX_VECTOR_PRINT], precision=6, separator=", "
    )
    if len(query_vector) > MAX_VECTOR_PRINT:
        pretty_vector = pretty_vector.rstrip("]") + ", ...]"
    print(f"Vector: {pretty_vector}")

    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_k,
        include=["metadatas", "distances", "documents"],
    )

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not ids:
        print("No results returned from ChromaDB.")
        return

    print("\n=== Top Matches ===")
    for rank, (doc_id, distance, raw_meta) in enumerate(
        zip(ids, distances, metadatas), start=1
    ):
        metadata = parse_metadata(raw_meta or {})
        title = metadata.get("product_title", "<unknown>")
        price = metadata.get("price")
        caption = shorten(metadata.get("sitemap_caption", ""), width=140, placeholder="...")

        print(f"\n#{rank} — ID: {doc_id}")
        print(f"  Title: {title}")
        if price is not None:
            print(f"  Price: {price}")
        print(f"  Distance (cosine): {distance:.4f}")
        if caption:
            print(f"  Caption: {caption}")
        local_image = metadata.get("local_image_path")
        if local_image:
            print(f"  Image: {local_image}")
        if metadata:
            metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
            print("  Metadata:")
            for line in metadata_json.splitlines():
                print(f"    {line}")


def interactive_loop(collection, clip_model, collection_dim: int, top_k: int):
    print(
        "\nEnter your query (press Enter on an empty line to exit). "
        "Example: 'royal blue sharara suit'"
    )
    while True:
        try:
            query_text = input("\nQuery> ").strip()
        except EOFError:
            print("\nEOF received. Exiting.")
            break
        if not query_text:
            print("Goodbye!")
            break
        run_query(collection, clip_model, collection_dim, query_text, top_k)


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal product search demonstration."
    )
    parser.add_argument(
        "--query",
        help="Run a single query in non-interactive mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of nearest neighbors to show (default: {DEFAULT_TOP_K}).",
    )
    args = parser.parse_args()

    try:
        clip_model = llm.get_embedding_model("clip")
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize CLIP via llm. "
            "Ensure 'llm install llm-clip' has been executed in this environment."
        ) from exc

    collection = load_collection()
    collection_dim = determine_embedding_dim(collection)

    if args.query:
        run_query(collection, clip_model, collection_dim, args.query, args.top_k)
    else:
        interactive_loop(collection, clip_model, collection_dim, args.top_k)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

