#!/usr/bin/env python3
"""
Real use case: pass a vector, run vector search, print similarity scores.

Usage:
  python3 test_similarity.py                    # use default query vector
  python3 test_similarity.py 1,0,0,0,0        # query with comma-separated numbers
  python3 test_similarity.py --seed-only       # only seed sample data, no search

Vectors must be 5 dimensions (match your index). Example: 0.9,0.1,0,0,0
"""

import sys
from db import get_client, insert_frame, search

# Sample frames with different embeddings so similarities vary
SAMPLE_FRAMES = [
    ([1.0, 0.0, 0.0, 0.0, 0.0], "frame_A"),
    ([0.9, 0.1, 0.0, 0.0, 0.0], "frame_B"),
    ([0.7, 0.3, 0.0, 0.0, 0.0], "frame_C"),
    ([0.0, 0.0, 1.0, 0.0, 0.0], "frame_D"),
    ([0.5, 0.5, 0.0, 0.0, 0.0], "frame_E"),
]


def seed_sample_data():
    """Insert sample frames so we have something to search."""
    client = get_client()
    client.admin.command("ping")
    for emb, frame_id in SAMPLE_FRAMES:
        insert_frame(emb, frame_id=frame_id, source="test_similarity")
    print(f"Seeded {len(SAMPLE_FRAMES)} sample frames.\n")


def parse_vector(s: str):
    """Parse '0.9,0.1,0,0,0' -> [0.9, 0.1, 0.0, 0.0, 0.0]."""
    parts = s.strip().split(",")
    if len(parts) != 5:
        raise ValueError(f"Need exactly 5 numbers, got {len(parts)}. Example: 0.9,0.1,0,0,0")
    return [float(x.strip()) for x in parts]


def main():
    seed_only = "--seed-only" in sys.argv
    if seed_only:
        seed_sample_data()
        return

    # Default query: similar to frame_A and frame_B
    query_str = "1,0,0,0,0"
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        query_str = sys.argv[1]

    try:
        query = parse_vector(query_str)
    except ValueError as e:
        print("Error:", e)
        print("Usage: python3 test_similarity.py [v1,v2,v3,v4,v5]")
        print("Example: python3 test_similarity.py 0.9,0.1,0,0,0")
        sys.exit(1)

    # Seed if you want fresh sample data (optional; comment out to use existing data)
    # seed_sample_data()

    print("Query vector:", query)
    print("Running vector search (similarity = score, higher = more similar)...\n")

    results = search(query, top_k=5)

    if not results:
        print("No results. Seed sample data first: python3 test_similarity.py --seed-only")
        return

    print("Results (by similarity):")
    print("-" * 50)
    for i, doc in enumerate(results, 1):
        score = doc.get("score", 0)
        frame_id = doc.get("frame_id", "?")
        print(f"  {i}. similarity = {score:.4f}  frame_id = {frame_id}")
    print("-" * 50)
    print(f"Total: {len(results)} result(s)")


if __name__ == "__main__":
    main()
