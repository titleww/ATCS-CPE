# -*- coding: utf-8 -*-
# Problem 09: CHUNK_SIZE / OVERLAP / EVAL_K_VALUES on astronomy data
from data_loader import load_qa

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
EVAL_K_VALUES = [1, 3, 5, 10]


def ranges(total):
    result = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    start = 0
    while start < total:
        end = min(start + CHUNK_SIZE, total)
        result.append((start, end))
        start += step
    return result


def run():
    data = load_qa()
    full_text = " ".join(d["text"] for d in data)
    total_chars = len(full_text)

    r = ranges(total_chars)
    print(f"Document total length: {total_chars} characters ({len(data)} Q&A entries)")
    print(f"Chunks generated (CHUNK_SIZE={CHUNK_SIZE}, OVERLAP={CHUNK_OVERLAP}):", len(r))
    print("Sample Chunk Boundaries:")
    for x in r[:4]:
        print(f"  Range: {x[0]} to {x[1]} (length = {x[1] - x[0]})")

    print(f"\nBoundary Overlap between Chunk 1 & 2 = {r[0][1] - r[1][0]} characters")

    print("\nRetrieval Evaluation Plan:")
    print(f"Golden Set Benchmark: {len(data)} test questions with Ground Truth chunks")
    for k in EVAL_K_VALUES:
        print(f"  Hit@{k:2}: checks if ground truth chunk appears in top {k}")

    print("\nExample Interpretation:")
    print("  Target chunk found at Rank 8 -> Hit@5 = 0, Hit@10 = 1")
    print("  Diagnostic: Recall is adequate, but Ranking requires Re-ranking optimization.")


if __name__ == "__main__":
    run()
