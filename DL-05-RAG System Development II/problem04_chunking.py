# -*- coding: utf-8 -*-
# Problem 04: Chunk too large / too small / Overlap
# Uses real content from astronomy_knowledge_base.txt
from data_loader import load_qa

CATEGORY = "ระบบสุริยะและดาวเคราะห์"


def chunk(words, size, overlap=0):
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]


def run():
    data = load_qa()
    doc = " ".join(d["answer"] for d in data if d["category"] == CATEGORY)
    words = doc.split()

    print(f"Sample document: combined answers from category \x27{CATEGORY}\x27 ({len(words)} words)")

    print("\nLarge chunk (size=150):")
    for c in chunk(words, 150)[:2]:
        print("-", c[:130] + "...")

    print("\nSmall chunk (size=15):")
    for c in chunk(words, 15)[:4]:
        print("-", c)

    print("\nChunk + Overlap (size=50, overlap=15):")
    for c in chunk(words, 50, 15)[:3]:
        print("-", c[:100] + "...")

    print("\nCause:")
    print("- Too large: mixes multiple planetary facts, diluting vector representations.")
    print("- Too small: cuts critical explanations in half, losing context.")
    print("- Overlap: bridges the boundary between consecutive chunks to prevent information loss.")


if __name__ == "__main__":
    run()
