# -*- coding: utf-8 -*-
# Problem 03: Duplicate / Noise / Broken Text / Normalization
# Simulates noise on real astronomy questions from astronomy_knowledge_base.txt
import re
from data_loader import load_qa


def make_noisy_samples(data, n=2):
    base = [d["question"] for d in data[:n]]
    raw = []
    for q in base:
        raw.append(q)                                        # original
        raw.append(q)                                        # exact duplicate
        raw.append("   " + q + "   ")                        # extra whitespace
        raw.append(q.replace(" ", "_") + "!!!")              # symbol noise
    raw.append("")                                           # empty row from broken extraction
    return raw


def normalize(text):
    text = text.lower()
    text = re.sub(r"[_@!\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def run():
    data = load_qa()
    raw = make_noisy_samples(data)

    print("Before cleaning (simulated noise on real astronomy questions):")
    for x in raw:
        print(repr(x))

    normalized = [normalize(x) for x in raw if x.strip()]
    unique = list(dict.fromkeys(normalized))

    print("\nAfter Normalization + Deduplication:")
    for x in unique:
        print(repr(x))

    print("\nCount before:", len(raw))
    print("Count after :", len(unique))

    all_q_normalized = [normalize(d["question"]) for d in data]
    dup_count = len(all_q_normalized) - len(set(all_q_normalized))
    print(f"\nChecked full astronomy_knowledge_base.txt ({len(data)} entries): found {dup_count} duplicate question(s)")
    print("Cause: Duplicates waste top-k slots, while noise prevents exact and semantic matches.")


if __name__ == "__main__":
    run()
