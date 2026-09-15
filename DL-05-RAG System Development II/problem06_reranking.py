# -*- coding: utf-8 -*-
# Problem 06: Relevant document ranked near bottom by First-stage Retrieval
# Uses real astronomy entries from astronomy_knowledge_base.txt
from data_loader import load_qa

GENERIC_TERMS = ["กล้องโทรทรรศน์", "อวกาศ", "ส่อง"]
SPECIFIC_TERMS = ["เจมส์เวบบ์", "JWST", "อินฟราเรด"]


def first_stage(doc):
    text = doc["question"] + doc["answer"]
    return sum(t in text for t in GENERIC_TERMS)


def rerank(doc):
    text = doc["question"] + doc["answer"]
    score = first_stage(doc)
    score += sum(4 for t in SPECIFIC_TERMS if t in text)
    return score


def run():
    data = load_qa()

    first = sorted(data, key=first_stage, reverse=True)[:6]
    second = sorted(first, key=rerank, reverse=True)

    print("Before Re-ranking (Top 6 from First-stage: based on generic terms \x27กล้องโทรทรรศน์\x27, \x27อวกาศ\x27):")
    for d in first:
        print(f"  score={first_stage(d)} | [{d['category']}] {d['question']}")

    print("\nAfter Re-ranking (Cross-Encoder style weight on specific terms \x27เจมส์เวบบ์\x27, \x27JWST\x27):")
    for d in second:
        print(f"  score={rerank(d)} | [{d['category']}] {d['question']}")

    print("\nCause: First-stage retrieval treats all term matches uniformly.")
    print("Cross-Encoder Re-ranking performs deep attention between query and candidate chunks, boosting target chunks to top.")


if __name__ == "__main__":
    run()
