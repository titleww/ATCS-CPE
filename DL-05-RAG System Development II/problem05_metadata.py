# -*- coding: utf-8 -*-
# Problem 05: Similarity Search returns a document with right content but wrong Metadata
# Uses real data from astronomy_knowledge_base.txt
from data_loader import load_qa

QUERY = "กล้องโทรทรรศน์อวกาศ ส่องดู หลุมดำ"


def score(query, text):
    return sum(word in text for word in query.split())


def search(data, query, category=None):
    docs = data if category is None else [d for d in data if d["category"] == category]
    return max(docs, key=lambda d: score(query, d["question"] + " " + d["answer"]))


def run():
    data = load_qa()

    target_cat = "การสำรวจอวกาศและกล้องโทรทรรศน์"
    unfiltered = search(data, QUERY)
    filtered = search(data, QUERY, category=target_cat)

    print("Query:", QUERY)
    print("\nWithout Metadata Filtering -> picks document purely by keyword match count:")
    print(f"  [{unfiltered['category']}] {unfiltered['question']}")

    print(f"\nWith Metadata Filtering (category=\x27{target_cat}\x27):")
    print(f"  [{filtered['category']}] {filtered['question']}")

    print("\nCause: Pure vector/keyword similarity picks documents with matching words,")
    print("but without metadata filtering, it cannot guarantee matching target categories or document versions.")


if __name__ == "__main__":
    run()
