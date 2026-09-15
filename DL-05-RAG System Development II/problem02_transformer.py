# -*- coding: utf-8 -*-
# Problem 02: Vocabulary Mismatch and Token order (Position)
# Uses real astronomy questions from astronomy_knowledge_base.txt
from data_loader import load_qa


def bow(text):
    result = {}
    for token in text.split():
        result[token] = result.get(token, 0) + 1
    return result


def with_position(text):
    return [(i, token) for i, token in enumerate(text.split())]


def run():
    q_formal = "ดาวเคราะห์ดวงใดในระบบสุริยะที่มีขนาดใหญ่ที่สุด"
    q_casual = "ดาวดวงไหนใหญ่สุดในระบบสุริยะของเรา"

    print("Technical query:", q_formal)
    print("Casual query   :", q_casual)
    print("BoW (Technical):", bow(q_formal))
    print("BoW (Casual)   :", bow(q_casual))
    common = set(bow(q_formal)) & set(bow(q_casual))
    print("Exact-token overlap:", common or "None")
    print("-> Both queries ask about Jupiter, but BoW fails to capture synonymous expressions")
    print("   because wording differs (Vocabulary Mismatch).")

    print("\nExample: effect of Token order on meaning (Positional Encoding):")
    a = "ดาวพฤหัสบดี ดึงดูด ดาวหาง ด้วย แรงโน้มถ่วง มหาศาล"
    b = "ดาวหาง ดึงดูด ดาวพฤหัสบดี ด้วย แรงโน้มถ่วง มหาศาล"
    print("Sentence A (Original):", with_position(a))
    print("Sentence B (Reversed):", with_position(b))
    print("BoW identical:", bow(a) == bow(b))
    print("Meaning identical?: False (Jupiter attracts the comet vs Comet attracts Jupiter)")

    print("\nCause: Bag-of-Words lacks word order and synonym semantics.")
    print("Transformers use Positional Encoding + Self-Attention to preserve word order and semantic context.")


if __name__ == "__main__":
    run()
