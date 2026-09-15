# -*- coding: utf-8 -*-
# Problem 07: Retrieval is correct, but Generated Answer distorts critical numerical facts (Faithfulness)
# Uses real astronomy facts (Venus surface temperature)
from data_loader import load_qa


def find_entry(data):
    return next(d for d in data if "465 องศาเซลเซียส" in d["answer"])


def bad_generator(context):
    # Simulated hallucination / number distortion
    return context.replace("465 องศาเซลเซียส", "850 องศาเซลเซียส").replace("96%", "50%")


def grounded_generator(context):
    return context


def run():
    data = load_qa()
    entry = find_entry(data)
    context = entry["answer"]

    print("Question:", entry["question"])
    print("\nRetrieved Ground Truth Context:")
    print(context)

    print("\nBad Generation (distorts temperature from 465°C to 850°C):")
    print(bad_generator(context))

    print("\nGrounded Generation (strictly faithful to Context):")
    print(grounded_generator(context))

    print("\nCause: LLM may hallucinate or distort numerical figures when temperature is too high.")
    print("A strict grounding prompt and low temperature (e.g. 0.2) ensure faithfulness.")


if __name__ == "__main__":
    run()
