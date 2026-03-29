def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    import json
    from .cot import CoTModel
    from .data import Dataset

    # get dataset and llm
    llm = CoTModel()
    data = Dataset("train")
    results = []

    # go through each example in training dataset to create the same example but with
    # reasoning
    for question, answer in data:
      prompt = [llm.format_prompt(question)]
      completions = llm.batched_generate(prompt,num_return_sequences=oversample, temperature=temperature)

      # go through each of the completions to see if the answer is correct (only need one)
      for completion in completions:
        if llm.parse_answer(completion) == float(answer):
          results.append([question, float(answer), completion])
          break

    # store the output
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
