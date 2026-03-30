def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
  import json
  from .cot import CoTModel
  from .data import Dataset

  llm = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
  data = Dataset("train")
  results = []
  question_counts = {}

  while len(results) < 1000:
    for question, answer in data:
      if len(results) >= 1000:
        break
      if question_counts.get(question, 0) >= 3:
        continue
      prompt = [llm.format_prompt(question)]
      completions = llm.batched_generate(prompt, num_return_sequences=oversample, temperature=temperature)

      for completion in completions[0]:
        true = float(answer)
        parsed = llm.parse_answer(completion)
        if true == 0:
          correct = abs(parsed) < 1e-2
        else:
          correct = abs(parsed - true) / abs(true) <= 0.1

        if correct:
          results.append([question, float(answer), completion])
          question_counts[question] = question_counts.get(question, 0) + 1
          break  # one correct per batch_generate call, move to next question

  with open(output_json, "w") as f:
    json.dump(results, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
