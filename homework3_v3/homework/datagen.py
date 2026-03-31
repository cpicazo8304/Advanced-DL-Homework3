def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.7):
    import json
    from .cot import CoTModel
    from .data import Dataset

    llm = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    data = list(Dataset("train"))
    results = []
    question_counts = {}
    prompts = [llm.format_prompt(q) for q, _ in data]

    while len(results) < 1000:
        all_completions = llm.batched_generate(prompts, num_return_sequences=oversample, temperature=temperature)

        for (question, answer), completions in zip(data, all_completions):
            if len(results) >= 1000:
                break
            true = float(answer)
            for completion in completions:
                parsed = llm.parse_answer(completion)
                correct = abs(parsed) < 1e-2 if true == 0 else abs(parsed - true) / abs(true) <= 0.1
                if correct and question_counts.get(question, 0) < 3:
                    results.append([question, true, completion])
                    question_counts[question] = question_counts.get(question, 0) + 1
                    break

        print(f"Generated {len(results)} examples so far")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
