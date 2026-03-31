def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.7):
    import json
    from .cot import CoTModel
    from .data import Dataset

  llm = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    data = Dataset("train")
    results = []
    question_counts = {}
    batch_size = 8  # process multiple questions at once
    
    while len(results) < 1000:
        batch_questions = []
        batch_answers = []
        for question, answer in data:
            if len(results) + len(batch_questions) >= 1000:
                break
            if question_counts.get(question, 0) >= 3:
                continue
            batch_questions.append(question)
            batch_answers.append(answer)
            if len(batch_questions) == batch_size:
                break

        if not batch_questions:
            break

        prompts = [llm.format_prompt(q) for q in batch_questions]
        all_completions = llm.batched_generate(prompts, num_return_sequences=oversample, temperature=temperature)

        for question, answer, completions in zip(batch_questions, batch_answers, all_completions):
            true = float(answer)
            for completion in completions:
                parsed = llm.parse_answer(completion)
                correct = abs(parsed) < 1e-2 if true == 0 else abs(parsed - true) / abs(true) <= 0.1
                if correct:
                    results.append([question, float(answer), completion])
                    question_counts[question] = question_counts.get(question, 0) + 1
                    if len(results) % 100 == 0:
                      print(f"Generated {len(results)} examples so far")
                    break

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
