from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
      messages = [
          {
              "role": "system",
              "content": (
                  "You are a precise unit conversion assistant.\n"
                  "Solve step-by-step but keep reasoning minimal.\n"
                  "ALWAYS output the final answer inside <answer></answer> tags.\n"
                  "The answer must be a single number with no units."
              ),
          },
          {
              "role": "user",
              "content": "Convert 10 meters to yards.",
          },
          {
              "role": "assistant",
              "content": (
                  "1 meter = 1.09361 yards.\n"
                  "10 × 1.09361 = 10.9361\n"
                  "<answer>10.9361</answer>"
              ),
          },
          {
              "role": "user",
              "content": question,
          },
      ]

      return self.tokenizer.apply_chat_template(
          messages,
          add_generation_prompt=True,
          tokenize=False,
      )

        


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
