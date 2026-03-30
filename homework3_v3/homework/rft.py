from .base_llm import BaseLLM
from .sft import test_model, TokenizedDataset
from .datagen import generate_dataset

def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel
    from .cot import CoTModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    cot = CoTModel()
    llm.format_prompt = lambda q: f"question: {q}"

    llm.model.eval()

    return llm

def format_example_rft(prompt: str, answer: float, reasoning: str) -> dict[str, str]:
    return {
        "question": f"question: {prompt}",
        "answer": reasoning
    }


def train_model(
    output_dir: str,
    **kwargs,
):
    import json
    from peft import get_peft_model, LoraConfig
    from transformers import TrainingArguments, Trainer
    from pathlib import Path
    # create LoRA model
    llm = BaseLLM()

    config = LoraConfig(
        task_type="CAUSAL_LM",
        bias="none",
        target_modules="all-linear",
        r=16,
        lora_alpha=64,
    )

    peft_model = get_peft_model(llm.model, config)
    peft_model.enable_input_require_grads()

    training_args = TrainingArguments(
        gradient_checkpointing=True,
        learning_rate=2e-4,
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=5,
        per_device_train_batch_size=8,  # reduce from 32
        gradient_accumulation_steps=4,  # simulate batch size of 32
        fp16=True,                       # use half precision
    )
    # create dataset
    rft_data = json.load(open(Path(__file__).parent.parent / "data" / "rft.json"))
    dataset = TokenizedDataset(llm.tokenizer, rft_data, format_example_rft)

    # initialize trainer with args, dataset, and 
    trainer = Trainer(
        peft_model,
        training_args,
        train_dataset=dataset
    )

    # train model
    trainer.train()
    trainer.save_model("homework/rft_model")

    test_model("homework/rft_model")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
