from .base_llm import BaseLLM
from .sft import test_model, format_example, TokenizedDataset
from .datagen import generate_dataset

def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


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
        r=32,
        lora_alpha=128,
    )

    peft_model = get_peft_model(llm.model, config)
    peft_model.enable_input_require_grads()

    training_args = TrainingArguments(
      gradient_checkpointing=True,
      learning_rate=5e-4,
      output_dir=output_dir,
      logging_dir=output_dir,
      report_to="tensorboard",
      num_train_epochs=5,
      per_device_train_batch_size=32
    )

    # create dataset
    generate_dataset('data/rft.json', 20)
    rft_data = json.load(open(Path(__file__).parent.parent / "data" / "rft.json"))
    dataset = TokenizedDataset(llm.tokenizer, rft_data, format_example)

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
