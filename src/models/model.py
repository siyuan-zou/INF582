import torch
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

class DS7b4bit(pl.LightningModule):
    def __init__(self, cfg):
        super(DS7b4bit, self).__init__()
        model_name = "deepseek-ai/deepseek-llm-7b-base"

        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16  # Use float16 for faster computation
        )

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        # Apply LoRA for memory-efficient fine-tuning
        lora_config = LoraConfig(
            r=8,  # Low-rank adaptation size
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
            lora_dropout=0.05,
            bias="none"
        )
        

        self.cfg = cfg
        self.model = get_peft_model(model, lora_config)
        # model.print_trainable_parameters()
        self.tokenizer = tokenizer

    def forward(self, x):
        return self.model(x)

