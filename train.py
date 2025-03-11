from src.datamodules.datamodule import TextSummaryDataModule
from src.models.model import DS7b4bit
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

import torch
import hydra
import wandb
import logging
import pytorch_lightning as pl

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from src.utils.sizedatamodule import SizeDatamodule

import os
from datetime import datetime
from omegaconf import DictConfig

torch.cuda.empty_cache()

def training(cfg: DictConfig, date=""):
    
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    if cfg.logger.mode == "offline":
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = cfg.paths.logs

    ### callbacks ###
    name = f"{date}_{cfg.logger.name}"

    ######### logger #########

    ### load model ###
    
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
    

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if cfg.train:
        # 创建DataModule实例
        data_module = TextSummaryDataModule(data_folder=f"{cfg.paths.data}", tokenizer=tokenizer, batch_size=32)
        # 准备数据
        data_module.prepare_data()
        # 设置数据
        data_module.setup()

        trainer = Trainer(
            model=model,
            args=hydra.utils.instantiate(cfg.args),
            train_dataset=data_module.train_dataset,
            eval_dataset=data_module.val_dataset,
        )

        trainer.train()

        trainer.save_model("./results/deepseek7b_finetuned")  # 保存模型
        tokenizer.save_pretrained("./results/deepseek7b_finetuned")  # 保存 tokenizer
    
        
    ### END ###
    if cfg.log:
        wandb.finish()

@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """Train function for SLM model."""
    ### set-up ###
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    torch.set_float32_matmul_precision("high")
    logger_console = logging.getLogger(__name__)
    logger_console.info("start")
    date_time = datetime.now().strftime("%Y-%m-%d_%Hh%M")

    training(cfg=cfg, date=date_time)
    logger_console.info("finished")


if __name__ == "__main__":
    main()
