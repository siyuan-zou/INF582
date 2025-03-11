from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from src.datamodules.datasets.dataset import TextSummaryDataset
import pandas as pd
import torch


# 加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained("./results/deepseek7b_finetuned")
tokenizer = AutoTokenizer.from_pretrained("./results/deepseek7b_finetuned")

# data_module = TextSummaryDataModule(data_folder="./data/summarized_data/", tokenizer=tokenizer, batch_size=1)
# # 准备数据
# data_module.prepare_data()
# # 设置数据
# data_module.setup()

test_data = pd.read_csv("./data/splited_data/test.csv")
test_dataset = TextSummaryDataset(test_data, tokenizer, max_length=512)

# 用于存储生成的摘要
predictions = []

# 逐批进行预测
model.eval()
with torch.no_grad():
    batch = test_dataset[0]
    input_ids = batch['input_ids'].unsqueeze(0)
    attention_mask = batch['attention_mask'].unsqueeze(0)
    
    # 生成预测
    outputs = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id)
    
    # 解码并将预测结果添加到列表
    for output in outputs:
        predicted_summary = tokenizer.decode(output, skip_special_tokens=True)
        predictions.append(predicted_summary)

    # 保存预测结果
    with open("./results/predictions1.txt", "w") as f:
        for prediction in predictions:
            f.write(prediction + "\n")

