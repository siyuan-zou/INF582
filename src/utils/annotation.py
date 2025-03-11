import pandas as pd
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.cuda.empty_cache()
model_name = "Qwen/Qwen2.5-14B-Instruct-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,  # Change to torch.bfloat16 if needed
    trust_remote_code=True
)
model.eval()  # Set model to evaluation mode

def summarize_text(text):
    try:
        # Truncate text to 2000 characters and keep the paragraph
        prompt = f"Please give a brief summary of the following text:\n{text}"

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        summary = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return summary  # Return truncated text along with summary
    except Exception as e:
        logger.error(f"Error summarizing text: {text[:50]}... Error: {e}")
        return text, "Error in summarization"  # Return original text in case of error





if __name__ == "__main__":
    import pandas as pd
    import time
    import os

    df = pd.read_csv("data/output_dir/cleaned_filtered_extracted_18.csv")  # Assume a CSV file with 'text' column
    print(df.head())
    print(f"Loaded {df.shape[0]} texts from CSV")
    # Track progress and save intermediate results
    output_file = "summarized_documents18.csv"
    log_file = "summary_progress.log"
    checkpoint_csv = 'checkpoint_summarized_texts.csv'  # Path for checkpoint saving
    logging.basicConfig(filename='summarization_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    if checkpoint_csv and os.path.exists(checkpoint_csv):
        checkpoint_df = pd.read_csv(checkpoint_csv)
        checkpoint_index = len(checkpoint_df)
    else:
        checkpoint_df = pd.DataFrame(columns=['text', 'summary'])  # Empty DataFrame if no checkpoint
        checkpoint_index = 0

    # Summarize each text in the CSV and store the summaries with progress bar
    summaries = checkpoint_df['summary'].tolist()  # Start with the existing summaries in checkpoint
    for idx, text in tqdm(enumerate(df['text'][checkpoint_index:], start=checkpoint_index), 
                        desc="Summarizing texts", unit="text"):
        summary = summarize_text(text)
        summaries.append(summary)
        # print(f"Text length {len(text)}, Summary length {len(summary)}")
        # print(f"Summary: {summary}") 
        # Save checkpoint after every 100 texts (or adjust this frequency)
        if (idx + 1) % 100 == 0:
            checkpoint_df = pd.DataFrame({'summary': summaries})
            checkpoint_df.to_csv(checkpoint_csv, index=False)
            
            logger.info(f"Checkpoint saved at index {idx + 1}")

    # Add the final truncated texts and summaries to the dataframe
    df['summary'] = summaries

    # Save the results to a new CSV file
    df.to_csv(output_file, index=False)
    logger.info(f"Summarization complete. Results saved to {output_file}")