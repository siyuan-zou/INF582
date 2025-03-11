import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_FOLDER = "./data/summarized_data/"
DATA_OUTPUT = "./data/splited_data/"
VAL_SIZE = 0.1
TEST_SIZE = 0.1

def load_data_from_folder(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            all_data.append(df)
    
    # 合并所有文件的数据
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

def main():
    
    data = load_data_from_folder(DATA_FOLDER)

    train_data, temp_data = train_test_split(data, test_size=VAL_SIZE + TEST_SIZE, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE), random_state=42)

    train_data.to_csv(DATA_OUTPUT + "train.csv", index=False)
    val_data.to_csv(DATA_OUTPUT + "val.csv", index=False)
    test_data.to_csv(DATA_OUTPUT + "test.csv", index=False)

if __name__ == "__main__":
    main()
