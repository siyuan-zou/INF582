import os
import pandas as pd
import re
from bs4 import BeautifulSoup
import opencc

# 初始化繁体转简体转换器
converter = opencc.OpenCC("t2s")

# 判断文本是否包含足够的中文字符（避免只包含乱码或无关字符）
def is_mostly_chinese(text, threshold=0.8):
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)  # 统计中文字符
    return len(chinese_chars) / max(len(text), 1) > threshold  # 中文字符占比超过阈值
# 文本清理函数
def clean_text(text):
    # 移除不需要的内容
    text = re.sub(r'\[.*?\]', '', text)  # 移除方括号内的内容（如链接）
    text = re.sub(r'\<.*?\>', '', text)  # 移除尖括号内的内容（HTML 标签）
    text = re.sub(r'\n+', '\n', text)  # 规范换行符
    text = text.strip()  # 移除首尾空白字符
    text = re.sub(r" ", " ", text)  # 移除额外空格
    text = re.sub(r"-\{[A-Za-z]\|[^{}]+\}-", "", text)  # 移除 `{H|英文:中文;英文:中文;}` 这种格式
    text = re.sub(r"-\{\}-", "", text)  # 移除 `-{}-`
    text = re.sub(r"[\「\(\[\{\（\【\《][^\w\u4e00-\u9fa5]*[\)\]\}\）\】\」\》]", "", text)  # 移除包含无效字符的括号
    
    return text
def truncate_text(text, max_length=2000):
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    last_punctuation = max(truncated.rfind("。"), truncated.rfind("！"), truncated.rfind("？"))
    
    if last_punctuation != -1:
        return truncated[:last_punctuation + 1]  # 保留到最后一个标点符号
    else:
        return truncated  # 若无标点，则直接截断

# 数据提取与清理
def extract_data(data_path, output_path, len_doc,batch=1):
    all_texts = set()  # 使用 set 存储去重的文本
    count = 0

    # 遍历目录中的所有文件
    for subdir, _, _ in os.walk(data_path):
        for root, _, files in os.walk(subdir):
            for file in files:
                file_path = os.path.join(root, file)
                
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()

                # 移除 <templatestyles> 相关标签
                raw_text = re.sub(r"<templatestyles[^>]*>", "", raw_text, flags=re.IGNORECASE | re.DOTALL)
                raw_text = re.sub(r"templatestyles\s+src=\"[^\"]*\"", "", raw_text, flags=re.IGNORECASE)

                # 进行繁体转简体转换
                raw_text = converter.convert(raw_text)

                # 按 </doc> 分割文本
                sections = raw_text.split("</doc>")

                # 清理文本
                for section in sections:
                    section = BeautifulSoup(section, "html.parser").get_text().strip()
                    section = clean_text(section)
                    section = truncate_text(section)
                    # 过滤短文本、非中文文本和去重
                    if len(section) >= 1000 and is_mostly_chinese(section) and section not in all_texts:
                        all_texts.add(section)
                        count += 1
                        print(f"\r已处理 {count} 条文本,共{len_doc}条文本", end="")

                    # 限制最大处理文本数
                    if count >= len_doc:
                        break
                if count >= len_doc:
                    break
            if count >= len_doc:
                break

    # 转换为 DataFrame 并保存
    df = pd.DataFrame({"text": list(all_texts)})
    df=df.dropna()
    num_data=df.shape[0]
    num_per_batch=num_data//batch
    for i in range(batch):
        df[i*num_per_batch:(i+1)*num_per_batch].to_csv(output_path+"_"+str(i)+".csv", index=False, encoding="utf-8")
        print(f"已处理 {num_per_batch} 条有效文本，并存储到 {output_path}_{i}")
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"已处理 {num_data} 条有效文本，并存储到 {output_path}")

# 运行示例





if __name__ == "__main__":
    data_folder = "output_wiki_texts"  # 替换为你的数据目录
    output_file = "data/output_dir/cleaned_filtered_extracted"
    extract_data(data_folder,output_file, 12000,batch=30)