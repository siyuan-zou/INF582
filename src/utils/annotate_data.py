from ollama import chat
from ollama import ChatResponse
import time


def annotation_data():
    start_time = time.time()
    # 运行 DeepSeek-R1 进行文本处理
    response = chat(model='deepseek-r1', messages=[
    {
        'role': 'user',
        'content': 'Why is the sky blue?',
    },
    ])

    # 输出结果
    end_time = time.time()
    
    print(response['message']['content'])
    print("Time taken: ", end_time - start_time)

if __name__ == "__main__":
    annotation_data()