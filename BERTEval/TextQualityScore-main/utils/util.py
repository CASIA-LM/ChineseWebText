import os, json
import numpy as np

def text_select_with_pred(file, score_threshold):
    # 读取jsonl文件
    data = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    selected_data = []
    # 确定要保留的文本
    for item in data:
        score = item["score"]
        text_segs = item["text"]
        info = item["info"]

        # indexes = np.where(text_index_array == text_id)[0]
        cur_text_bool = np.array(score) > score_threshold

        # 获取连续被判定为高质量文本对应的 index
        continuous_ranges = np.where(np.diff(np.concatenate((
            [False], cur_text_bool, [False]))))[0].reshape(-1, 2)
        
        if len(continuous_ranges) == 0:
            continue

        for start, end in continuous_ranges:
            text = "".join([text_segs[index] for index in range(start, end)])
            text_dict = {"text": text, "info": info}

            # 将筛选的文本写入文件 (jsonl格式)
            selected_data.append(text_dict)
    
    return selected_data


if __name__ == "__main__":
    file = "test\data\cleared0_0000.jsonl"
    score_threshold = 0.36
    
    selected_data = text_select_with_pred(file, score_threshold)
    import random
    random.shuffle(selected_data)
    
    for n in range(20):
        print("第{}条文本：".format(n))
        print("\n")
        print(selected_data[n]["text"])
        print("\n\n\n\n")