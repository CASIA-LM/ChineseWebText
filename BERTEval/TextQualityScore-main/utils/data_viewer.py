import json    
import random

if __name__ == "__main__":
    with open("/mnt/data/jpu/cc_clean/2022-49/clean/cleared0_0000.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]

    
    # random.seed(20230813)
    random.shuffle(data)


    # random.shuffle(filter_data_25)
    for n in range(0, 20):
        print("得分前10%, 第 " + str(n) + " 条文本 ----------------------")
        print(data[n]["text"][:512])
        print("\n\n")

        print()
