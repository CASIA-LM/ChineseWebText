#coding = utf-8
import os
import json
import jieba
import random
import argparse
import fasttext
from tqdm.auto import tqdm
from multiprocessing import Pool

jieba.setLogLevel(20)


stopwords = set([x.strip() for x in open("./data/cn_stopwords.txt").readlines()])


def main(train_file, test_file):
    model = fasttext.train_supervised(input=train_file, lr=0.5, epoch=5, wordNgrams=2)
    model.save_model("./output/model.bin")
    print(model.test(test_file))


def build(text):
    segs = jieba.lcut(text)
    segs = [x for x in segs if len(x) > 1 and x not in stopwords]
    return " ".join(segs)


def predict(input_file):
    file_dir, file_name = os.path.split(input_file)
    
    output_dir = os.path.join(file_dir, "fasttext")
    output_dir = output_dir.replace("data2", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, file_name)
    
    lines = []
    with open(input_file, 'r', encoding='utf-8') as r_f:
        for line in r_f:
            lines.append(json.loads(line))
    
    # print("jieba cutting...")d
    seg_texts = [build(''.join(line['text'])) for line in lines]
    
    # print("predicting...")
    labels, values = model.predict(seg_texts)
    
    # print("writing...")
    with open(output_file, 'w', encoding='utf-8') as w_f:
        for label, value, line in zip(labels, values, lines):
            _label = label[0].replace("__label__", "")
            _value = value[0] if value[0] <= 1 else 1
            line['fasttext_value'] = float(_value) if _label == 'clean' else float(1 - _value)
            
            w_f.write(json.dumps(line, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="test")
    parser.add_argument('--train_file', type=str, default="./data/train.txt")
    parser.add_argument('--test_file', type=str, default="./data/test/txt")
    parser.add_argument('--dates', nargs='+', default=None)
    args = parser.parse_args()
    
    if args.mode == "train":
        main("./data/train.txt", "./data/test.txt")
    elif args.mode == "test":
        model = fasttext.load_model("./output/model.bin")

        for date in args.dates:
            print(date)
            clean_root = f"../{date}/remain"
            file_name_list = [x for x in os.listdir(clean_root) if x.endswith(".jsonl")]
            
            # for file_name in tqdm(file_name_list):
            #     file_path = os.path.join(clean_root, file_name)
            #     print(file_path)
            #     predict(file_path)
            
            with Pool(64) as p:
                for _ in tqdm(p.imap_unordered(predict, [os.path.join(clean_root, file_name) for file_name in file_name_list]), total=len(file_name_list)):
                    pass
    else:
        print("Invalid mode!")
    

