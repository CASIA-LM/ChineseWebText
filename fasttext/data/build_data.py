#coding = utf-8
import json
import jieba
import random
from tqdm.auto import tqdm
from multiprocessing import Pool

stopwords = set([x.strip() for x in open("./cn_stopwords.txt").readlines()])

def build(args):
    line, label = args
    text = line['text']
    segs = jieba.lcut(text)
    segs = [x for x in segs if len(x) > 1]
    segs = [x for x in segs if x not in stopwords]
    text = " ".join(segs)
    
    return f'__label__{label} ' + text

def main(input_file, label):
    data = []
    print("reading file...")
    with open(input_file, 'r', encoding='utf-8') as r_f:
        for line in tqdm(r_f):
            data.append(json.loads(line))
    
    processed_data = []
    data = [(x, label) for x in data]
    print("processing data...")
    with Pool(16) as p:
        for d in tqdm(p.imap_unordered(build, data), total=len(data)):
            processed_data.append(d)
    
    return processed_data
    

if __name__ == '__main__':
    print("processing clean...")
    clean_data = main("./clean.jsonl", label="clean")
    random.shuffle(clean_data)
    train_size = int(len(clean_data) * 0.99)
    clean_train_data = clean_data[:train_size]
    clean_test_data = clean_data[train_size:]

    print("processing dirty...")
    dirty_data = main("./dirty.jsonl", label="dirty")
    random.shuffle(dirty_data)
    train_size = int(len(dirty_data) * 0.99)
    dirty_train_data = dirty_data[:train_size]
    dirty_test_data = dirty_data[train_size:]    
    
    train_data = clean_train_data + dirty_train_data
    random.shuffle(train_data)
    
    test_data = clean_test_data + dirty_test_data
    
    with open("./train.txt", "w", encoding='utf-8') as w_f:
        for item in tqdm(train_data):
            w_f.write(item + '\n')
    
    with open("./test.txt", "w", encoding='utf-8') as w_f:
        for item in tqdm(test_data):
            w_f.write(item + '\n')