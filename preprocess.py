import os
import re
import json
import glob
import subprocess
import numpy as np
from tqdm import tqdm
from zhconv import issimp, convert
from collections import Counter, defaultdict
from multiprocessing import Pool
import argparse

space_p = re.compile(' |\n|\t')
punc_p = re.compile("\.|,|\?|\!|'|\"|。|，|？|！|‘|’|“|”")


def get_n_gram(txt, n):
    if len(txt) < n:
        return [txt]
    else:
        return [txt[i:i+n] for i in range(0, len(txt)-n)]


def norm_str2(s):
    norm_s = s.strip().lower()
    norm_s = space_p.sub('', norm_s)
    return norm_s


def count_lines(file_path):
    result = subprocess.run(['wc', '-l', file_path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.stderr:
        print(result.stderr)
        return None
    else:
        lines = int(result.stdout.decode().split()[0])
        return lines


def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            yield item


def judge_inner_repeat(text, threshold=0.5):
    norm_txt = norm_str2(text)
    n_gram_txt = get_n_gram(norm_txt, 13)
    n_gram_num = len(n_gram_txt)
    n_gram_counter = Counter(n_gram_txt)
    r_n_gram_num = sum([i for _, i in n_gram_counter.items() if i > 1])
    r2 = r_n_gram_num / n_gram_num if n_gram_num != 0 else 0
    if r2 > threshold:
        return True, f'n-gram {r2}, most_common {n_gram_counter.most_common(1)[0]}'
    return False, None



def preprocess(file_path, min_len=200, min_line_len=10, min_zh_percent=0.3, max_sensitive_word_per_line=0.5, max_repeat=0.5):
    sensitive_words = open("./dict.txt", "r").readlines()
    sensitive_words_dict = {}
    for word in sensitive_words:
        category, key = word.split('\t')
        key = key.strip()
        
        sensitive_words_dict[key] = category
    
    file_dir, file_name = os.path.split(file_path)
    
    text_extraction_dir = os.path.join(file_dir, "text_extraction")
    length_dir = os.path.join(file_dir, "length")
    character_dir = os.path.join(file_dir, "character")
    sensitive_dir = os.path.join(file_dir, "sensitive")
    duplication_dir = os.path.join(file_dir, "duplication")
    remain_dir = os.path.join(file_dir, "remain")
    
    os.makedirs(text_extraction_dir, exist_ok=True)
    os.makedirs(length_dir, exist_ok=True)
    os.makedirs(character_dir, exist_ok=True)
    os.makedirs(sensitive_dir, exist_ok=True)
    os.makedirs(duplication_dir, exist_ok=True)
    os.makedirs(remain_dir, exist_ok=True)


    text_extraction_path = os.path.join(text_extraction_dir, file_name)
    length_path = os.path.join(length_dir, file_name)
    character_path = os.path.join(character_dir, file_name)
    sensitive_path = os.path.join(sensitive_dir, file_name)
    duplication_path = os.path.join(duplication_dir, file_name)
    remain_path = os.path.join(remain_dir, file_name)


    # total_lines = count_lines(file_path)
    with open(text_extraction_path, "w") as f_text_extraction, open(length_path, "w") as f_length, open(character_path, "w") as f_character, open(sensitive_path, "w") as f_sensitive, open(duplication_path, "w") as f_duplication, open(remain_path, "w") as f_remain:
        try:
            # for item in tqdm(read_data(file_path), total=total_lines):
            for item in read_data(file_path):
                new_item = {
                    "text": item["raw_content"],
                    "info": {
                        "url": item["url"],
                        "title": item["title"],
                        "source_domain": item["source_domain"]
                    }
                }
                
                f_text_extraction.write(json.dumps(new_item, ensure_ascii=False) + '\n')
                
                
                text = new_item["text"]
                # 数据长度
                text_length = len(text)
                # 平均行长度
                average_line_length = text_length / len(text.split('\n')) if len(text.split('\n')) > 0 else 0

                if text_length <= min_len or average_line_length < min_line_len:
                    f_length.write(json.dumps(new_item, ensure_ascii=False) + '\n')
                    continue
            

                # 中文字符比例
                zh_chars = re.findall(r'[\u4e00-\u9fa5]', text)
                zh_percent = len(zh_chars) / text_length
                # 繁体字
                simplified_text = convert(text, "zh-cn")
                if zh_percent < min_zh_percent or text != simplified_text:
                    f_character.write(json.dumps(new_item, ensure_ascii=False) + '\n')
                    continue
                
                
                # 敏感词出现频率
                sensitive_word_times = 0
                for word in sensitive_words_dict:
                    if word in text:
                        sensitive_word_times += 1
                sensitive_word_per_line = sensitive_word_times / len(text.split('\n'))

                
                if sensitive_word_per_line > max_sensitive_word_per_line:
                    f_sensitive.write(json.dumps(new_item, ensure_ascii=False) + '\n')
                    continue
                
                
                
                # 内部重复率
                repeat, r = judge_inner_repeat(text, threshold=max_repeat)
                if repeat:
                    f_duplication.write(json.dumps(new_item, ensure_ascii=False) + '\n')
                    continue
                
                # 保留
                f_remain.write(json.dumps(new_item, ensure_ascii=False) + '\n')
                
                
        except Exception as e:
            print(e)
            print(file_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dates', nargs='+', default=None)
    args = parser.parse_args()
    
    for date in args.dates:
        print(date)
        folder_path = f"./{date}/*.jsonl"
        
        # 初步筛选
        with Pool(64) as p:
            for _ in tqdm(p.imap_unordered(preprocess, glob.glob(folder_path)), total=len(glob.glob(folder_path))):
                pass
        
        # for f in glob.glob(folder_path):
        #     filter_dict_list.append(preprocess(f))
        #     break
        # break
    
    
    
