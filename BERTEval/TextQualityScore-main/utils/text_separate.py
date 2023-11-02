import re
import string 
import os
import json
import threading
import multiprocessing


def text_separate(file, dist_dir):

    # 若不存在dist_dir，则创建
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)

    perfix = dist_dir + os.path.splitext(os.path.basename(file))[0]
    file_count = 0
    data = []

    # 获取json文件总行数
    max_num = 100000
    count = 0
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            # 获取jsonl中的json文件
            meta_data = json.loads(line)
            data.append(meta_data)
            count = count + 1
            
            # 若count大于max_num，则写入新的jsonl文件
            if count > max_num:
                
                dist_path = perfix + "_" + str(file_count).zfill(4) + '.json'
                with open(dist_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

                file_count = file_count + 1
                count = 0
                data = []
                print('地址: ' + dist_path + ', 状态: 保存完成')
            

if __name__ == '__main__':
    max_process = os.cpu_count() // 6
    print('CPU核心数: ' + str(max_process))
    pool = multiprocessing.Pool(processes=max_process)
    processes = multiprocessing.JoinableQueue()

    origin_dir = "/data/jpu/data/CommonCrawl/2023-06/"
    dist_dir = "/data/jpu/data/CommonCrawlJson/2023-06/"
    file_list = os.listdir(origin_dir)
    # origin_file_path = origin_dir + file_list[0]

    # text_separate(origin_file_path, dist_dir)
    
    print(file_list)

    for file in file_list:
        origin_file_path = origin_dir + file
        pool.apply_async(text_separate, (origin_file_path, dist_dir))
    
    pool.close()
    pool.join()
    print('全部完成')
