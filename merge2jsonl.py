from glob import glob
from gzip import open as gzopen
from tqdm import tqdm
from json import loads
from json import dumps
import os
import multiprocessing as mp
from math import floor
import argparse

# 分割总文件
def cut_list(lists, cut_len):
    res_data = []
    if len(lists) > cut_len:
        for i in range(int(len(lists) / cut_len)):
            cut_a = lists[cut_len * i:cut_len * (i + 1)]
            res_data.append(cut_a)

        last_data = lists[int(len(lists) / cut_len) * cut_len:]
        if last_data:
            res_data.append(last_data)
    else:
        res_data.append(lists)
    return res_data

def merge(subfiles, k): 
    # 保存文件
    write_pass = open(cleared + f"/cleared{k}.jsonl", "w", encoding='utf-8')
    
    passwrite = write_pass.write

    print(f"第{k + 1}个进程，一共处理{len(subfiles)} 文件。")

    for file in tqdm(subfiles, desc=f"第{k+1}个进程，合并中文文本", total=len(subfiles)):
        print('&') # 为了不覆盖掉实时记录
        with gzopen(file, "r") as r:
            for line in r:
                try:
                    line_json = loads(line)
                    passwrite(dumps(line_json, ensure_ascii=False))
                    passwrite("\n")
                except Exception as e:
                    pass

if __name__ == '__main__':

    #从命令行读取文件路径
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help = "Declare the path of unclean data. ")
    parser.add_argument('--target', type=str, help = "Declare output path.")
    arg = parser.parse_args()
    print("清洗数据目录：",arg.source,"\n数据保存目录：",arg.target)
    # 存储文件名
    dump_id_folder = arg.source
    cleared = arg.target
    # 创建存储目录
    if not os.path.exists(cleared):
        os.makedirs(cleared)
    # 读取待处理文件
    files = glob(dump_id_folder + "/*/zh*.gz")
    print(f"一共处理{len(files)} 文件。")

    N = 30  # 指定调用cpu数量
    length_of_each = floor(len(files)/N)  # 计算每个cpu处理的文件数量，向下取整

    # 获取分割文件的目录
    subfileslist = cut_list(files, length_of_each)

    # 多进程运行
    for j in range(N):
        subp = mp.Process(target=merge, kwargs={'subfiles': subfileslist[j], 'k': j})
        subp.start()