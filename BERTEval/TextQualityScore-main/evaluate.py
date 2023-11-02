import numpy as np
from tqdm import tqdm
import torch, json, os
from utils.encode import encode_document
from types import SimpleNamespace
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig
from utils.data import DocumentDataset, DocumentDatasetForEvaluation
from network.model_architechure_bert_multi_scale import DocumentBertScoringModel


def misc(predictions, labels):
        prediction_scores = []
        label_scores = []
        for index, item in enumerate(predictions):
            prediction_scores.append(item)
            label_scores.append(labels[index])

        # test_eva_res = evaluation(label_scores, prediction_scores)
        qwk = cohen_kappa_score(label_scores, prediction_scores, weights='quadratic')
        pearson = np.corrcoef(label_scores, prediction_scores)[0][1]
        return pearson, qwk


def eval_model(model, dataloader, current_epoch, device, config):
    model.eval()
    predictions = []
    labels = []
    texts = []
    print()
    for _, ((text, representation_doc_token, representation_segment_list), label) in enumerate(
        tqdm(dataloader, desc=f"Evaluation", leave=False)):
        representation_doc_token = representation_doc_token.to(device)
        representation_segment_list = [seg.to(device) for seg in representation_segment_list]
        label = label.to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(representation_doc_token, representation_segment_list)

        # 记录预测值和标签
        pred = output.detach().cpu().numpy().tolist()
        gt = label.detach().cpu().numpy().tolist()
        text = text[0]

        predictions.extend(pred)
        labels.extend(gt)
        texts.extend(text)
    
    return predictions, labels, texts


    

if __name__ == "__main__":
    # # config
    # with open("configs/base_config.json", "r", encoding="utf-8") as f:
    #     train_cfg = json.load(f)
    # train_cfg = SimpleNamespace(**train_cfg)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_cfg = BertConfig.from_pretrained(train_cfg.bert_model_path)

    # # dataloader
    # eval_file = "data/text_score/webtext.json"
    # tokenizer = BertTokenizer.from_pretrained(train_cfg.tokenizer_path)
    
    # num_workers = train_cfg.num_workers if hasattr(train_cfg, "num_workers") \
    #                                         else int(os.cpu_count() / 2) 
    
    # eval_dataset = DocumentDatasetForEvaluation(eval_file, tokenizer,
    #                                model_cfg.max_position_embeddings,
    #                                model_cfg.doc_cfg, model_cfg.segment_cfg, 
    #                                train_cfg)
    
    # eval_dataloader = DataLoader(eval_dataset, batch_size=train_cfg.eval_batch_size,
    #                              shuffle=False, num_workers=num_workers,
    #                              drop_last=True)
    
    # # model
    # model = DocumentBertScoringModel(train_cfg)

    # # 加载训练模型
    # ckpt_path = "models/text_score/pseudo_label/2023-09-05-08-26/model-epoch_99-step_all.pt"
    # checkpoint = torch.load(ckpt_path, map_location=device)
    # params = {}
    # # 将所有的 module. 删除
    # for k, v in checkpoint["state_dict"].items():
    #     if k.startswith("module."):
    #         k = k.replace("module.", "")
    #     params[k] = v
    # model.load_state_dict(params)
    # model.to(device)

    # # 验证模型
    # print("测试集指标")
    # preds, labels, texts = eval_model(model, eval_dataloader, 0, device, train_cfg)

    # data = []
    # for n in range(len(preds)):
    #     data.append({"text": texts[n], "pred": preds[n], "label": labels[n]})
   
    # # 保存结果
    # with open("data/text_score/webtext_pred.json", "w", encoding="utf-8") as f:
    #     json.dump(data, f, ensure_ascii=False, indent=4)
    
    # 结果分析
    with open("data/text_score/webtext_pred.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    import random
    # random.seed(20230813)
    random.shuffle(data)
    # 筛选
    # filter_data_10 = [d for d in data if d["pred"] >= 0.1]
    # filter_data_25 = [d for d in data if d["pred"] >= 0.25]
    # filter_data_50 = [d for d in data if d["pred"] >= 0.5]
    # filter_data_75 = [d for d in data if d["pred"] >= 0.75]
    # filter_data_90 = [d for d in data if d["pred"] >= 0.9]

    # print("分数大于0.1的数量: ", len(filter_data_10))
    # print("分数大于0.25的数量: ", len(filter_data_25))
    # print("分数大于0.5的数量: ", len(filter_data_50))
    # print("总文本数量: ", len(data))

    # 测试评分前 m % 的文本中高质量文本占比
    topk = round(0.3 * len(data))
    data_desc = sorted(data, key=lambda x: x["pred"], reverse=True)
    data_top_5 = data_desc[:topk]
    # 计算最低分数
    min_score = data_top_5[-1]["pred"]

    random.shuffle(data_top_5)


    # random.shuffle(filter_data_25)
    for n in range(0, 20):
        print("得分前10%, 第 " + str(n) + " 条文本 ----------------------")
        print(data_top_5[n]["text"][:512])
        print("\n\n")

        print()
