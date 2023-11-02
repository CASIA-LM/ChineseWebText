# 思想：尝试使用伪标签的方法，对于训练一定epoch的模型，尝试着将训练集中Common Crawl类别的样本中得分显著较高的(例如大于0.5)的样本移入正样本。同样的正样本中得分显著低的移入负样本。
import datetime
import numpy as np
from tqdm import tqdm
import torch, json, os
from evaluate import misc
from utils.sampler import *
from types import SimpleNamespace
from torch.utils.data import DataLoader
from utils.data import DocumentDatasetForUST
from train import freeze_bert_params, save_model
from network.loss import DocumentBertScoringLoss
from transformers import BertTokenizer, BertConfig
from network.model_architechure_bert_multi_scale import DocumentBertScoringModel


def train_one_epoch(model: torch.nn.Module, dataset: DocumentDatasetForUST, 
                    optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
                    loss_scaler, current_epoch: int, config: SimpleNamespace, device):
    
    model.train(True)
    dataset.set_mode("train")

    # train dataloader
    num_workers = train_cfg.num_workers if hasattr(train_cfg, "num_workers") \
                                        else int(os.cpu_count() / 2) 
    world_size = torch.cuda.device_count()
    dataloader = DataLoader(
        dataset, batch_size=train_cfg.batch_size * world_size, 
        num_workers=num_workers, shuffle=False, drop_last=False
    )

    predictions = []
    labels = []
    losses = []

    for representation_doc_token, label, confidence, text_list in tqdm(
        dataloader, desc=f"Epoch {current_epoch+1}/{config.epochs}", 
        leave=True):
        # representation_doc_token = representation_doc_token.to(device)
        confidence = confidence.to(device)
        label = label.to(device)

        # train one epoch
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(representation_doc_token)
            loss = criterion(output, label, confidence)
        
        loss_scaler.scale(loss).backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()

        # 记录预测值\标签\损失
        predictions.append(output.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())
        losses.append(loss.detach().cpu().numpy())

    # 拼接每个batch的预测值和标签
    predictions = np.concatenate(predictions, axis=0, dtype=np.float16)
    labels = np.concatenate(labels, axis=0, dtype=np.float16)
    losses = np.concatenate(losses, axis=0, dtype=np.float16)
    
    return predictions, labels, losses



def select_train_sample(model: torch.nn.Module, dataset: DocumentDatasetForUST, config):
    """
    siaamse model 交叉对各自的训练集进行预测，其中负样本评分前5%的样本移入正样本，
    正样本评分后5%的样本移入负样本
    3种采样方式: 
    ust_ 按照 UST 工作给出的方法进行采样
    score_ 按照平均得分采样
    uniform 均匀采样

    3种确定weight/confidence的方式
    var 按照 UST 工作给出的方法, 根据方差确定样本权重
    score 按照得分确定样本的权重
    uniform 均匀采样
    """
    # 候选数据采样
    dataset.candidate_data_generate()
    dataset.set_mode("sample")
    print("数据采集完成")

    if 'ust' in config.sample_scheme:
        # 蒙特卡洛dropout验证， 输出结果
        ids, y_pred, y_var, y_T, y_mean = mc_dropout_evaluate(model, dataset, config)
        # 对pseudo label进行采样
        if 'eas' in config.sample_scheme:
            f_ = sample_by_bald_easiness

        if 'eas' in config.sample_scheme and 'clas' in config.sample_scheme:
            f_ = sample_by_bald_class_easiness

        if 'dif' in config.sample_scheme:
            f_ = sample_by_bald_difficulty

        if 'dif' in config.sample_scheme and 'clas' in config.sample_scheme:
            f_ = sample_by_bald_class_difficulty

        ids_selected, pseudo_labels, var = f_(
            ids, y_var, y_pred, int(config.unsupvise_size), (0, 1), y_T)


    elif 'score' in config.sample_scheme:
        # 蒙特卡洛dropout验证， 输出结果
        ids, y_pred, y_var, y_T, y_mean = mc_dropout_evaluate(model, dataset, config)
        # 对pseudo label进行采样
        ids_selected, pseudo_labels, var = sample_by_score_class_easy(
            ids, y_pred, y_mean, y_var, int(config.unsupvise_size))

    else:
        # 蒙特卡洛dropout验证， 输出结果
        ids, y_pred, y_var, _, y_mean = mc_dropout_evaluate(model, dataset, config)
        ids_selected, pseudo_labels, var= uniform_sample(ids, y_pred, y_var, int(config.unsupvise_size), (0, 1))

    
    if config.confidence_scheme == "var":
        # 对数化 confidence
        confidences = - np.log(var + 1e-10) * config.alpha
    
    elif config.confidence_scheme == "score":
        misc = np.array([1 - y_mean[n] if y_pred[n] == 0 else y_mean[n] for n in range(len(y_pred))])
        misc = misc / np.sum(misc)
        confidences = - np.log(misc + 1e-10) * config.alpha
    else:
        confidences = np.ones((len(ids), ))


    # 根据 id_selected 变更数据集，并输入各样本的伪标签与confidence
    dataset.select_samples(ids_selected, pseudo_labels, confidences) 



def mc_dropout_evaluate(model: torch.nn.Module, dataset: DocumentDatasetForUST, config):

    # MC Dropout Sample DataLoader
    num_workers = train_cfg.num_workers if hasattr(train_cfg, "num_workers") \
                                        else int(os.cpu_count() / 2) 
    world_size = torch.cuda.device_count()
    batch_size = train_cfg.eval_batch_size * world_size
    sample_dataloader = DataLoader(
        dataset, num_workers=num_workers, shuffle=False,
        batch_size=batch_size,
        drop_last=False
    )
    
    # 若基于score采样，则设 T=1
    if 'score' in config.sample_scheme:
        config.T = 1

    y_T  = np.zeros((int(config.candidate_data_num), config.T))
    ids_list = []
    text_list = []
    
    # MC Dropout 随机推理实验
    model.train()
    for t in range(config.T):
        
        for batch_index, (ids, representation_doc_token, text_segs) in enumerate(tqdm(
            sample_dataloader, desc=f"Data with pesudo label sampling, MC Dropout {t + 1}/{config.T}", 
            leave=True)):

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output = model(representation_doc_token)
            
            y_T[batch_index * batch_size:(batch_index + 1) * batch_size, t] = output.detach().cpu().numpy()

            if t == 0:
                ids_list.extend(ids.tolist())
                text_list.extend(text_segs)
    
    print("MC Dropout 随机推理完成")

    # 收集预测的正负样本
    pos_threshold = np.sort(y_T.flatten())[-int(config.positive_ratio * len(y_T.flatten()))]
    neg_threshold = np.sort(y_T.flatten())[-int(config.negative_ratio * len(y_T.flatten()))]
    y_pred_pos = np.array([0 if np.argmax(np.bincount(pred_row))==0 else 1 for pred_row in y_T > pos_threshold])
    y_pred_neg = np.array([0 if np.argmax(np.bincount(pred_row))==0 else 1 for pred_row in y_T < neg_threshold])
    pos_indexes = np.argwhere(y_pred_pos==1)[:, 0]
    neg_indexes = np.argwhere(y_pred_neg==1)[:, 0]
    y_T = np.concatenate((y_T[pos_indexes, :], y_T[neg_indexes, :]), axis=0)
    id_array = np.concatenate((np.array(ids_list)[pos_indexes], np.array(ids_list)[neg_indexes]))

    # # 计算均值
    y_mean = np.mean(y_T, axis=1) 

    # 计算预测值
    y_pred = np.concatenate((np.ones((len(pos_indexes),)), np.zeros((len(neg_indexes),))))
    
    # 计算方差
    y_var = np.var(y_T, axis=1)

    return id_array, y_pred, y_var, y_T, y_mean


def ust_train_one_epoch(model, dataset, optimizer, criterion, loss_scaler, cur_epoch, 
                        save_path, config, device):
    
    # step1: 对未标记数据进行采样生成伪标签
    select_train_sample(model, dataset, config)

    # step2: 训练 siamese model1 

    pred, labels, losses = train_one_epoch(model, dataset, optimizer, criterion, loss_scaler, 
                                           cur_epoch, config, device)
    
    # 训练集评估指标

    pearson, qwk = misc(pred, labels)

    print("训练集指标")
    print("epoch " + str(cur_epoch) + ", loss: " + str((np.mean(losses).item())))
    print("epoch " + str(cur_epoch) + ", pearson:", float(pearson))
    print("epoch " + str(cur_epoch) + ", qwk:", float(qwk))

    # 保存模型
    save_model(model, optimizer, "all", cur_epoch, save_path, config)


def train(train_cfg):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 定义模型
    model_cfg = BertConfig.from_pretrained(train_cfg.bert_model_path)
    model = DocumentBertScoringModel(train_cfg)
    freeze_bert_params(model, train_cfg.bert_freeze_layers)
    model.to(device)

    # 加载 checkpoint
    if train_cfg.checkpoint != "":
        checkpoint = torch.load(train_cfg.checkpoint)
        params = {}
        # 将所有的 module. 删除
        for k, v in checkpoint["state_dict"].items():
            if k.startswith("module."):
                k = k.replace("module.", "")
            params[k] = v
        model.load_state_dict(params)
    
    model = torch.nn.DataParallel(model)
        
    # dataloader
    tokenizer = BertTokenizer.from_pretrained(train_cfg.tokenizer_path)
    train_dataset = DocumentDatasetForUST(train_cfg, tokenizer, model_cfg.max_position_embeddings)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                   lr=train_cfg.learning_rate, betas=(train_cfg.beta1, train_cfg.beta2),
                                   weight_decay=train_cfg.weight_decay)
    
    # loss scaler
    loss_scaler = torch.cuda.amp.GradScaler()

    # criterion
    criterion = DocumentBertScoringLoss(train_cfg.loss_cfg, device, use_confidence=True)

    # 模型保存地址
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    save_path = train_cfg.model_directory + '/' + train_cfg.train_mode + "/" + datetime_str + "/"
    os.makedirs(save_path)

    # train
    for epoch in range(train_cfg.epochs):
        print("epoch " + str(epoch) + " start training -------------------------------------------------\n")

        # 训练一个epoch
        ust_train_one_epoch(model, train_dataset, optimizer, criterion, loss_scaler, epoch, 
                            save_path, train_cfg, device)
    

if __name__ == "__main__":
    with open("configs/ust_config.json", "r", encoding="utf-8") as f:
        train_cfg = json.load(f)
    
    train_cfg = SimpleNamespace(**train_cfg)
    train(train_cfg)


