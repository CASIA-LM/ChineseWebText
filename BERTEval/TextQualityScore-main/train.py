import datetime
import numpy as np
from tqdm import tqdm
from evaluate import misc
import torch, json, re, os
from utils.data import DocumentDataset
from types import SimpleNamespace
from torch.utils.data import DataLoader
from network.loss import DocumentBertScoringLoss
from transformers import BertTokenizer, BertConfig
from network.model_architechure_bert_multi_scale import DocumentBertScoringModel

# 冻结bert模型前 N 层的参数
def freeze_bert_params(model, freeze_layers):
    pattern1 = r'bert.embeddings'
    pattern2 = r"bert.encoder.layer"
    num_pattern = r'\d+'
    for name, param in model.named_parameters():
        if re.search(pattern1, name):
            param.requires_grad = False
            # print(name)
        elif re.search(pattern2, name):
            if int(re.search(num_pattern, name).group()) < freeze_layers:
                param.requires_grad = False
                # print(name)


def train_one_epoch(model: torch.nn.Module, dataloader: DataLoader, 
                    optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, 
                    device: torch.device, loss_scaler, current_epoch: int, save_path: str,
                    config: SimpleNamespace):
    
    model.train(True)
    predictions = []
    labels = []
    losses = []
    # print("epoch " + str(current_epoch) + "/" + str(config.epochs) + '...')
    for step, (representation_doc_token, label) in enumerate(
        tqdm(dataloader, desc=f"Epoch {current_epoch+1}/{config.epochs}", leave=False)):

        representation_doc_token = representation_doc_token.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(representation_doc_token)
            loss = criterion(output, label)
        
        loss_scaler.scale(loss).backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()

        if (step + 1) % config.log_steps == 0:
            pearson, qwk = misc(np.concatenate(predictions, axis=0, dtype=np.float16), 
                                np.concatenate(labels, axis=0, dtype=np.float16))
            
            # 损失/misc打印
            print(f'\nEpoch [{current_epoch +1}/{config.epochs}], Loss: {loss.item():.6f}, Pearson: {pearson: .6f}, QWK: {qwk: .6f}')

        # 记录预测值\标签\损失
        predictions.append(output.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())
        losses.append(loss.detach().cpu().numpy())

        # 指定step保存模型
        if config.save_stragtegy == "steps" and (step + 1) % config.save_steps == 0:
            save_model(model, optimizer, step, current_epoch, save_path, config)

    # 拼接每个batch的预测值和标签
    predictions = np.concatenate(predictions, axis=0, dtype=np.float16)
    labels = np.concatenate(labels, axis=0, dtype=np.float16)
    losses = np.concatenate(losses, axis=0, dtype=np.float16)
    
    return predictions, labels, losses

# 评估模型 pearson, qwk 两个指标
def eval_model(model, dataloader, current_epoch, device, config):
    model.eval()
    predictions = []
    labels = []
    print()
    for _, (representation_doc_token, label) in enumerate(
        tqdm(dataloader, desc=f"Evaluation {current_epoch+1}/{config.epochs}", leave=False)):
        representation_doc_token = representation_doc_token.to(device)
        label = label.to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(representation_doc_token)

        # 记录预测值和标签
        predictions.append(output.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())

    # 拼接每个batch的预测值和标签
    predictions = np.concatenate(predictions, axis=0, dtype=np.float16)
    labels = np.concatenate(labels, axis=0, dtype=np.float16)

    pearson, qwk = misc(predictions, labels)
    print("epoch " + str(current_epoch) + ", pearson:", float(pearson))
    print("epoch " + str(current_epoch) + ", qwk:", float(qwk))



def save_model(model, optimizer, step, epoch, save_path, config):
    # 保存模型
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 保存pytorch模型
    state = {"epoch": epoch, "step": step, "state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict(), "config": config}

    torch.save(state, save_path + "model-epoch_" + str(epoch) + "-step_" + str(step) + ".pt")

    # 如果save_path 下有超过设定的文件数，则删除最旧的文件
    file_list = os.listdir(save_path)
    if len(file_list) > config.save_total_limit:
        file_list.sort(key=lambda fn: os.path.getmtime(save_path + "/" + fn))
        os.remove(save_path + "/" + file_list[0])


def train(train_cfg: SimpleNamespace):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model config
    model_cfg = BertConfig.from_pretrained(train_cfg.bert_model_path)
    
    # dataloader
    tokenizer = BertTokenizer.from_pretrained(train_cfg.tokenizer_path)
    
    train_dataset = DocumentDataset(train_cfg.train_file, tokenizer, 
                                    model_cfg.max_position_embeddings, 
                                    model_cfg.doc_cfg, model_cfg.segment_cfg, 
                                    train_cfg)
    
    num_workers = train_cfg.num_workers if hasattr(train_cfg, "num_workers") \
                                            else int(os.cpu_count() / 2) 
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, 
                                  shuffle=True, num_workers=num_workers)
    
    eval_dataset = DocumentDataset(train_cfg.eval_file, tokenizer,
                                   model_cfg.max_position_embeddings,
                                   model_cfg.doc_cfg, model_cfg.segment_cfg, 
                                   train_cfg)
    
    eval_dataloader = DataLoader(eval_dataset, batch_size=train_cfg.batch_size,
                                 shuffle=True, num_workers=num_workers)
    
    # model
    model = DocumentBertScoringModel(train_cfg)
    model.to(device)

    # 将所有bert模型最后一层外的参数冻结
    freeze_bert_params(model, train_cfg.bert_freeze_layers)
    
    # optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=train_cfg.learning_rate, 
                                 betas=(train_cfg.beta1, train_cfg.beta2),
                                 weight_decay=train_cfg.weight_decay)
    
    # loss scaler
    loss_scaler = torch.cuda.amp.GradScaler()

    # criterion
    criterion = DocumentBertScoringLoss(train_cfg.loss_cfg, device)

    # 加载 checkpoint
    if train_cfg.checkpoint != "":
        checkpoint = torch.load(train_cfg.checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # 模型保存地址
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    save_path = train_cfg.model_directory + '/' + train_cfg.train_mode + "/" + datetime_str + "/"

    # train
    for epoch in range(train_cfg.epochs):

        print("epoch " + str(epoch) + " start training -------------------------------------------------\n")
        # 训练一个epoch
        preds, labels, losses = train_one_epoch(model, train_dataloader, optimizer, criterion,
                                        device, loss_scaler, current_epoch=epoch, save_path=save_path,
                                        config=train_cfg)
        
        # 训练集评估指标
        print("训练集指标")
        pearson, qwk = misc(preds, labels)
        print("epoch " + str(epoch) + ", loss: " + str((np.mean(losses).item())))
        print("epoch " + str(epoch) + ", pearson:", float(pearson))
        print("epoch " + str(epoch) + ", qwk:", float(qwk))

        # 测试集评估指标
        print("测试集指标")
        eval_model(model, eval_dataloader, epoch, device, train_cfg)

    # 保存模型
    save_model(model, train_cfg)



if __name__ == "__main__":
    with open("configs/base_config.json", "r", encoding="utf-8") as f:
        train_cfg = json.load(f)
    train_cfg = SimpleNamespace(**train_cfg)
    train(train_cfg)
    # 加载训练参数
    

    
        


