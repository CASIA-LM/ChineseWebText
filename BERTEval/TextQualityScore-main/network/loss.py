import torch
import torch.nn.functional as F


class MarginRankingLoss(torch.nn.Module):

    def __init__(self, device, bias=0.0):
        super(MarginRankingLoss, self).__init__()
        self.bias = bias
        self.device = device

    def forward(self, predictions, correct_output, weight=None):
        assert len(predictions) == len(correct_output)
        if weight is not None:
            assert len(weight) == len(correct_output)
        mr_loss = torch.tensor(0.0, device=self.device)
        batch_size = len(predictions)

        for m in range(batch_size):                                                                                                     
            for n in range(batch_size):
                pred_n, pred_m = predictions[n], predictions[m]
                gt_n, gt_m = correct_output[n], correct_output[m]
                if pred_m > pred_n:
                    r = 1.0
                elif pred_m < pred_n:
                    r = -1.0
                else:
                    r = - torch.sgn(gt_m - gt_n)
                
                if weight is not None:
                    r = r * (weight[m] + weight[n]) / 2

                loss = torch.max(torch.tensor(0.0, device=self.device), 
                                 self.bias - r * (pred_m - pred_n))
                
                mr_loss += loss

        mr_loss = mr_loss / (batch_size * batch_size)
        return mr_loss
    
    
class SIMLoss(torch.nn.Module):

    def __init__(self):
        super(SIMLoss, self).__init__()
    
    def forward(self, predictions, correct_output, weight=None):
        assert len(predictions) == len(correct_output)
        if weight is not None:
            assert len(weight) == len(correct_output)


        if weight is not None:
            # 余弦相似度中，添加样本权重
            predictions = predictions * torch.sqrt(weight)
            correct_output = correct_output * torch.sqrt(weight)
        
        # 计算 predictions 与 correct_output 的余弦相似度
        sim_loss = 1 - F.cosine_similarity(predictions.unsqueeze(0),
                                           correct_output.unsqueeze(0),
                                           dim=1)
        
        return sim_loss
    

class DocumentBertScoringLoss(torch.nn.Module):

    def __init__ (self, config, device, use_confidence=False):
        """
        config: dict {"loss_weights": {"alpha": float, "beta": float, "gamma": float},
                      "mr_bias": float} 
            分别为MSE, MarginRankingLoss, SIMLoss的权重
        """
        super(DocumentBertScoringLoss, self).__init__()
        self.weights = config["loss_weights"]
        self.use_confidence = use_confidence
        self.mr_loss = MarginRankingLoss(device=device, bias=config["mr_bias"])
        self.sim_loss = SIMLoss()
    
    def forward(self, predictions, correct_output, confidence=None):
        assert len(predictions) == len(correct_output)
        
        mse_loss = self.mse_loss(predictions, correct_output, confidence)
        mr_loss = self.mr_loss(predictions, correct_output, confidence)
        sim_loss = self.sim_loss(predictions, correct_output, confidence)
        
        loss = self.weights["alpha"] * mse_loss + \
               self.weights["beta"] * mr_loss + \
               self.weights["gamma"] * sim_loss
        
        return loss
    
    def mse_loss(self, predictions, correct_output, confidence):
        if self.use_confidence:
            mse = F.mse_loss(predictions, correct_output, reduction="none")
            mse = torch.mean(mse * confidence)
        else:
            mse = F.mse_loss(predictions, correct_output, reduction="mean")
        
        return mse

    

