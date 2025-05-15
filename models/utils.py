import numpy as np
import torch
import dgl
import random
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, mean_absolute_error

MEAN_LABELS, STD_LABELS = 0, 0

def multi_label_auc(y_label, y_score):
    valid_n = y_score.shape[-1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_score.shape[-1]):
        f, t, _ = roc_curve(y_label[:, i], y_score[:, i])
        a = auc(f, t)

        if not np.isnan(a):
            fpr[i], tpr[i] = f, t
            roc_auc[i] = a
        else:
            valid_n -= 1
            
    if valid_n == 0:
        return -1
    else:
        return sum(roc_auc.values()) / valid_n

def AUC(tesYAll, tesPredictAll):
    tesAUC = multi_label_auc(tesYAll, tesPredictAll)
    return tesAUC

def MAE(tesYAll, tesPredictAll):
    tesYAll = zscore_denormalize(tesYAll)
    tesPredictAll = zscore_denormalize(tesPredictAll)
    mae = mean_absolute_error(tesYAll, tesPredictAll)
    return mae

def append_matrix_to_file(matrix, file_path):
    with open(file_path, 'a') as f:
        np.savetxt(f, matrix, fmt='%s')
        f.write("\n")
        print('Matrix has been saved')

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    
def zscore_normalize(x):
    global MEAN_LABELS, STD_LABELS
    mean_vals = x.mean(dim=0, keepdim=True)
    std_vals = x.std(dim=0, keepdim=True)

    x_norm = (x - mean_vals) / (std_vals + 1e-8)

    MEAN_LABELS = mean_vals.cpu().numpy()
    STD_LABELS = std_vals.cpu().numpy()

    return x_norm

def zscore_denormalize(x_norm):
    mean_vals, std_vals = MEAN_LABELS, STD_LABELS
    x = x_norm * (std_vals + 1e-8) + mean_vals
    return x


class GraphDataLoader_Classification(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self.collate_fn
        super(GraphDataLoader_Classification, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        batched_gs = dgl.batch([item[1] for item in batch])
        batched_ys = torch.stack([item[2] for item in batch])
        batched_ms = torch.stack([item[3] for item in batch])
        return (batched_gs, batched_ys, batched_ms)


class GraphDataLoader_Regression(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self.collate_fn
        super(GraphDataLoader_Regression, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        batched_gs = dgl.batch([item[1] for item in batch])
        batched_ys = torch.stack([item[2] for item in batch])
        return (batched_gs, batched_ys)