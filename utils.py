import numpy as np
import random
import math
import torch
from torch_scatter import scatter_add
import pickle
import logging

def gumbel_topk(sim, K, tau=0.1):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(sim)))
    y = sim + gumbel_noise
    weights = torch.softmax(y / tau, dim=0)
    return weights

# ==========================================================
# =============== 图处理函数 ===============================
# ==========================================================
def add_self_loop(edge_index: torch.Tensor, edge_weight: torch.Tensor, item_size: int):
    """为稀疏邻接矩阵添加自环"""
    loop_index = torch.arange(item_size, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    loop_weight = edge_weight.new_full((item_size,), 1.0)
    edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index, edge_weight

def normalize(adj: torch.sparse.FloatTensor):
    """对稀疏图邻接矩阵做对称归一化"""
    edge_index = adj._indices()
    edge_weight = adj._values()
    item_size = adj.size(0)
    edge_index, edge_weight = add_self_loop(edge_index, edge_weight, item_size)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=item_size)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, edge_weight

# ==========================================================
# =============== 评估指标计算 ==============================
# ==========================================================
def HR(recList, selectList, L):
    count = 0
    for item_id in selectList:
        item_id = int(item_id)
        rank = recList.index(item_id)
        if rank < L:
            count += 1
    return float(count)

def ndcg(recList, selectList, L):
    resultList = []
    flag = 0
    for i in range(L):
        item = str(recList[i])
        if item in selectList or int(item) in selectList:
            resultList.append(1)
            flag = 1
        else:
            resultList.append(0)
    if flag == 0:
        return 0.0
    dcg = getDCG(resultList)
    idcg = getIdcg(resultList)
    return dcg / idcg

def getDCG(resultList):
    dcg = resultList[0]
    i = 3
    for val in resultList[1:]:
        dcg += (pow(2, val) - 1) / math.log(i, 2)
        i += 1
    return dcg

def getIdcg(resultList):
    resultList.sort()
    resultList.reverse()
    return getDCG(resultList)

def getAUC(score_dict, selectList):
    pos_num = 0
    for item in selectList:
        item_score = score_dict[item]
        for c_item in score_dict:
            c_item_score = score_dict[c_item]
            if c_item in selectList:
                continue
            if item_score > c_item_score:
                pos_num += 1
    total_num = (len(score_dict)-len(selectList)) * len(selectList)
    return pos_num/total_num

# ==========================================================
# =============== 测试批次生成器 ============================
# ==========================================================
def get_test_batch(dataset):
    """
    根据训练集与测试集构造测试样本 (user, anchor_items, candidate_items ...)
    """
    itemset = dataset.itemset

    for user in dataset.testing:
        anchor_items = dataset.training[user]
        candidate_items = list(itemset - set(anchor_items))
        true_items = dataset.testing[user]

        # 价格差、mask矩阵
        price_diff, anchor_prices, att_mask, mask = [], [], [], []

        for c_item in candidate_items:
            diffs = []
            for a_item in anchor_items:
                c_price = dataset.item_meta[c_item]["price"]
                a_price = dataset.item_meta[a_item]["price"]
                diffs.append(a_price - c_price)
            price_diff.append(diffs)

        for a_item in anchor_items:
            anchor_prices.append(dataset.item_meta[a_item]["price"])

        anchor_len = len(anchor_items)
        for _ in candidate_items:
            att_mask.append([0.0] * anchor_len)
            mask.append([1.0 / anchor_len] * anchor_len)

        yield user, anchor_items, candidate_items, true_items, price_diff, anchor_prices, att_mask, mask

# ==========================================================
# =============== 模型评估主函数 ============================
# ==========================================================
def test_util(model, data, device, epoch, logger=None):
    """测试模型在测试集上的 HR/NDCG/AUC 表现"""
    metrics_sum = {"HR@5": 0, "NDCG@5": 0,
                   "HR@10": 0, "NDCG@10": 0,
                   "HR@15": 0, "NDCG@15": 0,
                   "AUC": 0}
    count = 0
    user2att = {}

    for user, anchors, candidates, true_items, price_diff, anchor_prices, att_mask, mask in get_test_batch(data):
        count += 1
        num_candidates = len(candidates)

        # ===== 构造输入 =====
        user_tensor = torch.LongTensor([user]).to(device).repeat(num_candidates, 1)
        anchors = torch.LongTensor(anchors).to(device).unsqueeze(0).repeat(num_candidates, 1)
        candidates = torch.LongTensor(candidates).to(device).unsqueeze(1)
        price_diff = torch.FloatTensor(price_diff).to(device)
        anchor_prices = torch.FloatTensor(anchor_prices).to(device).unsqueeze(0).repeat(num_candidates, 1)
        att_mask = torch.FloatTensor(att_mask).to(device)
        mask = torch.FloatTensor(mask).to(device)

        # ===== 模型推理 =====
        rec_score, att_vals = model.test_score(user_tensor, anchors, candidates, price_diff, anchor_prices, att_mask, mask)
        user2att[user] = att_vals[0].detach().cpu().numpy()

        scores = rec_score.detach().cpu().numpy().tolist()
        result = {int(candidates[i]): scores[i] for i in range(num_candidates)}
        auc = getAUC(result, true_items)

        # ===== 排序并计算指标 =====
        top_sorted = [k for k, _ in sorted(result.items(), key=lambda x: x[1], reverse=True)]

        metrics_sum["HR@5"] += HR(top_sorted, true_items, 5)
        metrics_sum["NDCG@5"] += ndcg(top_sorted, true_items, 5)
        metrics_sum["HR@10"] += HR(top_sorted, true_items, 10)
        metrics_sum["NDCG@10"] += ndcg(top_sorted, true_items, 10)
        metrics_sum["HR@15"] += HR(top_sorted, true_items, 15)
        metrics_sum["NDCG@15"] += ndcg(top_sorted, true_items, 15)
        metrics_sum["AUC"] += auc

    # ===== 结果平均 =====
    metrics_avg = {k: v / count for k, v in metrics_sum.items()}

    msg = f"[Epoch {epoch}] " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics_avg.items()])
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return (
        metrics_avg["HR@5"],
        metrics_avg["NDCG@5"],
        metrics_avg["HR@10"],
        metrics_avg["NDCG@10"],
        metrics_avg["HR@15"],
        metrics_avg["NDCG@15"],
        metrics_avg["AUC"]
    )