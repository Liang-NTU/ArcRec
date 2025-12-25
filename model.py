import numpy as np
import time
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD, Adam, ASGD, RMSprop
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax, softmax
import torch.nn.functional as F
from configparser import ConfigParser
import torch.nn.init as init

from Data import Data
from utils import *
import sys
import os
import time

# batch_size & initial is important
FType = torch.FloatTensor
LType = torch.LongTensor

class GnnLayer(nn.Module):
    def __init__(self):
        super(GnnLayer, self).__init__()

    def forward(self, x, norm_GG):
        x = torch.sparse.mm(norm_GG, x)
        return x

class ArcRec(torch.nn.Module):
    def __init__(self, item_num, user_num, attr_num, emb_dim, device, layer_num, lamb, dataset, k, logger=None,
                 utility_mode="joint", rpoint_source="user", price_weight=0.5):
        super(ArcRec, self).__init__()
        self.item_num = item_num
        self.user_num = user_num
        self.attr_num = attr_num
        self.emb_dim = emb_dim
        self.device = device
        self.layer_num = layer_num
        self.lamb = lamb
        self.logger = logger
        self.dataset = dataset
        self.k = k
        self.utility_mode = utility_mode
        self.rpoint_source = rpoint_source
        self.price_weight = price_weight

        self.item_embed = nn.Embedding(item_num, emb_dim*attr_num)
        self.ispe_item_embed = nn.Embedding(item_num, emb_dim*attr_num)

        layers = []
        for i in range(self.layer_num):
            layers.append(GnnLayer())
        self.layers = nn.ModuleList(layers)

        self.is_net = nn.Linear(emb_dim+1, 1, bias=False)

        self.alpha = nn.Linear(emb_dim, 1, bias=False)
        self.beta_pos = nn.Linear(emb_dim+1, 1, bias=False)
        self.beta_neg = nn.Linear(emb_dim+1, 1, bias=False)

        self.rp_net = nn.Linear(attr_num, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.item_embed.weight.data.uniform_(-1./np.sqrt(self.item_num), 1./np.sqrt(self.item_num))
        self.ispe_item_embed.weight.data.uniform_(-1./np.sqrt(self.item_num), 1./np.sqrt(self.item_num))
        init.xavier_uniform_(self.is_net.weight)
        init.xavier_uniform_(self.beta_pos.weight)
        init.xavier_uniform_(self.beta_neg.weight)
        init.xavier_uniform_(self.rp_net.weight)

    def set_paras(self, attr_adjs, item_ids, attr_ids):
        self.attr_adjs = attr_adjs
        self.item_ids = item_ids
        self.attr_ids = attr_ids

        self.item_price_tensor = torch.zeros(self.item_num, dtype=torch.float32, device=self.attr_ids.device)
        for item_id in self.dataset.item_meta:
            self.item_price_tensor[item_id] = self.dataset.item_meta[item_id]["price"]

    def conv(self, attr_item_emb, attr_adj):
        hiddens = [attr_item_emb]
        _input = attr_item_emb
        for layer in self.layers:
            _input = layer(_input, attr_adj)
            hiddens.append(_input)
        
        conv_emb = 1.0 * hiddens[0]
        norm = 1.0
        for _l, hidden in enumerate(hiddens):
            if _l == 0:
                continue
            conv_emb += 1/(_l+1) * hidden
            norm += 1/(_l+1)
        conv_emb = conv_emb / norm

        # without GNN (ablation for GNN layer)
        # conv_emb = attr_item_emb

        return conv_emb

    # embeddings - for collection // emb for instance
    def update_embed(self, item_embed_source):
        # item_embeddings = self.item_embed(self.item_ids)
        item_embeddings = item_embed_source(self.item_ids)

        attr_item_embeddings = torch.chunk(item_embeddings, self.attr_num, dim=1)

        updated_attr_embeddings  = []
        for _attr_id in range(self.attr_num):
            attr_adj = self.attr_adjs[_attr_id]
            attr_item_emb = attr_item_embeddings[_attr_id]

            updated_attr_emb = self.conv(attr_item_emb, attr_adj)
            updated_attr_embeddings.append(updated_attr_emb)
        return updated_attr_embeddings

    def augmented_refer_points(self, candidate_item, attr_item_emb):
        candidate_item_emb = attr_item_emb[candidate_item]
        candidate_item_emb = candidate_item_emb.squeeze(1)  # [B, d]
        similarity = torch.matmul(attr_item_emb, candidate_item_emb.T).T # [B, N]

        batch_indices = torch.arange(similarity.size(0), device=similarity.device)
        similarity[batch_indices, candidate_item.squeeze()] = -float("inf")

        topk_val, topk_idx = torch.topk(similarity, k=self.k, dim=1)
        topk_price = self.item_price_tensor[topk_idx]

        candidate_price = self.item_price_tensor[candidate_item]
        topk_price_diff = topk_price - candidate_price
        return topk_idx, topk_price_diff

    def get_item_info(self, candidate_item, updated_attr_embeddings):
        attr_embs = []
        for _attr_id in range(self.attr_num):
            attr_item_emb = updated_attr_embeddings[_attr_id]
            emb = attr_item_emb[candidate_item].squeeze() # B*64
            attr_embs.append(emb)

        mean_attr_emb = torch.stack(attr_embs, dim=1).mean(dim=1)
        price = self.item_price_tensor[candidate_item]
        combined_info = torch.cat([mean_attr_emb, price], dim=1)
        return combined_info

    def utility_modeling(self, updated_attr_embeddings, ru_items, candidate_item, ru_price_diff, mask, awtp_att_vals):
        # i-specific B*(d+1)
        updated_ispe_embeddings = self.update_embed(self.ispe_item_embed)
        updated_ispe_embeddings = torch.stack(updated_ispe_embeddings, dim=1)
        updated_ispe_embeddings = updated_ispe_embeddings.reshape(updated_ispe_embeddings.size(0), -1)
        ispe_item_emb = updated_ispe_embeddings[candidate_item].squeeze()
        anchor_items_emb = updated_ispe_embeddings[ru_items].squeeze()  # N*L*E
        ispe_mask = mask.unsqueeze(1)  # N*1*L
        user_emb = ispe_mask.matmul(anchor_items_emb).squeeze()
        is_score = user_emb.mul(ispe_item_emb).sum(dim=1).view(-1, 1)

        # reference-specific
        refer_scores = []
        for _attr_id in range(self.attr_num):
            attr_item_emb = updated_attr_embeddings[_attr_id]
            ri_items, ri_price_diff = self.augmented_refer_points(candidate_item, attr_item_emb) # Ri
            ru_items, ru_price_diff, mask = ru_items, ru_price_diff, mask # Ru

            # model pair part
            # Ru set
            ru_item_emb = attr_item_emb[ru_items]
            batch_size, max_len, emb_dim = ru_item_emb.size()
            candidate_item_emb = attr_item_emb[candidate_item]
            candidate_item_emb = candidate_item_emb.repeat(1, max_len, 1)

            # ru_alpha_input = ru_item_emb - candidate_item_emb
            ru_alpha_input = ru_item_emb * candidate_item_emb
            ru_prefer_score = self.alpha(ru_alpha_input).squeeze()

            ru_attr_awtp = awtp_att_vals[:, _attr_id]
            ru_attr_awtp = ru_attr_awtp.view(-1, 1, 1).repeat(1, max_len, 1)
            ru_beta_input = torch.cat([ru_alpha_input, ru_attr_awtp], dim=2)
            ru_beta_pos_out = self.beta_pos(ru_beta_input).squeeze(-1)  # [B, K]
            ru_beta_neg_out = self.beta_neg(ru_beta_input).squeeze(-1)  # [B, K]
            ru_mask_pos = (ru_price_diff > 0).float()  # [B, K], 1 for pos, 0 for neg
            ru_mask_neg = 1.0 - ru_mask_pos
            ru_price_beta = ru_beta_pos_out * ru_mask_pos + ru_beta_neg_out * ru_mask_neg  # [B, K]
            ru_price_score = ru_price_beta * ru_price_diff
            ru_score = ru_prefer_score + self.price_weight * ru_price_score

            ru_sum = torch.sum(ru_score * mask, dim=1)  # [B]
            ru_count = torch.sum(mask, dim=1)  # [B]

            # Ri set
            ri_item_emb = attr_item_emb[ri_items]
            batch_size, max_len, emb_dim = ri_item_emb.size()
            candidate_item_emb = attr_item_emb[candidate_item]
            candidate_item_emb = candidate_item_emb.repeat(1, max_len, 1)

            # ri_alpha_input = ri_item_emb - candidate_item_emb
            ri_alpha_input = ri_item_emb * candidate_item_emb
            ri_prefer_score = self.alpha(ri_alpha_input).squeeze()

            ri_attr_awtp = awtp_att_vals[:, _attr_id]
            ri_attr_awtp = ri_attr_awtp.view(-1, 1, 1).repeat(1, max_len, 1)
            ri_beta_input = torch.cat([ri_alpha_input, ri_attr_awtp], dim=2)
            ri_beta_pos_out = self.beta_pos(ri_beta_input).squeeze(-1)  # [B, K]
            ri_beta_neg_out = self.beta_neg(ri_beta_input).squeeze(-1)  # [B, K]
            ri_mask_pos = (ri_price_diff > 0).float()  # [B, K], 1 for pos, 0 for neg
            ri_mask_neg = 1.0 - ri_mask_pos
            ri_price_beta = ri_beta_pos_out * ri_mask_pos + ri_beta_neg_out * ri_mask_neg  # [B, K]
            ri_price_score = ri_price_beta * ri_price_diff
            ri_score = ri_prefer_score + self.price_weight * ri_price_score

            ri_sum = torch.sum(ri_score, dim=1)  # [B]
            ri_count = torch.tensor(ri_score.size(1), device=ri_score.device, dtype=ru_count.dtype)  # 常数

            if self.rpoint_source == "user_item":
                # all score
                total_sum = ru_sum + ri_sum  # [B]
                total_count = ru_count + ri_count  # [B]
            if self.rpoint_source == "user":
                total_sum = ru_sum
                total_count = ru_count
            if self.rpoint_source == "item":
                total_sum = ri_sum  # [B]
                total_count = ri_count  # [B]

            total_refer_score = total_sum / total_count
            refer_scores.append(total_refer_score)

        refer_scores = torch.stack(refer_scores, dim=1)
        rp_score = self.rp_net(refer_scores)
        # important to test, using the relu function or not (scale the score)
        # rp_score = F.relu(rp_score)

        # fusion
        if self.utility_mode == "joint":
            # 0.1 refer weight setting for rp_score [1:0.1]
            utility_score = is_score + self.lamb * rp_score
        if self.utility_mode == "is":
            utility_score = is_score
        return utility_score

    # obtain user-attr attention based on price elasticity
    def awtp_attention(self, updated_attr_embeddings, anchor_items, anchor_prices, mask):
        awtp_att_vals = []
        for _attr_id in range(self.attr_num):
            attr_item_emb = updated_attr_embeddings[_attr_id]
            anchor_items_emb = attr_item_emb[anchor_items]
            batch_size, max_len, emb_dim = anchor_items_emb.size()

            avg_mask = mask.unsqueeze(1)
            avg_anchor_emb = avg_mask.matmul(anchor_items_emb) # b*1*E
            avg_prices = avg_mask.matmul(anchor_prices.unsqueeze(2)).squeeze(2) # b*1

            avg_anchor_emb = avg_anchor_emb.repeat(1, max_len, 1) # b*L*E
            demand_dis = torch.sqrt(torch.sum((anchor_items_emb-avg_anchor_emb)**2, dim=2)) # b*L
            demand_base = torch.norm(avg_anchor_emb, p=2, dim=2)
            demand_ratio = demand_dis/(demand_base + 1e-2)

            avg_prices = avg_prices.repeat(1, max_len) # b*L
            price_dis = torch.abs(anchor_prices-avg_prices) # b*L
            price_base = avg_prices
            price_ratio = price_dis/(price_base + 1e-2)

            awtp = price_ratio/(demand_ratio + 1e-2) # in case demand_ratio=0
            awtp_att = torch.sum(mask * awtp, dim=1) / torch.sum(mask, dim=1)
            awtp_att_vals.append(awtp_att)

        awtp_att_vals = torch.stack(awtp_att_vals, dim=1)
        awtp_att_vals = F.softmax(awtp_att_vals, dim=1)
        return awtp_att_vals

    def forward(self, batch_data):
        user, pos_item, neg_item, anchor_items, pos_price_diff, neg_price_diff, anchor_prices, att_mask, mask = batch_data

        # Reference
        updated_attr_embeddings = self.update_embed(self.item_embed)
        awtp_att_vals = self.awtp_attention(updated_attr_embeddings, anchor_items, anchor_prices, mask)

        # for pos item
        pos_utility_score = self.utility_modeling(updated_attr_embeddings, anchor_items, pos_item, pos_price_diff, mask, awtp_att_vals)
        # for neg item
        neg_utility_score = self.utility_modeling(updated_attr_embeddings, anchor_items, neg_item, pos_price_diff, mask, awtp_att_vals)

        loss_rec = torch.sum((neg_utility_score - pos_utility_score).sigmoid())
        return loss_rec

    def test_score(self, user, anchor_items, c_items, price_diff, anchor_prices, att_mask, mask):
        updated_attr_embeddings = self.update_embed(self.item_embed)
        awtp_att_vals = self.awtp_attention(updated_attr_embeddings, anchor_items, anchor_prices, mask)

        utility_score = self.utility_modeling(updated_attr_embeddings, anchor_items, c_items, price_diff, mask,
                                              awtp_att_vals)
        return utility_score, awtp_att_vals