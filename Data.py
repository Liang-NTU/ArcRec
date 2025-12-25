from torch.utils.data import Dataset
import numpy as np
import sys
import random
import torch
import linecache
import pickle
import time
from configparser import ConfigParser
import scipy.sparse as sp

import itertools
from scipy.sparse import coo_matrix
from utils import *
from torch.utils.data import DataLoader

class Data(Dataset):
    def __init__(self, args):
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.threshold = args.threshold

        data_dir = args.filepath
        dataset_file = open(data_dir + "/dataset.pkl", 'rb')
        training, testing, coding_user_meta, coding_item_meta = pickle.load(dataset_file)

        self.user_meta, self.item_meta = coding_user_meta, coding_item_meta
        self.training, self.testing = training, testing
        self.userset, self.itemset = set(), set()
        self.dataset = []
        self.max_len = 0
        self.norm_item_price()

        for user in self.training:
            self.userset.add(user)
            for item in self.training[user]:
                self.itemset.add(item)
                self.dataset.append([user,item])
            self.max_len = max(self.max_len, len(self.training[user])-1)
        self.user_size = len(self.userset)
        self.item_size = len(self.itemset)
        self.datalen = len(self.dataset)
        # self.max_len = 10

        print("max of length", self.max_len)
        print("num of user", self.user_size)
        print("num of item", self.item_size)
        print("num of interactions", self.datalen, "density", self.datalen/(self.user_size*self.item_size))

        co_adj = self.build_co_graph()
        attr_adjs = self.build_attr_graph(co_adj)
        self.attr_adjs = attr_adjs

    def norm_item_price(self):
        prices = []
        for item in self.item_meta:
            item_price = self.item_meta[item]["price"]
            if item_price < 10:
                item_price = 10
            if item_price > 800:
                item_price = 800
            prices.append(item_price)

        p_max = max(prices)
        p_min = min(prices)
        for item in self.item_meta:
            item_price = self.item_meta[item]["price"]
            if item_price < 10:
                item_price = 10
            if item_price > 800:
                item_price = 800

            norm_item_price = (item_price - p_min) / (p_max - p_min)
            self.item_meta[item]["price"] = norm_item_price

    def build_co_graph(self):
        co_num_dict = {}
        for user in self.training:
            user_training = self.training[user]
            for i, itemi in enumerate(user_training):
                for j, itemj in enumerate(user_training[i+1:]):
                    key = str(min(itemi, itemj)) + "_" + str(max(itemi, itemj))
                    co_num_dict[key] = co_num_dict.get(key, 0) + 1
        row, col, values = [], [], []
        for key in co_num_dict:
            num = co_num_dict[key]
            itemi, itemj = key.split("_")
            itemi, itemj = int(itemi), int(itemj)
            if num >= self.threshold:
                row += [itemi, itemj]
                col += [itemj, itemi]
                values += [1.0, 1.0]
        print("co-purchase", "before normalize:", len(row))
        co_adj = [row, col, values]
        return co_adj

    def build_attr_graph(self, co_adj):
        row, col, values = co_adj

        attrs = ["品牌", "品类", "容量"]
        attr_adjs = []
        for attr in attrs:
            attr_adj = self.graph_generation(row, col, attr)
            attr_adjs.append(attr_adj)
        return attr_adjs

    def graph_generation(self, row, col, attr):
        attr_row, attr_col, attr_values = [], [], []

        for _i in range(len(row)):
            item_row, item_col = row[_i], col[_i]
            item_row_feat = self.item_meta[item_row][attr]
            item_col_feat = self.item_meta[item_col][attr]
            if item_row_feat == item_col_feat:
                attr_row.append(item_row)
                attr_col.append(item_col)
                attr_values.append(1.0)
        print("attr name", attr, "before normalize:", len(attr_row))

        attr_adj = torch.sparse_coo_tensor([attr_row, attr_col], attr_values, (self.item_size, self.item_size))
        edge_index, edge_weight = normalize(attr_adj)
        print("attr name", attr, 'after normalize:', len(edge_weight))
        attr_adj = torch.sparse.FloatTensor(edge_index, edge_weight, attr_adj.shape)

        print("symmetric or not", attr_adj.to_dense().equal(attr_adj.to_dense().T))
        return attr_adj

    def get_user_item_dim(self):
        return self.user_size, self.item_size

    def __len__(self):
        return self.datalen

    def __getitem__(self, idx):
        user, pos_item = self.dataset[idx]
        anchor_items = self.training[user]
        before_len = len(anchor_items)

        anchor_items = [itemID for itemID in anchor_items if itemID != pos_item]
        # pos_position = anchor_items.index(pos_item)
        # anchor_items = anchor_items[:pos_position] + anchor_items[pos_position+1:]
        assert len(anchor_items) == before_len-1

        neg_items = [0]
        while True:
            neg_items = np.random.choice(self.item_size, 1)
            if neg_items[0] not in anchor_items:
                break
        neg_item = int(neg_items[0])

        # price diff --> defined as refer-pos
        pos_price_diff = []
        for anchor_item in anchor_items:
            pos_price = self.item_meta[pos_item]["price"]
            anchor_price = self.item_meta[anchor_item]["price"]
            price_diff = anchor_price - pos_price
            pos_price_diff.append(price_diff)

        neg_price_diff = []
        for anchor_item in anchor_items:
            neg_price = self.item_meta[neg_item]["price"]
            anchor_price = self.item_meta[anchor_item]["price"]
            price_diff = anchor_price - neg_price
            neg_price_diff.append(price_diff)

        anchor_prices = []
        for anchor_item in anchor_items:
            anchor_prices.append(self.item_meta[anchor_item]["price"])
            assert self.item_meta[anchor_item]["price"] >= 0

        # padding to max_len and mask
        anchor_len = len(anchor_items)
        if anchor_len > self.max_len:
            anchor_items = anchor_items[:self.max_len]
            pos_price_diff = pos_price_diff[:self.max_len]
            neg_price_diff = neg_price_diff[:self.max_len]
            anchor_prices = anchor_prices[:self.max_len]
        anchor_items = anchor_items + [0] * (self.max_len - anchor_len)
        pos_price_diff = pos_price_diff + [0.0] * (self.max_len - anchor_len)
        neg_price_diff = neg_price_diff + [0.0] * (self.max_len - anchor_len)
        anchor_prices = anchor_prices + [0.0] * (self.max_len - anchor_len)

        mask = [1.0 / anchor_len] * anchor_len + [0] * (self.max_len - anchor_len)
        # attention mask
        att_mask = [0.0] * anchor_len + [-1e13] * (self.max_len - anchor_len)

        sample = {
            'user': torch.LongTensor([user]),
            'pos_item': torch.LongTensor([pos_item]),
            'neg_item': torch.LongTensor([neg_item]),
            'anchor_items': torch.LongTensor(anchor_items),
            'pos_price_diff': torch.FloatTensor(pos_price_diff),
            'neg_price_diff': torch.FloatTensor(neg_price_diff),
            'anchor_prices': torch.FloatTensor(anchor_prices),
            'att_mask': torch.FloatTensor(att_mask),
            'mask': torch.FloatTensor(mask)
        }
        return sample