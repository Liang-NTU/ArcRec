import torch
import argparse
import os
import time
from train_utils import train_model
from logger_utils import create_logger, set_seed
from Data import Data

def parse_args():
    parser = argparse.ArgumentParser(description="Train ArcRec Model")

    # --- 基本参数 ---
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device, e.g., cuda:0 or cpu")
    parser.add_argument("--seed", type=int, default=3407, help="random seed")
    parser.add_argument("--filepath", type=str, default="raw_data", help="dataset dir")
    parser.add_argument("--threshold", type=int, default=1, help="threshold for filtering users/items")

    # --- 模型参数 ---
    parser.add_argument("--utility_mode", type=str, default="joint", help="utility from single or reference")
    parser.add_argument("--rpoint_source", type=str, default="user_item", help="reference points source")
    parser.add_argument("--price_weight", type=float, default=1.0, help="weight for price based utility")
    parser.add_argument("--emb_dim", type=int, default=64, help="embedding dimension")
    parser.add_argument("--layer_num", type=int, default=1, help="number of GNN layers")
    parser.add_argument("--lamb", type=float, default=0.1, help="weight for i-specific info")
    parser.add_argument("--k", type=int, default=3, help="number of augmented reference points")
    parser.add_argument("--refer_operation", type=str, default="cdot", help="embedding diff modeling")

    # --- 训练参数 ---
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=500, help="batch size")
    parser.add_argument("--epochs", type=int, default=80, help="number of epochs")
    parser.add_argument("--inter_epoch", type=int, default=10, help="interval for evaluation")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay for Adam")

    # --- 输出目录 ---
    parser.add_argument("--log_dir", type=str, default="./logs", help="directory to save logs")
    parser.add_argument("--model_dir", type=str, default="./model", help="directory to save models")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # === 初始化环境 ===
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    log_file = os.path.join(args.log_dir, f"{int(time.time())}_{args.utility_mode}_{args.rpoint_source}_{args.price_weight}_{args.seed}_{args.lamb}_referweight_dualattrgnn.log")
    logger = create_logger(log_file)
    logger.info(f"[INFO] Device: {device}, Seed: {args.seed}")
    logger.info(f"[INFO] Args: {args}")

    # === 训练模型 ===
    train_model(args, device, logger)