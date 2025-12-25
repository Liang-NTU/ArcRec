import torch
import time
from torch.utils.data import DataLoader
from Data import Data
from model import ArcRec
from utils import test_util

def test_model(model, data, device, logger, epoch):
    hr_5, ndcg_5, hr_10, ndcg_10, hr_15, ndcg_15, auc = test_util(model, data, device, epoch)
    metrics = {
        "HR@5": hr_5, "NDCG@5": ndcg_5,
        "HR@10": hr_10, "NDCG@10": ndcg_10,
        "HR@15": hr_15, "NDCG@15": ndcg_15,
        "AUC": auc
    }
    logger.info(f"[Epoch {epoch}] Eval Results: {metrics}")
    return metrics

def train_model(args, device, logger):
    # === 加载数据 ===
    data = Data(args)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    user_num, item_num = data.get_user_item_dim()
    attr_adjs = [adj.to(device) for adj in data.attr_adjs]
    attr_num = len(attr_adjs)
    item_ids = torch.arange(item_num, device=device)
    attr_ids = torch.arange(attr_num, device=device)

    # === 模型 ===
    model = ArcRec(
        item_num=item_num,
        user_num=user_num,
        attr_num=attr_num,
        emb_dim=args.emb_dim,
        device=device,
        layer_num=args.layer_num,
        lamb=args.lamb,
        dataset=data,
        k=args.k,
        logger=logger,
        utility_mode=args.utility_mode,
        rpoint_source=args.rpoint_source,
        price_weight=args.price_weight
    ).to(device)
    model.set_paras(attr_adjs, item_ids, attr_ids)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 200], gamma=0.5)

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        total_loss = 0.0

        for batch in loader:
            user = batch['user'].to(device)
            pos_item = batch['pos_item'].to(device)
            neg_item = batch['neg_item'].to(device)
            anchor_items = batch['anchor_items'].to(device)
            pos_price_diff = batch['pos_price_diff'].to(device)
            neg_price_diff = batch['neg_price_diff'].to(device)
            anchor_prices = batch['anchor_prices'].to(device)
            att_mask = batch['att_mask'].to(device)
            mask = batch['mask'].to(device)
            batch_data = [
                user, pos_item, neg_item, anchor_items,
                pos_price_diff, neg_price_diff, anchor_prices, att_mask, mask
            ]

            optimizer.zero_grad()
            loss = model(batch_data)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()
        avg_loss = total_loss / len(data)
        logger.info(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Time: {time.time()-start:.2f}s")

    model.eval()
    metrics = test_model(model, data, device, logger, epoch=80)
    score = metrics["HR@10"]
    best_model_path = f"{args.model_dir}/best_model.pth"
    torch.save(model.state_dict(), best_model_path)
    logger.info(f"✅ Best model saved: {best_model_path} | HR@10={score:.4f}")