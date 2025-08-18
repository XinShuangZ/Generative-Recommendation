import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    
    ############################################### START #########################################################
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types
    # 1. 创建基础 Embedding 表
    item_emb = torch.nn.Embedding(itemnum + 1, args.hidden_units, padding_idx=0)
    user_emb = torch.nn.Embedding(usernum + 1, args.hidden_units, padding_idx=0)
    # 2. 创建稀疏和数组特征的 Embedding 表
    sparse_emb = torch.nn.ModuleDict()
    def initialize_sparse_embeddings(feat_ids, feat_stats, emb_dim):
        embeddings = {}
        for k in feat_ids:
            embeddings[k] = torch.nn.Embedding(feat_stats[k] + 1, emb_dim, padding_idx=0)
        return embeddings
    # 从 feat_types 获取所有需要创建 embedding 的特征
    all_sparse_feat_ids = feat_types['user_sparse'] + feat_types['item_sparse']
    all_array_feat_ids = feat_types['user_array'] + feat_types['item_array']
    sparse_emb.update(initialize_sparse_embeddings(all_sparse_feat_ids, feat_statistics, args.hidden_units))
    sparse_emb.update(initialize_sparse_embeddings(all_array_feat_ids, feat_statistics, args.hidden_units))
    # 3. 创建多模态特征的线性变换层
    emb_transform = torch.nn.ModuleDict()
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    for k in feat_types['item_emb']:
        emb_transform[k] = torch.nn.Linear(EMB_SHAPE_DICT[k], args.hidden_units)
    # 4. 将所有需要的模块和参数打包，传递给 dataset
    #    注意：我们将 embedding 模块移到 GPU 的操作放在 collate_fn 内部，
    #    这样可以避免在多进程（num_workers > 0）时出现 CUDA 初始化问题。
    #    主进程中创建的模块仍在 CPU 上。
    embedding_modules = {
        'item_emb': item_emb,
        'user_emb': user_emb,
        'sparse_emb': sparse_emb,
        'emb_transform': emb_transform,
        'device': args.device,
        'feat_types': feat_types,
        'item_emb_feat_shapes': {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}
    }
    # 使用一个新方法来设置这些模块
    dataset.set_embedding_modules(embedding_modules)
    # 如果是 random_split 后的子集，也需要设置
    train_dataset.dataset.set_embedding_modules(embedding_modules)
    valid_dataset.dataset.set_embedding_modules(embedding_modules)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    user_feat_dim = args.hidden_units * (len(feat_types['user_sparse']) + 1 + len(feat_types['user_array'])) + len(feat_types['user_continual'])
    item_feat_dim = args.hidden_units * (len(feat_types['item_sparse']) + 1 + len(feat_types['item_array']) + len(feat_types['item_emb'])) + len(feat_types['item_continual'])
    
    model = BaselineModel(usernum, itemnum, user_feat_dim, item_feat_dim, args).to(args.device)

    ############################################### END ############################################################
    
    # 1. 初始化模型本身的参数 (除了 Embedding 层)
    for name, param in model.named_parameters():
        # model 内部现在只有 pos_emb, dnn, attention, layernorm 等层的参数
        if 'pos_emb' not in name: # pos_emb 单独处理
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass # 有些层可能不支持 xavier_normal_

    # 2. 初始化在模型外部创建的 Embedding 层
    #    这些模块我们之前定义为 item_emb, user_emb, sparse_emb
    #    它们现在是独立的 torch.nn.Module 对象

    # a. 初始化 pos_emb (它仍在模型内部)
    model.pos_emb.weight.data[0, :] = 0

    # b. 初始化 item_emb 和 user_emb (在模型外部)
    with torch.no_grad():
        torch.nn.init.xavier_normal_(item_emb.weight.data)
        torch.nn.init.xavier_normal_(user_emb.weight.data)
        item_emb.weight.data[0, :] = 0
        user_emb.weight.data[0, :] = 0

    # c. 初始化 sparse_emb (在模型外部)
    with torch.no_grad():
        for k in sparse_emb:
            torch.nn.init.xavier_normal_(sparse_emb[k].weight.data)
            sparse_emb[k].weight.data[0, :] = 0

    # =======================================================

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        # 加载模型状态字典时，也需要区分模型内部和外部的参数
        # 一个简单的方法是分别保存和加载
        # 假设你保存时将所有参数打包了
        # state_dict = torch.load(args.state_dict_path, map_location=torch.device(args.device))
        # model.load_state_dict(state_dict['model'])
        # item_emb.load_state_dict(state_dict['item_emb'])
        # user_emb.load_state_dict(state_dict['user_emb'])
        # sparse_emb.load_state_dict(state_dict['sparse_emb'])
        # emb_transform.load_state_dict(state_dict['emb_transform'])
        # 这种方式更健壮，但需要修改保存逻辑。

        # 如果只加载模型本身的参数，可以这样做：
        try:
            # 假设 state_dict 只包含模型内部的参数
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            # 注意：这种方式不会加载预训练的 embedding，如果需要，必须分开加载
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    # =================== Optimizer 创建修改 ===================
    # 优化器需要包含所有需要训练的参数，包括模型外部的 Embedding 层
    all_params = list(model.parameters()) + \
                list(item_emb.parameters()) + \
                list(user_emb.parameters()) + \
                list(sparse_emb.parameters()) + \
                list(emb_transform.parameters())

    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(all_params, lr=args.lr, betas=(0.9, 0.98))
    # ===========================================================

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        # 将所有模块设置为训练模式
        model.train()
        item_emb.train()
        user_emb.train()
        sparse_emb.train()
        emb_transform.train()

        if args.inference_only:
            break
            
        # =================== 训练循环修改 ===================
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # 之前的方式:
            # seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            # seq = seq.to(args.device)
            # ...

            # 新的方式：batch 已经是一个字典，并且里面的 Tensor 已经在 collate_fn 中被放到了正确的 device 上
            
            # 模型调用方式改变
            pos_logits, neg_logits = model(batch)

            # 标签和损失计算逻辑基本不变
            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)
            
            optimizer.zero_grad()
            
            # 从 batch 中获取 next_mask (之前叫 next_token_type)
            indices = torch.where(batch['next_mask'] == 1)
            
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1

            # L2 正则化修改：直接对外部的 embedding 层操作
            for param in item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            # 如果需要对其他 embedding 也做正则化，可以添加
            # for param in user_emb.parameters():
            #     loss += args.l2_emb * torch.norm(param)
            # for module in sparse_emb.values():
            #     for param in module.parameters():
            #         loss += args.l2_emb * torch.norm(param)
            
            loss.backward()
            optimizer.step()
        # =======================================================

        # 将所有模块设置为评估模式
        model.eval()
        item_emb.eval()
        user_emb.eval()
        sparse_emb.eval()
        emb_transform.eval()
        
        valid_loss_sum = 0
        with torch.no_grad(): # 在评估时使用 no_grad 是个好习惯
            # =================== 验证循环修改 ===================
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                # 同样，batch 是一个字典
                pos_logits, neg_logits = model(batch)
                
                pos_labels = torch.ones(pos_logits.shape, device=args.device)
                neg_labels = torch.zeros(neg_logits.shape, device=args.device)
                
                indices = torch.where(batch['next_mask'] == 1)
                
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                valid_loss_sum += loss.item()
            # =======================================================
            
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
