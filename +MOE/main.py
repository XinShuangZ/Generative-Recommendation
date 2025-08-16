import argparse
import json
import os
import time
from pathlib import Path
import math

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from dataset import MyDataset
from model import BaselineModel

def get_args():
    parser = argparse.ArgumentParser()
    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--use_grad_clip', default=True, type=bool)
    parser.add_argument('--grad_clip', default=3, type=float, help='梯度剪裁的L2范数阈值')
    parser.add_argument('--maxlen', default=101, type=int)
    # user dnn
    parser.add_argument('--user_dnn_hidden_units', default=[], type=list)
    parser.add_argument('--item_dnn_hidden_units', default=[], type=str)
    # Baseline Model construction
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--num_epochs', default=6, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--mha_bias', default=True, type=bool)
    parser.add_argument('--ffn_bias', default=True, type=bool)
    parser.add_argument('--n_routed_experts', default=64, type=int)
    parser.add_argument('--n_shared_experts', default=2, type=int)
    parser.add_argument('--n_activated_experts', default=6, type=int)
    parser.add_argument('--n_expert_groups', default=1, type=int)
    parser.add_argument('--n_limited_groups', default=1, type=int)
    parser.add_argument('--score_func', default="sigmoid", type=str)
    
    parser.add_argument('--activation', 
                    default='relu', 
                    type=str, 
                    choices=['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'prelu'],
                    help='选择激活函数类型')
    # MoE
    parser.add_argument('--route_scale', default=1., type=float)
    parser.add_argument('--moe_inter_dim', default=64, type=int)

    parser.add_argument('--norm_method', default='rms', type=str, choices=['rms', 'layer'])

    # nce
    parser.add_argument('--temperature', default=0.07, type=float)
    
    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    # Warm up training parameters
    parser.add_argument('--warmup_steps', default=6300, type=int, help='Warm up steps')
    parser.add_argument('--warmup_lr', default=0.0001, type=float, help='Initial learning rate for warm up')

    parser.add_argument('--use_cos_similarity', default=True, type=bool, help='use cosine similarity for recall')

    #triple loss
    parser.add_argument('--Triple_Loss_lambda', default=0.3, type=float)
    parser.add_argument('--margin', default=0.6, type=float)
    
    args = parser.parse_args()

    return args


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """创建学习率调度器，包含warm up和余弦退火"""
    def lr_lambda(step):
        if step < warmup_steps:
            # Warm up阶段：线性增长
            return step / warmup_steps
        else:
            # 余弦退火阶段
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    # 计算总步数用于学习率调度
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)

    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            optimizer.zero_grad()
            
            '''前向传播'''
            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )

            '''主体损失（InfoNCE损失）'''
            loss = model.compute_infonce_loss(pos_logits, neg_logits)
            
            '''三元损失'''
            if args.Triple_Loss_lambda != 0:
                loss += model.compute_triple_loss(pos_logits, neg_logits)

            loss.backward()
            
            optimizer.step()
            scheduler.step()  # 更新学习率
                        
            # 记录学习率
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('Learning_Rate', current_lr, global_step)
            
            # <<< MODIFIED: 添加此行以记录训练损失到TensorBoard
            writer.add_scalar('Train/Loss', loss.item(), global_step)

            # 日志记录
            log_json = json.dumps(
                {'global_step': global_step, 
                'loss': loss.item(), 
                'lr': current_lr,
                'epoch': epoch, 
                'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            if global_step % 20 == 1:
                print(log_json)            
            global_step += 1
            

        '''验证集评估'''
        model.eval()
        valid_loss_sum = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)

                '''前向传播'''
                pos_logits, neg_logits = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                )
                '''主体损失（InfoNCE损失）'''
                loss = model.compute_infonce_loss(pos_logits, neg_logits)

                # 记录验证集指标
                valid_loss_sum += loss.item()

        
        # 计算平均损失
        valid_loss_avg = valid_loss_sum / len(valid_loader)
        writer.add_scalar('Valid/Loss', valid_loss_avg, global_step)        

        # 权重文件产出，路径和命名符合规范
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step={global_step}.valid_loss={valid_loss_avg:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
