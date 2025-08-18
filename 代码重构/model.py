from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import save_emb


class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1)
            )
        else:
            # 降级到标准注意力机制
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)

        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        # 最终的线性变换
        output = self.out_linear(attn_output)

        return output, None


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs


class BaselineModel(torch.nn.Module):
    def __init__(self, user_num, item_num, user_feature_dim, item_feature_dim, args):
        super(BaselineModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        # DNN 层的输入维度现在是确定的
        self.userdnn = torch.nn.Linear(user_feature_dim, args.hidden_units)
        self.itemdnn = torch.nn.Linear(item_feature_dim, args.hidden_units)

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(FlashMultiHeadAttention(args.hidden_units, args.num_heads, args.dropout_rate))
            self.forward_layernorms.append(torch.nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))

    def log2feats(self, log_seqs, log_item_emb, log_user_emb, mask):
        # 1. 融合特征 embedding
        item_emb_fused = torch.relu(self.itemdnn(log_item_emb))
        user_emb_fused = torch.relu(self.userdnn(log_user_emb))
        
        seqs = item_emb_fused + user_emb_fused

        # 2. 添加位置编码
        seqs *= self.itemdnn.out_features ** 0.5
        
        batch_size, maxlen = log_seqs.shape
        poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1)
        poss = poss * (log_seqs != 0)
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        # 3. Transformer 模块
        attention_mask_pad = (mask != 0)
        attention_mask = torch.tril(torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev))
        attention_mask = attention_mask.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        for i in range(len(self.attention_layers)):
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, batch):
        log_seqs = batch['log_seqs']
        log_item_emb = batch['log_item_emb']
        log_user_emb = batch['log_user_emb']
        pos_embs_raw = batch['pos_embs']
        neg_embs_raw = batch['neg_embs']
        mask = batch['mask']
        next_mask = batch['next_mask']

        log_feats = self.log2feats(log_seqs, log_item_emb, log_user_emb, mask)
        
        pos_embs = torch.relu(self.itemdnn(pos_embs_raw))
        neg_embs = torch.relu(self.itemdnn(neg_embs_raw))

        loss_mask = (next_mask == 1)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        
        pos_logits = pos_logits * loss_mask
        neg_logits = neg_logits * loss_mask

        return pos_logits, neg_logits

    def predict(self, batch):
        """
        计算用户序列的最终表征，用于推理。

        Args:
            batch: 一个字典，包含预处理好的数据。
                需要包含:
                - "log_seqs": 原始序列ID, [B, L]
                - "log_item_emb": 物品特征的拼接 Embedding, [B, L, item_feat_dim]
                - "log_user_emb": 用户特征的拼接 Embedding, [B, L, user_feat_dim]
                - "mask": token 类型掩码, [B, L]

        Returns:
            final_feat: 用户序列的最终表征，取最后一个时间步的输出。形状为 [batch_size, hidden_units]
        """
        # 1. 从 batch 解包
        log_seqs = batch['log_seqs']
        log_item_emb = batch['log_item_emb']
        log_user_emb = batch['log_user_emb']
        mask = batch['mask']

        # 2. 调用 log2feats 获取序列中每个时间步的表征
        log_feats = self.log2feats(log_seqs, log_item_emb, log_user_emb, mask)

        # 3. 提取最后一个时间步的表征作为用户的最终表征
        #    注意：这里假设序列是左 padding 的，所以最后一个有效 token 在 `[:, -1, :]`
        #    如果序列中有 padding，需要找到每个序列的最后一个非 padding token 的位置来提取。
        #    一个更健壮的方法是：
        row_indices = torch.arange(log_seqs.size(0), device=self.dev)
        last_indices = (mask != 0).sum(dim=1) - 1 # 计算每个序列的最后一个有效 token 的索引
        final_feat = log_feats[row_indices, last_indices, :]

        return final_feat


    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """
        生成候选库item embedding，用于检索

        Args:
            item_ids: 候选item ID（re-id形式）
            retrieval_ids: 候选item ID（检索ID，从0开始编号，检索脚本使用）
            feat_dict: 训练集所有item特征字典，key为特征ID，value为特征值
            save_path: 保存路径
            batch_size: 批次大小
        """
        all_embs = []

        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))

            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])

            batch_feat = np.array(batch_feat, dtype=object)

            batch_emb = self.feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0)

            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))

# 这是一个独立的辅助函数，不再是 BaselineModel 的一部分

def save_item_emb(item_ids, retrieval_ids, feat_dict, save_path, embedding_modules, item_dnn, batch_size=1024):
    """
    生成候选库 item embedding，用于检索。
    这个函数现在是独立的，并接收所有必要的模块。

    Args:
        item_ids: 候选 item ID（re-id 形式）。
        retrieval_ids: 候选 item ID（检索 ID）。
        feat_dict: 所有 item 的特征字典，key 为 item re-id。
        save_path: 保存路径。
        embedding_modules: 一个包含所有 embedding 层的字典 (item_emb, sparse_emb, etc.)。
        item_dnn: 预训练好的 item dnn 层 (从模型中获取 model.itemdnn)。
        batch_size: 批次大小。
    """
    # 确保所有模块都在评估模式
    item_dnn.eval()
    for module in embedding_modules.values():
        if isinstance(module, torch.nn.Module):
            module.eval()

    all_embs = []
    
    # 这里的逻辑和我们重构的 collate_fn 非常相似
    # 我们可以复用 collate_fn 中的特征处理逻辑
    # 为了简化，我们在这里重新实现一遍，专门针对 item
    
    # 从 embedding_modules 中解包
    device = embedding_modules['device']
    item_emb = embedding_modules['item_emb'].to(device)
    sparse_emb = embedding_modules['sparse_emb'].to(device)
    emb_transform = embedding_modules['emb_transform'].to(device)
    feat_types = embedding_modules['feat_types']
    item_emb_feat_shapes = embedding_modules['item_emb_feat_shapes']
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))
            current_batch_size = end_idx - start_idx

            # 1. 准备 ID 和特征
            batch_ids = torch.tensor(item_ids[start_idx:end_idx], device=device)
            # 注意：这里的数据结构是 [batch_size] 的特征字典列表
            batch_feat_list = [feat_dict[i] for i in item_ids[start_idx:end_idx]]

            # 2. 计算 ID Embedding
            # 形状: [current_batch_size, hidden_units]
            id_embedding = item_emb(batch_ids)
            item_feat_tensors = [id_embedding]

            # 3. 处理其他特征
            # 辅助函数，将特征列表转换为 padded tensor
            def feature_to_tensor_flat(feat_list, k, is_array=False, is_emb=False, emb_dim=0):
                # 这个版本比 collate_fn 的简单，因为没有序列长度维度
                if is_array:
                    max_len = max(len(item[k]) for item in feat_list if item[k] is not None)
                    data = np.zeros((len(feat_list), max_len), dtype=np.int64)
                    for i, item in enumerate(feat_list):
                        if item[k] is not None:
                            actual_len = min(len(item[k]), max_len)
                            data[i, :actual_len] = item[k][:actual_len]
                    return torch.from_numpy(data).to(device)
                elif is_emb:
                    data = np.zeros((len(feat_list), emb_dim), dtype=np.float32)
                    for i, item in enumerate(feat_list):
                        if k in item and item[k] is not None:
                            data[i] = item[k]
                    return torch.from_numpy(data).to(device)
                else: # Sparse or Continual
                    dtype = np.float32 if k in feat_types.get('item_continual', []) else np.int64
                    data = np.array([item[k] for item in feat_list], dtype=dtype)
                    tensor = torch.from_numpy(data).to(device)
                    return tensor if dtype == np.int64 else tensor.unsqueeze(1)

            # 循环处理所有 item 特征
            feature_map = {
                'item_sparse': (feat_types['item_sparse'], False, False),
                'item_array': (feat_types['item_array'], True, False),
                'item_continual': (feat_types['item_continual'], False, False),
                'item_emb': (feat_types['item_emb'], False, True),
            }
            for ftype, (feat_ids, is_arr, is_emb) in feature_map.items():
                for k in feat_ids:
                    tensor_feat = feature_to_tensor_flat(batch_feat_list, k, is_array=is_arr, is_emb=is_emb, emb_dim=item_emb_feat_shapes.get(k, 0))
                    
                    if ftype.endswith('sparse'):
                        item_feat_tensors.append(sparse_emb[k](tensor_feat))
                    elif ftype.endswith('array'):
                        item_feat_tensors.append(sparse_emb[k](tensor_feat).sum(1))
                    elif ftype.endswith('continual'):
                        item_feat_tensors.append(tensor_feat)
                    elif ftype.endswith('emb'):
                        item_feat_tensors.append(emb_transform[k](tensor_feat))

            # 4. 拼接所有特征的 embedding
            # 形状: [current_batch_size, item_feature_dim]
            concatenated_embs = torch.cat(item_feat_tensors, dim=1)

            # 5. 通过 item_dnn 得到最终的 embedding
            # 形状: [current_batch_size, hidden_units]
            final_batch_emb = torch.relu(item_dnn(concatenated_embs))
            
            all_embs.append(final_batch_emb.cpu().numpy().astype(np.float32))

    # 合并所有批次的结果并保存
    final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
    final_embs = np.concatenate(all_embs, axis=0)
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    save_emb(final_embs, save_path / 'embedding.fbin')
    save_emb(final_ids, save_path / 'id.u64bin')
