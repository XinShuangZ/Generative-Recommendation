from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from dataset import save_emb

def my_activation(method='relu'): 
    activation_map = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'prelu': nn.PReLU()
    }
    assert method in activation_map, f"不支持的激活函数: {method}，可选值为: {list(activation_map.keys())}"
    return activation_map[method]

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim, args, use_layer_norm=True, dropout_rate=0.0):
        super(DNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(my_activation(args.activation))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate, bias=False):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate
        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"
        self.q_linear = torch.nn.Linear(hidden_units, hidden_units, bias=bias)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units, bias=bias)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units, bias=bias)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units, bias=bias)
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

class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Gate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.hidden_units
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        assert args.score_func == "sigmoid", "仅支持sigmoid评分函数"
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.hidden_units))
        nn.init.normal_(self.weight, std=0.02)  # 标准初始化
    def forward(self, x: torch.Tensor):
        scores = torch.matmul(x, self.weight.t())
        original_scores = torch.sigmoid(scores)
        topk_scores, indices = torch.topk(original_scores, k=self.topk, dim=-1)
        if self.score_func == "sigmoid":
            weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices

class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, bias):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=bias)
        self.w2 = nn.Linear(inter_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, inter_dim, bias=bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.hidden_units
        self.n_routed_experts = args.n_routed_experts  # 专家总数
        self.n_activated_experts = args.n_activated_experts  # 每个输入激活的专家数
        self.gate = Gate(args)
        self.experts = nn.ModuleList([
            Expert(args.hidden_units, args.moe_inter_dim, args.ffn_bias) 
            for _ in range(self.n_routed_experts)
        ])
        self.shared_experts = MLP(args.hidden_units, args.n_shared_experts * args.moe_inter_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        T = x.size(0)  # 总token数 (batch_size * seq_len)
        expert_counts = torch.zeros(self.n_routed_experts, device=x.device)  # f_i
        expert_probs = torch.zeros(self.n_routed_experts, device=x.device)   # P_i
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
        for expert_idx in range(self.n_routed_experts):
            if counts[expert_idx] == 0:
                continue
            expert = self.experts[expert_idx]
            sample_indices, weight_indices = torch.where(indices == expert_idx)
            expert_output = expert(x[sample_indices])
            current_weights = weights[sample_indices, weight_indices].unsqueeze(1)
            y[sample_indices] += expert_output * weights[sample_indices, weight_indices].unsqueeze(1)
            expert_counts[expert_idx] = sample_indices.size(0)
            expert_probs[expert_idx] = current_weights.sum()
        f_i = (self.n_routed_experts * expert_counts) / (self.n_activated_experts * T)
        P_i = expert_probs / T
        L_ExpBal = (f_i * P_i).sum()
        z = self.shared_experts(x)
        return (y + z).view(original_shape), L_ExpBal

class BaselineModel(torch.nn.Module):
    """
    Args:
        user_num: 用户数量
        item_num: 物品数量
        feat_statistics: 特征统计信息，key为特征ID，value为特征数量
        feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        args: 全局参数

    Attributes:
        user_num: 用户数量
        item_num: 物品数量
        dev: 设备
        norm_first: 是否先归一化
        maxlen: 序列最大长度
        item_emb: Item Embedding Table
        user_emb: User Embedding Table
        sparse_emb: 稀疏特征Embedding Table
        emb_transform: 多模态特征的线性变换
        userdnn: 用户特征拼接后经过的全连接层
        itemdnn: 物品特征拼接后经过的全连接层
    """

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):  #
        super(BaselineModel, self).__init__()
        self.args = args
        self.temperature = args.temperature
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.norm_method = getattr(args, 'norm_method', 'layer')  # 默认为layer norm
        if self.norm_method not in ['rms', 'layer']:
            raise ValueError(f"不支持的归一化方法: {self.norm_method}，支持的方法: 'rms', 'layer'")
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.use_cos_similarity = args.use_cos_similarity
        self._init_feat_info(feat_statistics, feat_types)
        user_dim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT
        )
        item_dim = (
            args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT))
            + len(self.ITEM_CONTINUAL_FEAT)
            + args.hidden_units * len(self.ITEM_EMB_FEAT)
        )
        self.item_dnn = DNN(item_dim, args.item_dnn_hidden_units, args.hidden_units, args)
        self.user_dnn = DNN(user_dim, args.user_dnn_hidden_units, args.hidden_units, args)
        if self.norm_method == 'rms':
            self.last_layernorm = RMSNorm(args.hidden_units, eps=1e-8)
        else:
            self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        for _ in range(args.num_blocks):
            if self.norm_method == 'rms':
                new_attn_layernorm = RMSNorm(args.hidden_units, eps=1e-8)
            else:
                new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = FlashMultiHeadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate, args.mha_bias
            )  
            self.attention_layers.append(new_attn_layer)
            if self.norm_method == 'rms':
                new_fwd_layernorm = RMSNorm(args.hidden_units, eps=1e-8)
            else:
                new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = MoE(args)
            self.forward_layers.append(new_fwd_layer)
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)
    
    def _init_feat_info(self, feat_statistics, feat_types):
        """
        将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table
        Args:
            feat_statistics: 特征统计信息，key为特征ID，value为特征数量
            feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        """
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度

    def feat2tensor(self, seq_feature, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k: 特征ID
        Returns:
            batch_data: 特征值的tensor，形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        batch_size = len(seq_feature)
        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
            max_array_len = 0
            max_seq_len = 0
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))
            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]
            return torch.from_numpy(batch_data).to(self.dev)
        else:
            # 如果特征是Sparse类型，直接转换为tensor
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data

            return torch.from_numpy(batch_data).to(self.dev)

    def feat2emb(self, seq, feature_array, mask=None, include_user=False):
        """
        Args:
            seq: 序列ID
            feature_array: 特征list，每个元素为当前时刻的特征字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征，在两种情况下不打开：1) 训练时在转换正负样本的特征时（因为正负样本都是item）;2) 生成候选库item embedding时。

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        seq = seq.to(self.dev)
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
        ]
        if include_user:
            all_feat_types.extend(
                [
                    (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                    (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                    (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
                ]
            )
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue
            for k in feat_dict:
                tensor_feature = self.feat2tensor(feature_array, k)
                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type.endswith('continual'):
                    feat_list.append(tensor_feature.unsqueeze(2))
        for k in self.ITEM_EMB_FEAT:
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0])
            # pre-allocate tensor
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)
            for i, seq in enumerate(feature_array):
                for j, item in enumerate(seq):
                    if k in item:
                        batch_emb_data[i, j] = item[k]
            tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor_feature))
        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = self.item_dnn(all_item_emb)
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = self.user_dnn(all_user_emb)
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
        return seqs_emb

    def log2feats(self, log_seqs, mask, seq_feature):
        """
        Args:
            log_seqs: 序列ID
            mask: token类型掩码，1表示item token，2表示user token
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典

        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        seqs *= self.item_emb.embedding_dim**0.5
        poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        poss *= log_seqs != 0
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)
        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)
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
    
    def forward(
        self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, neg_feature, writer=None, global_step=None
    ):
        """
        训练时调用，计算正负样本的logits
        Args:
            user_item: 用户序列ID
            pos_seqs: 正样本序列ID
            neg_seqs: 负样本序列ID
            mask: token类型掩码，1表示item token，2表示user token
            next_mask: 下一个token类型掩码，1表示item token，2表示user token
            next_action_type: 下一个token动作类型，0表示曝光，1表示点击
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            pos_feature: 正样本特征list，每个元素为当前时刻的特征字典
            neg_feature: 负样本特征list，每个元素为当前时刻的特征字典
        Returns:
            pos_logits: 正样本logits，形状为 [batch_size, maxlen, batch_size]
            neg_logits: 负样本logits，形状为 [batch_size, maxlen, batch_size*neg_num]
        """
        log_feats = self.log2feats(user_item, mask, seq_feature)
        loss_mask = (next_mask == 1).to(self.dev)
        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)
        if self.use_cos_similarity:
            log_feats = F.normalize(log_feats, p=2, dim=-1, eps=1e-8)
            pos_embs = F.normalize(pos_embs, p=2, dim=-1, eps=1e-8)
            neg_embs = F.normalize(neg_embs, p=2, dim=-1, eps=1e-8)
        hidden_size = neg_embs.size(-1)
        pos_logits = (log_feats * pos_embs).sum(dim=-1, keepdim=True)
        neg_embedding_all = neg_embs.reshape(-1, hidden_size)
        neg_logits = torch.matmul(log_feats, neg_embedding_all.transpose(-1, -2))  # [B, L, BL]
        pos_logits = pos_logits[loss_mask.bool()]  
        neg_logits = neg_logits[loss_mask.bool()]  
        return pos_logits, neg_logits

    def predict(self, log_seqs, seq_feature, mask):
        """
        计算用户序列的表征
        Args:
            log_seqs: 用户序列ID
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            mask: token类型掩码，1表示item token，2表示user token
        Returns:
            final_feat: 用户序列的表征，形状为 [batch_size, hidden_units]
        """
        log_feats = self.log2feats(log_seqs, mask, seq_feature)
        if self.use_cos_similarity:
            log_feats = F.normalize(log_feats, p=2, dim=-1, eps=1e-8)
        final_feat = log_feats[:, -1, :]
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
            if self.use_cos_similarity:
                batch_emb = F.normalize(batch_emb, p=2, dim=-1, eps=1e-8)
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))
    
    def compute_infonce_loss(self, pos_logits, neg_logits):
        logits = torch.cat([pos_logits, neg_logits], dim=-1) / self.temperature
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)
        infonce_loss = F.cross_entropy(logits, labels)
        return infonce_loss
    
    def compute_triple_loss(self, pos_logits, neg_logits):
        delta_pctr = neg_logits.mean(dim=-1) - pos_logits.squeeze(-1)  
        loss_terms = torch.clamp(delta_pctr + self.args.margin, min=0) 
        triple_loss = self.args.Triple_Loss_lambda * loss_terms.mean()
        return triple_loss
    
    
