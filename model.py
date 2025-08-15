

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

    def compute_infonce_weight_loss(self, log_feats, pos_embs, neg_embs, loss_mask, next_action_type):
        """
        计算带行为权重加权的InfoNCE Loss。

        Args:
            log_feats (torch.Tensor): 序列的输出表示, 形状 [B, L, D]
            pos_embs (torch.Tensor): 正样本的嵌入, 形状 [B, L, D]
            neg_embs (torch.Tensor): 负样本的嵌入, 形状 [B, L, D]
            loss_mask (torch.Tensor): 损失掩码，标记哪些位置需要计算损失, 形状 [B, L]
            next_action_type (torch.Tensor): 下一个行为的类型, 形状 [B, L]

        Returns:
            torch.Tensor: 计算出的加权损失值 (一个标量)
        """
        # 1. 向量归一化 (L2-Normalization)，这是计算余弦相似度的标准步骤
        # log_feats = log_feats / log_feats.norm(dim=-1, keepdim=True)
        # pos_embs = pos_embs / pos_embs.norm(dim=-1, keepdim=True)
        # neg_embs = neg_embs / neg_embs.norm(dim=-1, keepdim=True)

        # 2. 计算正样本的 logits (余弦相似度)
        # 形状: [B, L] -> [B, L, 1]
        pos_logits = F.cosine_similarity(log_feats, pos_embs, dim=-1).unsqueeze(-1)
        
        # 3. 计算负样本的 logits
        # 为了进行高效的矩阵乘法，我们将负样本展平
        # neg_embs: [B, L, D] -> neg_embedding_all: [B*L, D]
        batch_size, seq_len, hidden_size = neg_embs.size()
        neg_embedding_all = neg_embs.reshape(-1, hidden_size)
        
        # log_feats 与所有负样本计算相似度
        # 形状: [B, L, D] x [D, B*L] -> [B, L, B*L]
        neg_logits = torch.matmul(log_feats, neg_embedding_all.transpose(-1, -2))
        
        # 4. 拼接正负样本的 logits
        # 形状: [B, L, 1+B*L]
        logits = torch.cat([pos_logits, neg_logits], dim=-1)

        # 5. 应用 loss_mask 筛选出需要计算损失的位置
        # loss_mask.bool() 会将 1/0 的掩码转为 True/False
        # logits[loss_mask.bool()] 会将所有为 True 位置的 logits 向量展平
        # 假设有 N 个 True 的位置, 结果形状为 [N, 1+B*L]
        valid_logits = logits[loss_mask.bool()] / 0.07
        
        # 6. 【核心修改】计算每个有效位置的权重
        # 首先，筛选出有效位置对应的 action_type
        # 形状: [N]
        valid_action_types = next_action_type[loss_mask.bool()]
        
        # 根据 action_type 生成权重 (点击为1.0, 曝光为alpha)
        # 形状: [N]
        weights = torch.where(valid_action_types == 1, 1.0, 0.1).to(torch.float32)

        # 7. 创建标签 (对于InfoNCE，正样本总是在第一个位置，所以标签是0)
        # 形状: [N]
        labels = torch.zeros(valid_logits.size(0), device=valid_logits.device, dtype=torch.int64)
        
        # 8. 计算带权重的交叉熵损失
        # F.cross_entropy 默认会对 batch 内的损失取平均
        # 为了应用我们自己的权重，需要设置 reduction='none'
        # unweighted_loss 的形状为 [N]
        unweighted_loss = F.cross_entropy(valid_logits, labels, reduction='none')
        
        # 应用权重并计算加权平均损失
        # (unweighted_loss * weights).sum() 是加权后的总损失
        # weights.sum() 是总权重，用总损失除以总权重得到加权平均损失
        # 添加一个很小的 epsilon 防止除以零
        loss = (unweighted_loss * weights).sum() / (weights.sum() + 1e-8)

        return loss
