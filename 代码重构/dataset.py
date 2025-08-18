import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

class MyDataset(torch.utils.data.Dataset):
    # __init__ 简化，不再需要 args
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        # 加载必要的数据
        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer
        # 这部分逻辑保持不变，因为 __getitem__ 仍然需要它
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()
        # 新增一个属性来存储 embedding 模块
        self.embedding_modules = None
    # 新增方法：用于接收从主脚本传递过来的模块
    def set_embedding_modules(self, modules):
        self.embedding_modules = modules
        # 动态加载 mm_emb_dict，因为需要知道 mm_emb_ids
        self.mm_emb_ids = self.embedding_modules['feat_types']['item_emb']
        self.mm_emb_dict = load_mm_emb(Path(self.data_dir, "creative_emb"), self.mm_emb_ids)
        # 从 modules 获取 maxlen
        self.maxlen = modules['args'].maxlen # 假设 args 也被传入了

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = json.loads(line)
        return data

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            pos_feat: 正样本特征，每个元素为字典，key为特征ID，value为特征值
            neg_feat: 负样本特征，每个元素为字典，key为特征ID，value为特征值
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100',
            '117',
            '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

    def collate_fn(self, batch):
        # 0. 检查模块是否已设置
        if self.embedding_modules is None:
            raise RuntimeError("Embedding modules not set! Call set_embedding_modules() first.")
        # 1. 解包原始数据
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        # 2. 将 ID 和 mask 转为 Tensor
        seq_ids = torch.from_numpy(np.array(seq))
        pos_ids = torch.from_numpy(np.array(pos))
        neg_ids = torch.from_numpy(np.array(neg))
        token_type_mask = torch.from_numpy(np.array(token_type))
        next_token_type_mask = torch.from_numpy(np.array(next_token_type))
        next_action_type_mask = torch.from_numpy(np.array(next_action_type))
        # 3. 定义核心处理函数 (这是原来模型中 feat2emb 的逻辑)
        def process_features(ids, features, mask=None, include_user=False):
            # 将模块移动到目标设备
            device = self.embedding_modules['device']
            item_emb = self.embedding_modules['item_emb'].to(device)
            user_emb = self.embedding_modules['user_emb'].to(device)
            sparse_emb = self.embedding_modules['sparse_emb'].to(device)
            emb_transform = self.embedding_modules['emb_transform'].to(device)
            feat_types = self.embedding_modules['feat_types']
            item_emb_feat_shapes = self.embedding_modules['item_emb_feat_shapes']
            ids = ids.to(device)
            # a. 处理 Item/User ID Embedding
            if include_user:
                mask = mask.to(device)
                user_m = (mask == 2)
                item_m = (mask == 1)
                user_embedding = user_emb(user_m * ids)
                item_embedding = item_emb(item_m * ids)
                item_feat_list = [item_embedding]
                user_feat_list = [user_embedding]
            else:
                item_embedding = item_emb(ids)
                item_feat_list = [item_embedding]
                user_feat_list = [] # 占位
            # b. 辅助函数：将特征列表转换为 padded tensor (这是原来 feat2tensor 的逻辑)
            def feature_to_tensor(feat_list, k, is_array=False, is_emb=False, emb_dim=0):
                batch_size = len(feat_list)
                max_seq_len = len(feat_list[0])
                if is_array:
                    max_array_len = 0
                    for i in range(batch_size):
                        seq_data = [item[k] for item in feat_list[i]]
                        max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data if item_data is not None))
                    batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
                    for i in range(batch_size):
                        for j, item in enumerate(feat_list[i]):
                            item_data = item[k]
                            if item_data is not None:
                                actual_len = min(len(item_data), max_array_len)
                                batch_data[i, j, :actual_len] = item_data[:actual_len]
                    return torch.from_numpy(batch_data).to(device)
                elif is_emb:
                    batch_data = np.zeros((batch_size, max_seq_len, emb_dim), dtype=np.float32)
                    for i, seq in enumerate(feat_list):
                        for j, item in enumerate(seq):
                            if k in item and item[k] is not None:
                                batch_data[i, j] = item[k]
                    return torch.from_numpy(batch_data).to(device)
                else: # Sparse or Continual
                    dtype = np.float32 if k in feat_types.get('user_continual', []) + feat_types.get('item_continual', []) else np.int64
                    batch_data = np.zeros((batch_size, max_seq_len), dtype=dtype)
                    for i in range(batch_size):
                        batch_data[i] = [item[k] for item in feat_list[i]]
                    tensor = torch.from_numpy(batch_data).to(device)
                    return tensor if dtype == np.int64 else tensor.unsqueeze(2)
            # c. 处理所有特征类型
            feature_map = {
                'item_sparse': (feat_types['item_sparse'], item_feat_list, False, False),
                'item_array': (feat_types['item_array'], item_feat_list, True, False),
                'item_continual': (feat_types['item_continual'], item_feat_list, False, False),
                'item_emb': (feat_types['item_emb'], item_feat_list, False, True),
            }
            if include_user:
                feature_map.update({
                    'user_sparse': (feat_types['user_sparse'], user_feat_list, False, False),
                    'user_array': (feat_types['user_array'], user_feat_list, True, False),
                    'user_continual': (feat_types['user_continual'], user_feat_list, False, False),
                })
            for ftype, (feat_ids, feat_l, is_arr, is_emb) in feature_map.items():
                for k in feat_ids:
                    tensor_feat = feature_to_tensor(features, k, is_array=is_arr, is_emb=is_emb, emb_dim=item_emb_feat_shapes.get(k, 0))
                    if ftype.endswith('sparse'):
                        feat_l.append(sparse_emb[k](tensor_feat))
                    elif ftype.endswith('array'):
                        feat_l.append(sparse_emb[k](tensor_feat).sum(2))
                    elif ftype.endswith('continual'):
                        feat_l.append(tensor_feat) # 已经是 [B, L, 1]
                    elif ftype.endswith('emb'):
                        feat_l.append(emb_transform[k](tensor_feat))
            # d. 拼接所有特征 embedding
            final_item_emb = torch.cat(item_feat_list, dim=2) if item_feat_list else None
            final_user_emb = torch.cat(user_feat_list, dim=2) if user_feat_list else None
            return final_item_emb, final_user_emb
        # 4. 对序列、正样本、负样本分别进行处理
        log_item_emb, log_user_emb = process_features(seq_ids, seq_feat, mask=token_type_mask, include_user=True)
        pos_embs, _ = process_features(pos_ids, pos_feat, include_user=False)
        neg_embs, _ = process_features(neg_ids, neg_feat, include_user=False)
        # 5. 返回一个字典，包含所有处理好的、可以直接输入模型的 Tensor
        return {
            "log_seqs": seq_ids.to(self.embedding_modules['device']),
            "log_item_emb": log_item_emb,
            "log_user_emb": log_user_emb,
            "pos_embs": pos_embs,
            "neg_embs": neg_embs,
            "mask": token_type_mask.to(self.embedding_modules['device']),
            "next_mask": next_token_type_mask.to(self.embedding_modules['device']),
            "next_action_type": next_action_type_mask.to(self.embedding_modules['device']),
        }

class MyTestDataset(MyDataset):
    """
    修改后的测试数据集。
    继承自修改后的 MyDataset，共享其大部分方法，但重写了数据加载和 collate_fn。
    """

    # __init__ 也简化，不再需要 args
    def __init__(self, data_dir):
        # 调用父类的构造函数，但跳过 MyDataset 的 _load_data_and_offsets
        # 我们将直接调用自己的 _load_data_and_offsets
        super(MyDataset, self).__init__() # 注意这里调用的是更上层的构造函数
        
        # 重新设置 MyDataset 的一些属性
        self.data_dir = Path(data_dir)
        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer
        
        # 调用自己的数据加载方法
        self._load_data_and_offsets()
        
        # 这部分逻辑和父类一样
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()
        self.embedding_modules = None

    def _load_data_and_offsets(self):
        """
        重写此方法以加载预测集的数据和偏移量。
        """
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    # _process_cold_start_feat 方法保持不变
    def _process_cold_start_feat(self, feat):
        # ... (您的代码不变) ...
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if type(feat_value) == list:
                value_list = []
                for v in feat_value:
                    if type(v) == str:
                        value_list.append(0)
                    else:
                        value_list.append(v)
                processed_feat[feat_id] = value_list
            elif type(feat_value) == str:
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    # __getitem__ 方法保持不变
    def __getitem__(self, uid):
        # ... (您的代码不变) ...
        # 它仍然返回 (seq, token_type, seq_feat, user_id)
        user_sequence = self._load_user_data(uid)

        ext_user_sequence = []
        user_id = '' # 初始化 user_id
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple
            if u:
                if type(u) == str:
                    user_id = u
                    u_reid = self.indexer['u'].get(u, 0) # 冷启动用户
                else:
                    user_id = self.indexer_u_rev.get(u, '')
                    u_reid = u
            if u and user_feat:
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u_reid, user_feat, 2))

            if i and item_feat:
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)

        idx = self.maxlen
        for record_tuple in reversed(ext_user_sequence): # 预测时可能没有下一个item，所以遍历全部
            i, feat, type_ = record_tuple
            feat = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        return seq, token_type, seq_feat, user_id

    # __len__ 方法保持不变
    def __len__(self):
        return len(self.seq_offsets)

    # 彻底重写 collate_fn
    def collate_fn(self, batch):
        """
        将多个 __getitem__ 返回的数据拼接成一个 batch，并进行特征到 Embedding 的转换。
        """
        # 0. 检查模块是否已设置
        if self.embedding_modules is None:
            raise RuntimeError("Embedding modules not set! Call set_embedding_modules() first.")

        # 1. 解包原始数据
        seq, token_type, seq_feat, user_id = zip(*batch)

        # 2. 将 ID 和 mask 转为 Tensor
        seq_ids = torch.from_numpy(np.array(seq))
        token_type_mask = torch.from_numpy(np.array(token_type))

        # 3. 复用 MyDataset 中的特征处理逻辑
        #    为了避免代码重复，理想情况下 process_features 应该是一个可以被两者调用的静态方法或辅助函数。
        #    这里我们再次定义它，逻辑与 MyDataset.collate_fn 中的完全一样。
        def process_features(ids, features, mask):
            # 将模块移动到目标设备
            device = self.embedding_modules['device']
            item_emb = self.embedding_modules['item_emb'].to(device)
            user_emb = self.embedding_modules['user_emb'].to(device)
            sparse_emb = self.embedding_modules['sparse_emb'].to(device)
            emb_transform = self.embedding_modules['emb_transform'].to(device)
            feat_types = self.embedding_modules['feat_types']
            item_emb_feat_shapes = self.embedding_modules['item_emb_feat_shapes']

            ids = ids.to(device)
            mask = mask.to(device)
            
            # a. 处理 Item/User ID Embedding
            user_m = (mask == 2)
            item_m = (mask == 1)
            user_embedding = user_emb(user_m * ids)
            item_embedding = item_emb(item_m * ids)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]

            # b. 辅助函数：将特征列表转换为 padded tensor (与 MyDataset.collate_fn 中的相同)
            def feature_to_tensor(feat_list, k, is_array=False, is_emb=False, emb_dim=0):
                batch_size = len(feat_list)
                max_seq_len = len(feat_list[0])
                
                if is_array:
                    max_array_len = 0
                    for i in range(batch_size):
                        seq_data = [item[k] for item in feat_list[i]]
                        max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data if item_data is not None))
                    
                    batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
                    for i in range(batch_size):
                        for j, item in enumerate(feat_list[i]):
                            item_data = item[k]
                            if item_data is not None:
                                actual_len = min(len(item_data), max_array_len)
                                batch_data[i, j, :actual_len] = item_data[:actual_len]
                    return torch.from_numpy(batch_data).to(device)
                
                elif is_emb:
                    batch_data = np.zeros((batch_size, max_seq_len, emb_dim), dtype=np.float32)
                    for i, seq_ in enumerate(feat_list):
                        for j, item in enumerate(seq_):
                            if k in item and item[k] is not None:
                                batch_data[i, j] = item[k]
                    return torch.from_numpy(batch_data).to(device)

                else: # Sparse or Continual
                    dtype = np.float32 if k in feat_types.get('user_continual', []) + feat_types.get('item_continual', []) else np.int64
                    batch_data = np.zeros((batch_size, max_seq_len), dtype=dtype)
                    for i in range(batch_size):
                        batch_data[i] = [item[k] for item in feat_list[i]]
                    tensor = torch.from_numpy(batch_data).to(device)
                    return tensor if dtype == np.int64 else tensor.unsqueeze(2)

            # c. 处理所有特征类型
            feature_map = {
                'item_sparse': (feat_types['item_sparse'], item_feat_list, False, False),
                'item_array': (feat_types['item_array'], item_feat_list, True, False),
                'item_continual': (feat_types['item_continual'], item_feat_list, False, False),
                'item_emb': (feat_types['item_emb'], item_feat_list, False, True),
                'user_sparse': (feat_types['user_sparse'], user_feat_list, False, False),
                'user_array': (feat_types['user_array'], user_feat_list, True, False),
                'user_continual': (feat_types['user_continual'], user_feat_list, False, False),
            }
            
            for ftype, (feat_ids, feat_l, is_arr, is_emb) in feature_map.items():
                for k in feat_ids:
                    tensor_feat = feature_to_tensor(features, k, is_array=is_arr, is_emb=is_emb, emb_dim=item_emb_feat_shapes.get(k, 0))
                    
                    if ftype.endswith('sparse'):
                        feat_l.append(sparse_emb[k](tensor_feat))
                    elif ftype.endswith('array'):
                        feat_l.append(sparse_emb[k](tensor_feat).sum(2))
                    elif ftype.endswith('continual'):
                        feat_l.append(tensor_feat)
                    elif ftype.endswith('emb'):
                        feat_l.append(emb_transform[k](tensor_feat))
            
            # d. 拼接所有特征 embedding
            final_item_emb = torch.cat(item_feat_list, dim=2)
            final_user_emb = torch.cat(user_feat_list, dim=2)
            
            return final_item_emb, final_user_emb

        # 4. 对序列特征进行处理
        log_item_emb, log_user_emb = process_features(seq_ids, seq_feat, mask=token_type_mask)

        # 5. 返回一个字典，包含所有处理好的、可以直接输入 predict 函数的 Tensor
        return {
            "log_seqs": seq_ids.to(self.embedding_modules['device']),
            "log_item_emb": log_item_emb,
            "log_user_emb": log_user_emb,
            "mask": token_type_mask.to(self.embedding_modules['device']),
            "user_id": user_id, # 原始的 user_id 仍然透传出去
        }




def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('*.json'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict
