import json
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from tqdm import tqdm
from utils.data_utils import bio_to_spans

class InputFeatures(object):
    
    def __init__(self,
                 input_ids,
                 indexes,
                 bpe_len,
                 word_len=None,
                 matrix=None,
                 offset=None,
                 example_id=None,
                 ent_target=None,
                 cand_indexes=None
                 ):
        self.input_ids = input_ids
        self.indexes = indexes
        self.bpe_len = bpe_len
        self.word_len = word_len
        self.matrix = matrix
        self.offset = offset
        self.example_id = example_id
        self.ent_target = ent_target
        self.cand_indexes = cand_indexes

class EntDataset(Dataset):
    def __init__(self, data, tokenizer, ent2id, model_name='bert', max_len=512, train_stride=1, window=128, is_train=True, json=False):
        self.tokenizer = tokenizer
        self.ent2id = ent2id
        self.max_len = max_len
        self.train_stride = train_stride
        self.is_train = is_train
        if 'roberta' in model_name:
            self.add_prefix_space = True
            self.cls = self.tokenizer.cls_token_id
            self.sep = self.tokenizer.sep_token_id
        elif 'deberta' in model_name:
            self.add_prefix_space = False
            self.cls = self.tokenizer.bos_token_id
            self.sep = self.tokenizer.eos_token_id
        elif 'bert' in model_name:
            self.add_prefix_space = False
            self.cls = self.tokenizer.cls_token_id
            self.sep = self.tokenizer.sep_token_id
        else:
            raise RuntimeError(f"Unsupported {model_name}")
        self.mlm_probability = 0.15
        self.window = window
        if json:
            self.data = self.convert_json(data)
        else:
            self.data = self.convert_conllx(data)

    def __len__(self):
        return len(self.data)
    
    def get_new_ins(self, bpes, spans, indexes, cand_indexes=None, offset=None, example_id=None):
            
            bpes.append(self.sep)
            cur_word_idx = indexes[-1]
            indexes.append(0)
            
            if self.is_train:

                matrix = np.zeros((cur_word_idx, 2*self.window+1, len(self.ent2id)), dtype=np.int8)
                for _ner in spans:
                    s, e, t = _ner
                    if (e-s) <= self.window:
                        matrix[s, self.window+e-s, t] = 1
                        matrix[e, self.window-e+s, t] = 1

                assert len(bpes)<=self.max_len, len(bpes)
                new_ins = InputFeatures(input_ids=bpes, indexes=indexes, bpe_len=len(bpes), matrix=matrix, cand_indexes=cand_indexes)
            
            else:
                ent_target = []
                for _ner in spans:
                    s, e, t = _ner
                    ent_target.append((s, e, t))
                assert len(bpes)<=self.max_len, len(bpes)
                new_ins = InputFeatures(input_ids=bpes, indexes=indexes, bpe_len=len(bpes), word_len=cur_word_idx, offset=offset, example_id=example_id, ent_target=ent_target)

            return new_ins

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]
        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)


    def convert_json(self, path):
        ins_lst = []
        word2bpes = {}
        max_sent_length = 0
        # 修改的部分开始
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                if line.strip():                        # 忽略空行
                    entry = json.loads(line.strip())    # 逐行解析 JSON
                    # 适配字段名称
                    raw_words = entry.get("sentence", entry.get("text", []))  # 兼容两种字段名
                    sent_len = len(raw_words)
                    max_sent_length = max(max_sent_length, sent_len)
                    # example_id = entry.get("doc_id", "")
                    # 处理依存关系（如果存在）
                    depheads = None
                    deplabels = None
                    if "dephead" in entry:
                        depheads = [head - 1 for head in entry["dephead"]]  # 转换为 root is -1 not 0
                    if "deplabel" in entry:
                        deplabels = entry["deplabel"]
                    # 解析 NER 信息 - 新格式
                    raw_ents = []
                    entities = entry.get("entities", [])
                    for entity in entities:
                        start = entity["start"]         # 实体起始位置
                        end = entity["end"]             # 实体结束位置
                        entity_type = entity["type"]    # 实体类型
                        
                        if start <= end and entity_type in self.ent2id:
                            raw_ents.append((start, end, self.ent2id[entity_type]))
                    # 处理关系信息（如果存在）
                    relations = entry.get("relations", [])

                    bpes = []
                    indexes = []
                    cand_indexes = []
                    for idx, word in enumerate(raw_words, start=0):
                        if word in word2bpes:
                            _bpes = word2bpes[word]
                        else:
                            _bpes = self.tokenizer.encode(' '+word if self.add_prefix_space else word,
                                                        add_special_tokens=False)
                            word2bpes[word] = _bpes
                        cand_indexes.append(list(range(len(bpes)+1, len(bpes)+len(_bpes)+1)))
                        indexes.extend([idx]*len(_bpes))
                        bpes.extend(_bpes)
                    
                    new_bpes = [[self.cls] + bpes[i:i+self.max_len-2] for i in range(0, len(bpes), self.max_len-self.train_stride-1)]
                    new_indexes = [indexes[i:i+self.max_len-2] for i in range(0, len(indexes), self.max_len-self.train_stride-1)]
                    
                    for _bpes, _indexes in zip(new_bpes, new_indexes):
                        spans = []
                        offset = _indexes[0]
                        for s, e, t in raw_ents:
                            if _indexes[0] <= s <= e <= _indexes[-1]:
                                spans += [(s-_indexes[0], e-_indexes[0], t)]
                        
                        if self.is_train:
                            _indexes = [0] + [i - offset + 1 for i in _indexes]
                            new_ins = self.get_new_ins(_bpes, spans, _indexes, cand_indexes)
                            ins_lst.append(new_ins)
                        else:
                            _indexes = [0] + [i - offset + 1 for i in _indexes]
                            new_ins = self.get_new_ins(_bpes, spans, _indexes, offset=offset)
                            ins_lst.append(new_ins)
        file_name = os.path.basename(path)
        print(f"{file_name} 最大句子长度: {max_sent_length}")
        return ins_lst
        
    def convert_conllx(self, path):
        ins_lst = []
        word2bpes = {}
        max_sent_length = 0
        
        with open(path, 'r', encoding='utf-8') as f:
            raw_words, raw_labels, depheads, deplabels  = [], [], [], []
            for line in tqdm(f):
                line = line.strip()
                # 跳过文档开始标记
                if line.startswith("-DOCSTART"):
                    continue
                # 遇到空行，处理当前句子
                if line == "" and len(raw_words) != 0:
                    sent_len = len(raw_words)
                    max_sent_length = max(max_sent_length, sent_len)
                    # 将 BIO 标签转换为实体spans
                    raw_ents = bio_to_spans(self.ent2id, raw_labels)
                    # 处理 BPE 编码
                    bpes, indexes, cand_indexes = [], [], []
                    
                    for idx, word in enumerate(raw_words, start=0):
                        if word in word2bpes:
                            _bpes = word2bpes[word]
                        else:
                            _bpes = self.tokenizer.encode(' '+word if self.add_prefix_space else word,
                                                        add_special_tokens=False)
                            word2bpes[word] = _bpes
                        cand_indexes.append(list(range(len(bpes)+1, len(bpes)+len(_bpes)+1)))
                        indexes.extend([idx]*len(_bpes))
                        bpes.extend(_bpes)
                    
                    # 分割长句子
                    new_bpes = [[self.cls] + bpes[i:i+self.max_len-2] for i in range(0, len(bpes), self.max_len-self.train_stride-1)]
                    new_indexes = [indexes[i:i+self.max_len-2] for i in range(0, len(indexes), self.max_len-self.train_stride-1)]
                    
                    for _bpes, _indexes in zip(new_bpes, new_indexes):
                        spans = []
                        offset = _indexes[0] if _indexes else 0
                        
                        # 过滤在当前窗口内的实体
                        for s, e, t in raw_ents:
                            if _indexes and _indexes[0] <= s <= e <= _indexes[-1]:
                                spans += [(s-_indexes[0], e-_indexes[0], t)]
                        
                        if self.is_train:
                            _indexes = [0] + [i - offset + 1 for i in _indexes]
                            new_ins = self.get_new_ins(_bpes, spans, _indexes, cand_indexes)
                            ins_lst.append(new_ins)
                        else:
                            _indexes = [0] + [i - offset + 1 for i in _indexes]
                            new_ins = self.get_new_ins(_bpes, spans, _indexes, offset=offset)
                            ins_lst.append(new_ins)
                    # 重置变量
                    raw_words, raw_labels, depheads, deplabels  = [], [], [], []
                    continue
                elif line == "" and len(raw_words) == 0:
                    continue
                
                # 解析每一行数据
                ls = line.split('\t')  # 使用制表符分割
                if len(ls) >= 9:  # 确保有足够的列
                    word_idx = int(ls[0]) - 1  # 词索引（从0开始）
                    word = ls[1]               # 词
                    head = int(ls[6]) - 1      # 依存头（转换为从0开始，root为-1）
                    dep_label = ls[7]          # 依存关系标签
                    ner_label = ls[-1]         # NER标签
                    
                    raw_words.append(word)
                    raw_labels.append(ner_label)
                    depheads.append(head)
                    deplabels.append(dep_label)
            
            # 处理文件末尾的最后一个句子（如果存在）
            if len(raw_words) != 0:
                sent_len = len(raw_words)
                max_sent_length = max(max_sent_length, sent_len)
                raw_ents = bio_to_spans(raw_labels)

                bpes = []
                indexes = []
                cand_indexes = []
                
                for idx, word in enumerate(raw_words, start=0):
                    if word in word2bpes:
                        _bpes = word2bpes[word]
                    else:
                        _bpes = self.tokenizer.encode(' '+word if self.add_prefix_space else word,
                                                    add_special_tokens=False)
                        word2bpes[word] = _bpes
                    cand_indexes.append(list(range(len(bpes)+1, len(bpes)+len(_bpes)+1)))
                    indexes.extend([idx]*len(_bpes))
                    bpes.extend(_bpes)
                
                new_bpes = [[self.cls] + bpes[i:i+self.max_len-2] for i in range(0, len(bpes), self.max_len-self.train_stride-1)]
                new_indexes = [indexes[i:i+self.max_len-2] for i in range(0, len(indexes), self.max_len-self.train_stride-1)]
                
                for _bpes, _indexes in zip(new_bpes, new_indexes):
                    spans = []
                    offset = _indexes[0] if _indexes else 0
                    
                    for s, e, t in raw_ents:
                        if _indexes and _indexes[0] <= s <= e <= _indexes[-1]:
                            spans += [(s-_indexes[0], e-_indexes[0], t)]
                    
                    if self.is_train:
                        _indexes = [0] + [i - offset + 1 for i in _indexes]
                        new_ins = self.get_new_ins(_bpes, spans, _indexes, cand_indexes)
                        ins_lst.append(new_ins)
                    else:
                        _indexes = [0] + [i - offset + 1 for i in _indexes]
                        new_ins = self.get_new_ins(_bpes, spans, _indexes, offset=offset)
                        ins_lst.append(new_ins)
        
        file_name = os.path.basename(path)
        print(f"{file_name} 最大句子长度: {max_sent_length}")
        return ins_lst
    
    def collate(self, examples):
        
        if self.is_train:

            batch_input_id, batch_index, batch_bpe_len, batch_matrix = [], [], [], []
            batch_mask_labels = []
            for item in examples:
                batch_input_id.append(item.input_ids)
                batch_index.append(item.indexes)
                batch_bpe_len.append(item.bpe_len)
                batch_matrix.append(item.matrix)
                
                # WholeWordMask
                random.shuffle(item.cand_indexes)
                num_to_predict = max(1, int(round(len(item.input_ids) * self.mlm_probability)))
                masked_lms = []
                covered_indexes = set()
                for index_set in item.cand_indexes:
                    if len(masked_lms) >= num_to_predict:
                        break
                    # If adding a whole-word mask would exceed the maximum number of
                    # predictions, then just skip this candidate.
                    if len(masked_lms) + len(index_set) > num_to_predict:
                        continue
                    is_any_index_covered = False
                    for index in index_set:
                        if index in covered_indexes:
                            is_any_index_covered = True
                            break
                    if is_any_index_covered:
                        continue
                    for index in index_set:
                        covered_indexes.add(index)
                        masked_lms.append(index)
                if len(covered_indexes) != len(masked_lms):
                    raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
                mask_labels = [1 if i in covered_indexes else 0 for i in range(len(item.input_ids))]
                batch_mask_labels.append(mask_labels)

            batch_input_ids = torch.tensor(self.sequence_padding(batch_input_id, value=self.tokenizer.pad_token_id)).long()
            batch_indexes = torch.tensor(self.sequence_padding(batch_index)).long()
            batch_bpe_lens = torch.tensor(batch_bpe_len).long()
            batch_labels = torch.tensor(self.sequence_padding(batch_matrix)).long()
            batch_masklabels = torch.tensor(self.sequence_padding(batch_mask_labels)).long()

            return self.torch_mask_tokens(batch_input_ids, batch_masklabels), batch_indexes, batch_bpe_lens, batch_labels
        
        else:
            
            batch_input_id, batch_index, batch_bpe_len, batch_word_len, batch_offset, batch_id, batch_ent_target = [], [], [], [], [], [], []
            for item in examples:
                batch_input_id.append(item.input_ids)
                batch_index.append(item.indexes)
                batch_bpe_len.append(item.bpe_len)
                batch_word_len.append(item.word_len)
                batch_offset.append(item.offset)
                batch_id.append(item.example_id)
                batch_ent_target.append(item.ent_target)

            batch_input_ids = torch.tensor(self.sequence_padding(batch_input_id, value=self.tokenizer.pad_token_id)).long()
            batch_indexes = torch.tensor(self.sequence_padding(batch_index)).long()
            batch_bpe_lens = torch.tensor(batch_bpe_len).long()
            batch_word_lens = torch.tensor(batch_word_len).long()

            return batch_input_ids, batch_indexes, batch_bpe_lens, batch_word_lens, batch_offset, batch_id, batch_ent_target

    def torch_mask_tokens(self, inputs, mask_labels=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        # labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(inputs.shape, self.mlm_probability) if mask_labels is None else mask_labels

        special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
            ]
        # probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        if self.tokenizer.pad_token is not None:
            padding_mask = inputs.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool() if mask_labels is None else probability_matrix.bool()
        # labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs

    def __getitem__(self, index):
        item = self.data[index]
        return item
    