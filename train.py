from utils.data_loader import EntDataset
from utils.data_utils import load_data
from transformers import AutoTokenizer, AutoModel, BertModel, RobertaModel
from torch.utils.data import DataLoader
import torch
import json
from models.model import CNNNer
from models.metrics import MetricsCalculator
from tqdm import tqdm
from utils.logger import logger
from transformers import set_seed
import argparse
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import gc

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

deepspeed = None # 单gpu/cpu

def clean_cache():
    """Clean cache to avoid memory leak.
    This fixes this issue: https://github.com/huggingface/transformers/issues/22801"""
    print(f"Cleaning GPU memory. Current memory usage: {torch.cuda.memory_allocated()}")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory usage after cleaning: {torch.cuda.memory_allocated()}")

def main(args, seed, max_len = 512):
    train_path = f"data/{args.task}/train.txt"
    dev_path = f"data/{args.task}/dev.txt"
    test_path = f"data/{args.task}/test.txt"
    if args.task == "ace2005" or args.task == "ace2004":
        ENT_CLS_NUM = 7
        json_flag = True
        ent2id = {"PER": 0, "ORG": 1, "LOC": 2, "GPE": 3, "FAC": 4, "VEH": 5, "WEA": 6}
    elif args.task == "genia":
        ENT_CLS_NUM = 5
        json_flag = True
        ent2id = {"DNA": 0, "RNA": 1, "cell_type": 2, "protein": 3, "cell_line": 4}
    elif args.task == "onto5en" or args.task == 'onto5cn':
        ENT_CLS_NUM = 18
        json_flag = False
        ent2id = {"NORP": 0, "ORG": 1, "PERSON":2, "DATE": 3, "GPE": 4, "FAC": 5, "CARDINAL": 6, "TIME": 7, "ORDINAL": 8, "EVENT": 9, 
                  "QUANTITY": 10, "PERCENT": 11, "LOC": 12, "WORK_OF_ART": 13, "MONEY": 14, "LAW": 15, "PRODUCT": 16, "LANGUAGE": 17}
    elif args.task == "conll03":
        ENT_CLS_NUM = 4
        json_flag = False
        ent2id = {"LOC": 0, "PER": 1, "MISC":2, "ORG": 3}
    elif args.task == "spanish" or args.task == "catalan":
        ENT_CLS_NUM = 4
        json_flag = False
        ent2id = {"org": 0, "misc": 1, "loc":2, "person": 3}
    elif args.task == "onto4cn":
        ENT_CLS_NUM = 4
        json_flag = False
        ent2id = {"LOC": 0, "PER": 1, "GPE":2, "ORG": 3}
    elif args.task == "resume":
        ENT_CLS_NUM = 8
        json_flag = False
        ent2id = {"NAME": 0, "PRO": 1, "EDU":2, "TITLE": 3, "ORG": 4, "CONT": 5, "RACE": 6, "LOC": 7}

    id2ent = {}
    for k, v in ent2id.items(): id2ent[v] = k

    weight_decay = 1e-2
    ent_thres = 0.5

    set_seed(seed)

    # deepspeed.init_distributed() 多卡采用
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    ner_train = EntDataset(train_path, tokenizer=tokenizer, ent2id=ent2id, model_name='roberta', max_len=max_len, window=args.chunks_size, json=json_flag)
    ner_dev = EntDataset(dev_path, tokenizer=tokenizer, ent2id=ent2id, model_name='roberta', max_len=max_len, window=args.chunks_size, is_train=False, json=json_flag)
    ner_test = EntDataset(test_path, tokenizer=tokenizer, ent2id=ent2id, model_name='roberta', max_len=max_len, window=args.chunks_size, is_train=False, json=json_flag)
    # ner_loader_evl = DataLoader(ner_evl, batch_size=2, collate_fn=ner_evl.collate, shuffle=False, num_workers=0)
    # 使用标准DataLoader
    ner_loader_train = DataLoader(ner_train, batch_size=args.batch_size, collate_fn=ner_train.collate, shuffle=True, num_workers=0)
    ner_loader_dev = DataLoader(ner_dev, batch_size=2, collate_fn=ner_dev.collate, shuffle=False, num_workers=0)
    ner_loader_test = DataLoader(ner_test, batch_size=2, collate_fn=ner_test.collate, shuffle=False, num_workers=0)
    dev_example = load_data(dev_path, ent2id, json_flag)
    test_example = load_data(test_path, ent2id, json_flag)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    encoder = RobertaModel.from_pretrained("roberta-base", use_safetensors=True)

    model = CNNNer(encoder, num_ner_tag=ENT_CLS_NUM, cnn_dim=args.cnn_dim, biaffine_size=args.biaffine_size,
                    size_embed_dim=0, logit_drop=args.logit_drop,
                   chunks_size=args.chunks_size, cnn_depth=args.cnn_depth, attn_dropout=0.2).to(device) # cuda()
    
    # config = LoraConfig(
    #         r=8,
    #         lora_alpha=16,
    #         target_modules=["query", "value"],  # BERT模块名, Roberta
    #         lora_dropout=0,
    #         bias="lora_only",
    #     )
    # model = get_peft_model(model, config)
    # enable trainable params
    for n, p in model.named_parameters():
        # if 'pretrain_model' not in n:
        p.requires_grad_()

    # model.print_trainable_parameters()

    # optimizer
    ln_params = []
    non_ln_params = []
    non_pretrain_params = []
    non_pretrain_ln_params = []

    import collections
    counter = collections.Counter()
    for name, param in model.named_parameters():
        counter[name.split('.')[0]] += torch.numel(param)
    print(counter)
    print("Total param ", sum(counter.values()))
    logger.info(json.dumps(counter, indent=2))
    logger.info(sum(counter.values()))

    # 优化器设置 - 全量微调版本
    def set_optimizer(model):
        ln_params = []
        non_ln_params = []
        non_pretrain_params = []
        non_pretrain_ln_params = []
        
        for name, param in model.named_parameters():
            name = name.lower()
            if not param.requires_grad:
                continue
            
            # 区分预训练部分和任务特定部分
            if 'encoder' in name:  # BERT编码器部分
                if 'norm' in name or 'bias' in name:
                    ln_params.append(param)
                else:
                    non_ln_params.append(param)
            else:  # CNN和分类器部分
                if 'norm' in name or 'bias' in name:
                    non_pretrain_ln_params.append(param)
                else:
                    non_pretrain_params.append(param)
        
        # 不同部分使用不同学习率
        optimizer_grouped_parameters = [
            {'params': non_ln_params, 'lr': args.lr, 'weight_decay': weight_decay},  # BERT权重
            {'params': ln_params, 'lr': args.lr, 'weight_decay': 0},                # BERT LayerNorm/bias
            {'params': non_pretrain_ln_params, 'lr': args.lr, 'weight_decay': 0},  # 任务层LayerNorm/bias
            {'params': non_pretrain_params, 'lr': args.lr, 'weight_decay': weight_decay},  # 任务层权重
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        return optimizer

    optimizer = set_optimizer(model)
    total_steps = (int(len(ner_train) / args.batch_size) + 1) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup * total_steps, num_training_steps = total_steps)
    # 移除DeepSpeed初始化，使用标准训练
    metrics = MetricsCalculator(ent_thres=ent_thres, id2ent=id2ent, allow_nested=True)
    best_dev_f1, best_test_f1 = 0.0, 0.0
    best_epoch, patience_counter = 0, 0

    for eo in range(args.n_epochs):
        loss_total = 0
        n_item = 0
        for idx, batch in enumerate(ner_loader_train):

            input_ids, indexes, bpe_len, matrix = batch
            input_ids, bpe_len, indexes, matrix = input_ids.to(device), bpe_len.to(device), indexes.to(device), matrix.to(device)
            # 标准PyTorch训练流程
            optimizer.zero_grad()
            loss = model(input_ids, bpe_len, indexes, matrix)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            loss_total += loss.item()
            cur_n_item = input_ids.shape[0]
            n_item += cur_n_item
        avg_loss = loss_total / n_item
        logger.info(f'*** Epoch {eo} loss: {avg_loss} ***')
        with torch.no_grad():
            model.eval()
            # 评估验证集
            dev_f1, dev_eval_info, dev_entity_info = evaluate_dataset(
                model, ner_loader_dev, dev_example, "Dev", eo, device, metrics, logger)
            
            # 评估测试集
            test_f1, test_eval_info, test_entity_info = evaluate_dataset(
                model, ner_loader_test, test_example, "Test", eo, device, metrics, logger)
        
            if dev_f1 > best_dev_f1:
                logger.info("Find best dev f1 at epoch {} (Dev F1: {:.6f}, Test F1: {:.6f})".format(eo, dev_f1, test_f1))
                best_epoch = eo
                best_dev_f1 = dev_f1
                best_test_f1 = test_f1
                best_dev_results = dev_eval_info.copy()  # 保存最佳验证集详细结果
                best_test_results = test_eval_info.copy()  # 保存对应测试集详细结果
                patience_counter = 0
            else:
                patience_counter += 1

        if patience_counter >= 10:
            break
    logger.info("\n" + "=" * 80)
    logger.info("FINAL TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info("Total training epochs: {}".format(eo))
    logger.info("Best epoch: {}".format(best_epoch if 'best_epoch' in locals() else 'N/A'))
    logger.info("Early stopping patience: {}".format(patience_counter))

    logger.info("\n--- BEST VALIDATION RESULTS ---")
    logger.info("Best Dev F1: {:.6f}".format(best_test_f1))
    if 'best_dev_results' in locals():
        logger.info("Best Dev Precision: {:.6f}".format(best_dev_results['acc']))
        logger.info("Best Dev Recall: {:.6f}".format(best_dev_results['recall']))
        logger.info("Best Dev Origin: {}".format(best_dev_results['origin']))
        logger.info("Best Dev Found: {}".format(best_dev_results['found']))
        logger.info("Best Dev Right: {}".format(best_dev_results['right']))

    logger.info("\n--- CORRESPONDING TEST RESULTS ---")
    if 'best_test_f' in locals():
        logger.info("Test F1 at best validation: {:.6f}".format(best_test_f1))
        if 'best_test_results' in locals():
            logger.info("Test Precision at best validation: {:.6f}".format(best_test_results['acc']))
            logger.info("Test Recall at best validation: {:.6f}".format(best_test_results['recall']))
            logger.info("Test Origin at best validation: {}".format(best_test_results['origin']))
            logger.info("Test Found at best validation: {}".format(best_test_results['found']))
            logger.info("Test Right at best validation: {}".format(best_test_results['right']))

    logger.info("\n--- MODEL INFORMATION ---")
    logger.info("Task: {}".format(args.task if 'args' in locals() else 'N/A'))
    logger.info("Max length: {}".format(max_len if 'max_len' in locals() else 'N/A'))
    logger.info("Seed: {}".format(seed if 'seed' in locals() else 'N/A'))
    logger.info("Device: {}".format(device))

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)

def evaluate_dataset(model, data_loader, examples, dataset_name, epoch, device, metrics, logger):
    total_X, total_Y, total_Z = [], [], []
    pre, pre_offset, pre_example_id, word_lens = [], [], [], []
    
    for batch in tqdm(data_loader, desc=dataset_name):
        input_ids, indexes, bpe_len, word_len, offset, example_id, ent_target = batch
        input_ids, bpe_len, indexes = input_ids.to(device), bpe_len.to(device), indexes.to(device)
        logits = model(input_ids, bpe_len, indexes)
        pre += logits
        pre_offset += offset
        pre_example_id += example_id
        word_lens += word_len

    total_X, total_Y, total_Z = metrics.get_evaluate_fpr_overlap(examples, pre, word_lens, pre_offset, pre_example_id)
    eval_info, entity_info = metrics.result(total_X, total_Y, total_Z)
    f1_score = round(eval_info['f1'], 6)
    
    logger.info('\n{} Eval Epoch{}  precision:{:.6f}  recall:{:.6f}  f1:{:.6f}  origin:{}  found:{}  right:{}'.format(
        dataset_name, epoch, round(eval_info['acc'], 6), round(eval_info['recall'], 6), 
        f1_score, eval_info['origin'], eval_info['found'], eval_info['right']))
   
    for item in entity_info.keys():
        logger.info('-- {} item:  {}  precision:{:.6f}  recall:{:.6f}  f1:{:.6f}  origin:{}  found:{}  right:{}'.format(
            dataset_name, item, round(entity_info[item]['acc'], 6), round(entity_info[item]['recall'], 6), 
            round(entity_info[item]['f1'], 6), entity_info[item]['origin'], 
            entity_info[item]['found'], entity_info[item]['right']))
    
    return f1_score, eval_info, entity_info


if __name__ == '__main__':
    
    import random
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('-n', '--n_epochs', default=30, type=int)
    parser.add_argument('--warmup', default=0.1, type=float)
    parser.add_argument('--cnn_depth', default=2, type=int)
    parser.add_argument('--cnn_dim', default=32, type=int)
    parser.add_argument('--logit_drop', default=0.1, type=float)
    parser.add_argument('--biaffine_size', default=100, type=int)
    # ace2004 max span size: 8
    # ace2005 max span size: 16, recommend 9, max seqlen 120
    # genia max span size: 18, recommend 12
    # onto5-en max span size: 28, recommend 15
    # onto5-ch max span size: 19, recommend 12
    # conll03 max seqlen 124
    parser.add_argument('--chunks_size', default=120, type=int)
    parser.add_argument('--task', default="conll03")

    args = parser.parse_args()

    seed = random.sample(range(1000,10000),3)

    max_len = [5120] 

    for l in max_len:
        for idx in range(len(seed)):
            main(args, int(seed[idx]), int(l))
            clean_cache()
    
    print("seed", seed)
