"""
模型训练脚本
用于训练基于Transformer的问答模型
"""

import os
import json
import logging
import torch
from torch.utils.data import Dataset
from torch.optim import AdamW
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    default_data_collator,
    TrainingArguments,
    Trainer
)
from datasets import Dataset as HFDataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=384):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 预编码所有数据
        self.encodings = self.tokenizer(
            [item['question'] for item in data],
            [item['context'] for item in data],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # 预计算所有答案位置
        self.start_positions = []
        self.end_positions = []
        for item in data:
            answer_start = item['context'].find(item['answer'])
            if answer_start == -1:
                self.start_positions.append(0)
                self.end_positions.append(0)
            else:
                start_pos = self.encodings.char_to_token(answer_start)
                end_pos = self.encodings.char_to_token(answer_start + len(item['answer']) - 1)
                if start_pos is None:
                    start_pos = 0
                if end_pos is None:
                    end_pos = 0
                start_pos = max(0, min(start_pos, self.max_length - 1))
                end_pos = max(0, min(end_pos, self.max_length - 1))
                if end_pos < start_pos:
                    end_pos = start_pos
                self.start_positions.append(start_pos)
                self.end_positions.append(end_pos)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'start_positions': torch.tensor(self.start_positions[idx], dtype=torch.long),
            'end_positions': torch.tensor(self.end_positions[idx], dtype=torch.long)
        }

def load_and_format_data(train_file, test_file):
    """加载并格式化训练和测试数据"""
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            
        # 确保数据格式正确
        formatted_train = []
        for item in train_data:
            if isinstance(item, list) and len(item) >= 3:
                formatted_train.append({
                    'question': item[0],
                    'context': item[1],
                    'answer': item[2]
                })
                
        formatted_test = []
        for item in test_data:
            if isinstance(item, list) and len(item) >= 3:
                formatted_test.append({
                    'question': item[0],
                    'context': item[1],
                    'answer': item[2]
                })
                
        return formatted_train, formatted_test
    except Exception as e:
        logger.error(f"加载数据时出错: {str(e)}")
        raise

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    start_preds, end_preds = predictions
    start_labels, end_labels = labels
    
    # 将预测结果转换为整数
    start_preds = np.argmax(start_preds, axis=1)
    end_preds = np.argmax(end_preds, axis=1)
    
    # 计算准确率
    start_acc = np.mean(start_preds == start_labels)
    end_acc = np.mean(end_preds == end_labels)
    
    # 计算F1分数
    def calculate_f1(preds, labels):
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    start_precision, start_recall, start_f1 = calculate_f1(start_preds, start_labels)
    end_precision, end_recall, end_f1 = calculate_f1(end_preds, end_labels)
    
    return {
        'start_accuracy': start_acc,
        'end_accuracy': end_acc,
        'start_f1': start_f1,
        'end_f1': end_f1,
        'start_precision': start_precision,
        'start_recall': start_recall,
        'end_precision': end_precision,
        'end_recall': end_recall
    }

def train_model(train_file, test_file, output_dir, model_name="bert-base-chinese"):
    """训练问答模型"""
    try:
        # 加载数据
        train_data, test_data = load_and_format_data(train_file, test_file)
        logger.info(f"加载了 {len(train_data)} 条训练数据和 {len(test_data)} 条测试数据")

        # 初始化tokenizer和模型
        local_model_path = os.path.join(os.path.dirname(__file__), '..', 'qa_model')

        if os.path.exists(local_model_path):
            logger.info(f"从本地路径加载模型: {local_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModelForQuestionAnswering.from_pretrained(local_model_path)
        else:
            logger.info(f"本地模型不存在 ({local_model_path})，尝试从Hugging Face Hub加载: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)

        # 创建数据集
        train_dataset = QADataset(train_data, tokenizer)
        test_dataset = QADataset(test_data, tokenizer)
        
        # 转换为HuggingFace数据集格式
        train_hf_dataset = HFDataset.from_dict({
            'input_ids': [item['input_ids'] for item in train_dataset],
            'attention_mask': [item['attention_mask'] for item in train_dataset],
            'start_positions': [item['start_positions'] for item in train_dataset],
            'end_positions': [item['end_positions'] for item in train_dataset]
        })
        
        test_hf_dataset = HFDataset.from_dict({
            'input_ids': [item['input_ids'] for item in test_dataset],
            'attention_mask': [item['attention_mask'] for item in test_dataset],
            'start_positions': [item['start_positions'] for item in test_dataset],
            'end_positions': [item['end_positions'] for item in test_dataset]
        })
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=32,  # 增加批次大小到32
            per_device_eval_batch_size=32,   # 评估批次也相应增加
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            save_steps=200,
            eval_steps=200,
            learning_rate=2e-5,
            gradient_accumulation_steps=1,  # 由于批次增大，减少梯度累积步数
            max_grad_norm=1.0,
            fp16=True,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        )
        
        # 初始化Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_hf_dataset,
            eval_dataset=test_hf_dataset,
            compute_metrics=compute_metrics,
        )
        
        # 开始训练
        logger.info("开始训练模型...")
        trainer.train()
        
        # 保存模型
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"模型已保存到 {output_dir}")
        
        # 评估模型
        logger.info("开始评估模型...")
        eval_results = trainer.evaluate()
        logger.info(f"评估结果: {eval_results}")
        
        return True
        
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    # 测试代码
    train_file = "knowledge_base/train.json"
    test_file = "knowledge_base/test.json"
    output_dir = "models/bert_qa"
    train_model(train_file, test_file, output_dir)