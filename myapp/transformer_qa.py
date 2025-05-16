import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import json
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QADataset(Dataset):
    def __init__(self, questions, contexts, answers=None, tokenizer=None, max_length=384):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.contexts[idx]
        
        # 使用tokenizer处理输入
        inputs = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 移除batch维度，因为Dataset是一次返回一个样本
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # 如果有答案（训练模式）
        if self.answers is not None:
            answer = self.answers[idx]
            answer_start = answer.get('answer_start', [0])[0]
            answer_text = answer.get('text', [''])[0]
            
            # 计算开始和结束位置的标记
            start_positions = []
            end_positions = []
            
            # 获取偏移映射，用于定位答案
            offset_mapping = inputs.pop("offset_mapping")
            
            # 寻找答案的开始和结束位置
            start_char = answer_start
            end_char = answer_start + len(answer_text)
            
            # 特殊情况处理：答案不在上下文中
            if start_char >= len(context) or end_char > len(context) or context[start_char:end_char] != answer_text:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # 通过偏移映射找到token位置
                token_start_index = 0
                while token_start_index < len(offset_mapping) and offset_mapping[token_start_index][0] <= start_char:
                    token_start_index += 1
                token_start_index -= 1
                
                token_end_index = token_start_index
                while token_end_index < len(offset_mapping) and offset_mapping[token_end_index][1] <= end_char:
                    token_end_index += 1
                
                # 进行边界检查
                if offset_mapping[token_start_index][0] > start_char or offset_mapping[token_end_index-1][1] < end_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    start_positions.append(token_start_index)
                    end_positions.append(token_end_index - 1)
            
            # 添加答案位置信息到输入
            inputs["start_positions"] = torch.tensor(start_positions, dtype=torch.long)[0]
            inputs["end_positions"] = torch.tensor(end_positions, dtype=torch.long)[0]
            
        return inputs

class TransformerQA:
    def __init__(self, model_name="distilbert-base-cased-distilled-squad", device=None, model_path=None):
        """初始化TransformerQA模型
        
        Args:
            model_name: 预训练模型名称
            device: 运行设备(cpu/cuda)
            model_path: 本地模型路径
        """
        self.model_name = model_name
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        try:
            # 优先尝试从本地路径加载
            if model_path and os.path.exists(model_path):
                logger.info(f"从 {model_path} 加载模型")
                
                # 首先检查该路径是否包含必要的文件
                config_file = os.path.join(model_path, 'config.json')
                model_file = os.path.join(model_path, 'pytorch_model.bin')
                
                if os.path.exists(config_file) and os.path.exists(model_file):
                    try:
                        # 加载tokenizer
                        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                        # 加载模型
                        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
                        self.model.to(self.device)
                        logger.info("本地模型加载成功")
                        return
                    except Exception as e:
                        logger.error(f"从本地路径加载模型失败: {str(e)}")
                else:
                    logger.warning(f"本地路径不包含完整模型文件，将使用预训练模型")
            
            # 如果本地加载失败，使用预训练模型
            logger.info(f"加载预训练模型: {model_name}")
            
            # 加载tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                logger.error(f"加载预训练tokenizer失败: {str(e)}")
                # 尝试使用后备模型
                backup_model = "distilbert-base-cased"
                logger.info(f"尝试使用后备模型tokenizer: {backup_model}")
                self.tokenizer = AutoTokenizer.from_pretrained(backup_model)
                self.model_name = backup_model
            
            # 加载模型
            try:
                self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            except Exception as e:
                logger.error(f"加载预训练模型失败: {str(e)}")
                # 尝试使用后备模型
                backup_model = "distilbert-base-cased"
                logger.info(f"尝试使用后备模型: {backup_model}")
                self.model = AutoModelForQuestionAnswering.from_pretrained(backup_model)
                self.model_name = backup_model
            
            self.model.to(self.device)
            logger.info("预训练模型加载成功")
            
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise RuntimeError(f"无法初始化QA模型: {str(e)}")
    
    def train(self, train_questions, train_contexts, train_answers, 
              val_questions=None, val_contexts=None, val_answers=None,
              batch_size=8, epochs=3, learning_rate=5e-5, save_path="qa_model"):
        """训练问答模型"""
        # 准备训练数据
        train_dataset = QADataset(
            questions=train_questions,
            contexts=train_contexts,
            answers=train_answers,
            tokenizer=self.tokenizer
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 准备验证数据（如果有的话）
        val_loader = None
        if val_questions and val_contexts and val_answers:
            val_dataset = QADataset(
                questions=val_questions,
                contexts=val_contexts,
                answers=val_answers,
                tokenizer=self.tokenizer
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 设置优化器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # 计算总训练步数
        total_steps = len(train_loader) * epochs
        
        # 创建学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # 开始训练循环
        self.model.train()
        for epoch in range(epochs):
            logger.info(f"开始 Epoch {epoch+1}/{epochs}")
            total_loss = 0
            
            # 训练阶段
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch in progress_bar:
                # 将数据移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # 参数更新
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # 更新进度条
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # 计算并记录平均损失
            avg_train_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} 平均训练损失: {avg_train_loss:.4f}")
            
            # 验证阶段
            if val_loader:
                self.model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validating"):
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        outputs = self.model(**batch)
                        val_loss += outputs.loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                logger.info(f"Epoch {epoch+1} 验证损失: {avg_val_loss:.4f}")
                
                # 切回训练模式
                self.model.train()
        
        # 保存模型
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"模型保存到 {save_path}")
        
        return self.model
    
    def predict(self, question, context, max_length=384, stride=128, top_k=1):
        """使用模型预测问题的答案"""
        # 参数检查
        if not question or not question.strip():
            logger.warning("提供的问题为空")
            return []
            
        if not context or not context.strip():
            logger.warning("提供的上下文为空")
            return []
            
        try:
            # 切换到评估模式
            self.model.eval()
            
            # 使用tokenizer处理输入
            inputs = self.tokenizer(
                question,
                context,
                max_length=max_length,
                truncation="only_second",
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # 获取偏移映射，并移动到CPU
            offset_mapping = inputs.pop("offset_mapping")
            offset_mapping = offset_mapping.cpu().numpy()
            
            # 获取样本映射（如果上下文被分成多个部分）
            sample_mapping = inputs.pop("overflow_to_sample_mapping").cpu().numpy()
            
            # 保存一份input_ids供后面使用，确保在移动到设备前就保存
            input_ids = inputs["input_ids"].clone().cpu().numpy() if "input_ids" in inputs else None
            
            # 将输入移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 使用模型进行预测
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 获取开始和结束logits
            start_logits = outputs.start_logits.cpu().numpy()
            end_logits = outputs.end_logits.cpu().numpy()
            
            # 处理结果
            all_answers = []
            
            # 对预测的开始和结束位置进行后处理
            for i in range(len(start_logits)):
                # 获取当前样本在原始文本中的偏移
                current_offset = offset_mapping[i]
                
                # 创建分类器标签列表（用于过滤特殊标记的位置）
                # 检查input_ids是否存在
                if input_ids is None:
                    logger.warning("input_ids不存在，使用安全默认值")
                    cls_index = 0
                else:
                    # 使用保存的input_ids而不是从inputs字典中获取
                    cls_index = np.where(input_ids[i] == self.tokenizer.cls_token_id)[0][0]
                
                # 得到开始和结束位置的预测分数
                start_indexes = self._get_best_indexes(start_logits[i], n_best_size=20)
                end_indexes = self._get_best_indexes(end_logits[i], n_best_size=20)
                
                # 遍历所有可能的开始和结束位置对
                valid_answers = []
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # 跳过无效的位置：
                        # - 位置对应的是填充标记
                        # - 结束位置在开始位置之前
                        # - 答案长度超过了最大值
                        if (
                            start_index >= len(current_offset)
                            or end_index >= len(current_offset)
                            or current_offset[start_index] is None
                            or current_offset[end_index] is None
                            or end_index < start_index
                            or end_index - start_index + 1 > 50
                        ):
                            continue
                        
                        # 不考虑CLS标记（答案不应该从CLS标记开始或结束）
                        if start_index == cls_index or end_index == cls_index:
                            continue
                        
                        # 获取文本中的开始和结束字符位置
                        start_char = current_offset[start_index][0]
                        end_char = current_offset[end_index][1]
                        
                        # 如果位置无效则跳过
                        if start_char is None or end_char is None or start_char >= len(context) or end_char > len(context):
                            continue
                            
                        # 计算分数并添加到有效答案列表
                        answer_text = context[start_char:end_char]
                        if not answer_text.strip():
                            continue
                            
                        score = start_logits[i][start_index] + end_logits[i][end_index]
                        valid_answers.append({
                            'text': answer_text,
                            'start': start_char,
                            'end': end_char,
                            'score': float(score)
                        })
                
                # 添加有效答案到总结果
                all_answers.extend(valid_answers)
            
            # 如果没有找到有效答案
            if not all_answers:
                logger.warning("模型未能找到有效答案")
                # 返回一个特殊的答案，表示没有找到
                return [{
                    'text': f"我无法在提供的上下文中找到关于'{question}'的具体答案。",
                    'start': 0,
                    'end': 0,
                    'score': 0.0
                }]
            
            # 对所有答案进行去重和排序
            unique_answers = {}
            for answer in all_answers:
                answer_text = answer['text']
                # 对答案文本进行清理
                answer_text = answer_text.strip()
                
                # 跳过空答案或过短的答案
                if len(answer_text) < 2:
                    continue
                
                # 只保留每个答案文本的最高分数
                if answer_text not in unique_answers or answer['score'] > unique_answers[answer_text]['score']:
                    unique_answers[answer_text] = answer
            
            # 转换为列表并按分数排序
            sorted_answers = list(unique_answers.values())
            sorted_answers.sort(key=lambda x: x['score'], reverse=True)
            
            # 返回top_k个结果
            return sorted_answers[:top_k]
            
        except Exception as e:
            logger.error(f"预测过程中出错: {str(e)}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            # 出错时返回一个友好的错误信息
            return [{
                'text': "很抱歉，在处理您的问题时出现了技术问题。请稍后再试或联系系统管理员。",
                'start': 0,
                'end': 0,
                'score': 0.0
            }]
    
    def _get_best_indexes(self, logits, n_best_size):
        """获取logits中分数最高的几个位置"""
        # 为防止出现NaN或无穷大的值
        logits = np.nan_to_num(logits)
        
        # 获取排序后的索引
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
        
        # 取前n_best_size个
        best_indexes = [x[0] for x in index_and_score[:n_best_size]]
        return best_indexes

    def save(self, save_path):
        """保存模型和tokenizer"""
        try:
            logger.info(f"开始保存模型到 {save_path}")
            os.makedirs(save_path, exist_ok=True)
            
            # 直接使用内置方法保存
            logger.info("保存模型...")
            self.model.save_pretrained(save_path)
            
            logger.info("保存tokenizer...")
            self.tokenizer.save_pretrained(save_path)
            
            # 验证保存成功
            if os.path.exists(os.path.join(save_path, "config.json")):
                logger.info(f"模型配置成功保存到 {save_path}")
                # 如果模型文件不存在，尝试直接保存状态字典
                model_file = os.path.join(save_path, "pytorch_model.bin")
                if not os.path.exists(model_file):
                    logger.warning("未找到模型文件，尝试直接保存状态字典")
                    torch.save(self.model.state_dict(), model_file)
                
                # 再次检查
                if os.path.exists(model_file):
                    logger.info(f"模型权重成功保存到 {save_path}")
                    return True
                else:
                    logger.error(f"模型权重保存失败")
                    return False
            else:
                logger.error(f"模型配置保存失败")
                return False
                
        except Exception as e:
            logger.error(f"保存模型时出错: {str(e)}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return False
    
    def load(self, load_path):
        """加载预训练的模型和tokenizer"""
        if os.path.exists(load_path):
            logger.info(f"从 {load_path} 加载模型")
            self.model = AutoModelForQuestionAnswering.from_pretrained(load_path)
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            self.model.to(self.device)
            return True
        else:
            logger.error(f"模型路径 {load_path} 不存在")
            return False 