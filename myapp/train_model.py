"""
模型训练脚本
用于训练基于Transformer的问答模型
"""

import os
import json
import logging
import argparse
from .transformer_qa import TransformerQA

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_squad_data(squad_file, split_ratio=0.9):
    """准备SQuAD格式的数据进行训练"""
    logger.info(f"从文件加载SQuAD数据: {squad_file}")
    
    with open(squad_file, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)
    
    questions = []
    contexts = []
    answers = []
    
    # 处理SQuAD格式的数据
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                if not qa["is_impossible"]:
                    for answer in qa["answers"]:
                        questions.append(question)
                        contexts.append(context)
                        answers.append({
                            "text": [answer["text"]],
                            "answer_start": [answer["answer_start"]]
                        })
                        break  # 只取第一个答案
    
    # 划分训练集和验证集
    num_samples = len(questions)
    train_size = int(num_samples * split_ratio)
    
    train_questions = questions[:train_size]
    train_contexts = contexts[:train_size]
    train_answers = answers[:train_size]
    
    val_questions = questions[train_size:]
    val_contexts = contexts[train_size:]
    val_answers = answers[train_size:]
    
    logger.info(f"准备了 {len(train_questions)} 个训练样本和 {len(val_questions)} 个验证样本")
    
    return train_questions, train_contexts, train_answers, val_questions, val_contexts, val_answers

def prepare_custom_data(data_file, split_ratio=0.9):
    """准备自定义格式的训练数据"""
    logger.info(f"从文件加载自定义数据: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = []
    contexts = []
    answers = []
    
    # 处理自定义格式的数据
    for item in data:
        questions.append(item["question"])
        contexts.append(item["context"])
        answers.append({
            "text": [item["answer"]],
            "answer_start": [item["context"].find(item["answer"])]
        })
    
    # 检查并修复找不到答案的情况
    for i, answer in enumerate(answers):
        if answer["answer_start"][0] == -1:
            logger.warning(f"在上下文中找不到答案: {questions[i]}")
            # 尝试用更宽松的方式查找
            context_lower = contexts[i].lower()
            answer_lower = answer["text"][0].lower()
            new_start = context_lower.find(answer_lower)
            if new_start != -1:
                answers[i]["answer_start"][0] = new_start
                logger.info(f"通过不区分大小写找到了答案，位置: {new_start}")
            else:
                # 如果仍然找不到，可以跳过这个样本
                logger.warning(f"跳过样本: {questions[i]}")
                questions[i] = None
    
    # 过滤掉无效样本
    filtered_data = [(q, c, a) for q, c, a in zip(questions, contexts, answers) if q is not None]
    questions, contexts, answers = zip(*filtered_data) if filtered_data else ([], [], [])
    
    # 划分训练集和验证集
    num_samples = len(questions)
    train_size = int(num_samples * split_ratio)
    
    train_questions = questions[:train_size]
    train_contexts = contexts[:train_size]
    train_answers = answers[:train_size]
    
    val_questions = questions[train_size:]
    val_contexts = contexts[train_size:]
    val_answers = answers[train_size:]
    
    logger.info(f"准备了 {len(train_questions)} 个训练样本和 {len(val_questions)} 个验证样本")
    
    return train_questions, train_contexts, train_answers, val_questions, val_contexts, val_answers

def create_synthetic_data_from_knowledge_base(kb_dir, output_file, num_samples=100):
    """从知识库创建合成训练数据"""
    import random
    from .knowledge_manager import KnowledgeManager
    
    logger.info(f"从知识库创建合成数据: {kb_dir}")
    
    # 初始化知识库管理器
    km = KnowledgeManager()
    km.load_documents_from_directory(kb_dir)
    
    # 分块处理文档
    chunks = km.chunk_documents(chunk_size=300, chunk_overlap=50)  # 更小的块大小
    
    # 模拟问题和答案
    data = []
    chunk_items = list(chunks.items())
    
    # 确保有足够的分块
    if len(chunk_items) < num_samples:
        logger.warning(f"知识库分块数量({len(chunk_items)})小于请求的样本数({num_samples})，将生成所有可能的样本")
        num_samples = len(chunk_items)
    
    # 随机选择分块生成问答对
    selected_chunks = random.sample(chunk_items, num_samples)
    
    for i, (chunk_id, content) in enumerate(selected_chunks):
        # 选择一个简短的句子作为"答案"
        sentences = content.split('。')
        sentences = [s + '。' for s in sentences if s.strip() and len(s) < 100]  # 选择较短的句子
        if not sentences:
            continue
        
        # 优先选择短句子
        short_sentences = [s for s in sentences if len(s) < 50 and len(s) > 10]
        if short_sentences:
            answer_sentence = random.choice(short_sentences)
        else:
            answer_sentence = random.choice(sentences)
            
        answer_start = content.find(answer_sentence)
        
        if answer_start == -1:
            continue
        
        # 使用更简单的问题模板
        question_templates = [
            "什么是{}？",
            "请解释{}",
            "{}是什么？",
            "介绍一下{}"
        ]
        
        # 提取主题词 - 简化逻辑
        words = answer_sentence.split()[:3]  # 只取前几个词
        if not words:
            continue
        
        # 选择一个词作为焦点
        focus_word = words[0] if words else ""
        if not focus_word:
            continue
            
        # 去除标点符号
        focus_word = focus_word.strip('.,;:?!，。；：？！')
        if not focus_word:
            continue
        
        # 生成问题
        question = random.choice(question_templates).format(focus_word)
        
        # 添加到数据集
        data.append({
            "question": question,
            "context": content[:500],  # 限制上下文长度
            "answer": answer_sentence,
            "answer_start": answer_start
        })
    
    # 保存合成数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"生成了 {len(data)} 个合成样本并保存到: {output_file}")
    
    return output_file

def parse_qa_text_format(file_path):
    """解析问：答：格式的纯文本文件"""
    logger.info(f"从文本文件加载问答对数据: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 分割成问答对
        qa_pairs = []
        parts = content.strip().split('问：')
        parts = [p for p in parts if p.strip()]  # 去除空元素
        
        for part in parts:
            try:
                # 分割问题和答案
                qa_split = part.strip().split('答：', 1)
                if len(qa_split) == 2:
                    question = qa_split[0].strip()
                    answer = qa_split[1].strip()
                    qa_pairs.append({"question": question, "answer": answer})
            except Exception as e:
                logger.warning(f"处理问答对时出错: {part}, 错误: {str(e)}")
        
        logger.info(f"从文本文件中解析出 {len(qa_pairs)} 个问答对")
        return qa_pairs
    except Exception as e:
        logger.error(f"解析文本文件时出错: {str(e)}")
        return []

def prepare_qa_pairs_data(data_file, split_ratio=0.9):
    """准备一问一答格式的训练数据"""
    logger.info(f"从文件加载问答对数据: {data_file}")
    
    data = []
    # 检查文件扩展名
    file_ext = os.path.splitext(data_file)[1].lower()
    
    try:
        if file_ext == '.txt':
            # 对于txt文件，尝试解析特殊格式（问：答：）
            data = parse_qa_text_format(data_file)
        elif file_ext == '.json':
            # 对于JSON文件，使用原有逻辑
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                # 尝试以行分隔的格式读取
                try:
                    with open(data_file, 'r', encoding='utf-8') as f:
                        data = []
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    item = json.loads(line)
                                    data.append(item)
                                except json.JSONDecodeError:
                                    logger.warning(f"跳过无效行: {line}")
                except Exception as e:
                    logger.error(f"读取文件失败: {str(e)}")
                    return [], [], [], [], [], []
        else:
            logger.warning(f"不支持的文件格式: {file_ext}")
    except Exception as e:
        logger.error(f"读取文件失败: {str(e)}")
        return [], [], [], [], [], []
    
    if not data:
        logger.warning("数据文件为空或格式无效")
        return [], [], [], [], [], []
    
    # 后续处理逻辑保持不变
    questions = []
    contexts = []
    answers = []
    
    # 处理问答对数据
    for item in data:
        # 支持多种可能的字段名称
        question = item.get('question', item.get('query', item.get('问题', '')))
        answer_text = item.get('answer', item.get('response', item.get('回答', '')))
        
        # 如果没有额外上下文，使用问题作为上下文
        context = item.get('context', item.get('上下文', question))
        
        if not question or not answer_text:
            logger.warning(f"跳过无效数据: {item}")
            continue
        
        # 在上下文中定位答案
        answer_start = context.find(answer_text)
        
        # 如果在上下文中找不到答案，使用简单拼接
        if answer_start == -1:
            context = f"{question} {answer_text}"
            answer_start = context.find(answer_text)
        
        questions.append(question)
        contexts.append(context)
        answers.append({
            "text": [answer_text],
            "answer_start": [answer_start]
        })
    
    logger.info(f"从文件中加载了 {len(questions)} 个问答对")
    
    # 划分训练集和验证集
    num_samples = len(questions)
    train_size = int(num_samples * split_ratio)
    
    train_questions = questions[:train_size]
    train_contexts = contexts[:train_size]
    train_answers = answers[:train_size]
    
    val_questions = questions[train_size:]
    val_contexts = contexts[train_size:]
    val_answers = answers[train_size:]
    
    logger.info(f"准备了 {len(train_questions)} 个训练样本和 {len(val_questions)} 个验证样本")
    
    return train_questions, train_contexts, train_answers, val_questions, val_contexts, val_answers

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练Transformer问答模型")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="要使用的预训练模型名称")
    parser.add_argument("--data_file", type=str, required=False,
                        help="训练数据文件路径")
    parser.add_argument("--data_format", type=str, choices=["squad", "custom", "qa_pairs"], default="custom",
                        help="数据格式，可以是'squad'、'custom'或'qa_pairs'")
    parser.add_argument("--output_dir", type=str, default="qa_model",
                        help="保存模型的目录")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="训练批量大小")
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="学习率")
    parser.add_argument("--kb_dir", type=str, help="知识库目录，用于生成合成训练数据")
    parser.add_argument("--synthetic_samples", type=int, default=100,
                        help="生成的合成样本数量")
    
    args = parser.parse_args()
    
    # 检查是否指定了数据来源
    if not args.data_file and not args.kb_dir:
        logger.error("必须指定--data_file或--kb_dir参数")
        return
    
    # 如果指定了知识库目录，则生成合成数据
    if args.kb_dir:
        synthetic_data_file = os.path.join(args.output_dir, "synthetic_data.json")
        os.makedirs(args.output_dir, exist_ok=True)
        data_file = create_synthetic_data_from_knowledge_base(
            args.kb_dir, synthetic_data_file, args.synthetic_samples
        )
        data_format = "custom"
    else:
        data_file = args.data_file
        data_format = args.data_format
    
    # 准备训练数据
    if data_format == "squad":
        train_questions, train_contexts, train_answers, val_questions, val_contexts, val_answers = prepare_squad_data(data_file)
    elif data_format == "custom":
        train_questions, train_contexts, train_answers, val_questions, val_contexts, val_answers = prepare_custom_data(data_file)
    elif data_format == "qa_pairs":
        train_questions, train_contexts, train_answers, val_questions, val_contexts, val_answers = prepare_qa_pairs_data(data_file)
    
    # 初始化模型
    qa_model = TransformerQA(model_name=args.model_name)
    
    # 训练模型
    qa_model.train(
        train_questions=train_questions,
        train_contexts=train_contexts,
        train_answers=train_answers,
        val_questions=val_questions,
        val_contexts=val_contexts,
        val_answers=val_answers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_path=args.output_dir
    )
    
    logger.info(f"训练完成。模型保存在: {args.output_dir}")

if __name__ == "__main__":
    main() 