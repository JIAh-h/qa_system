import os
import json
import logging
import tempfile
import time
import shutil
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from .transformer_qa import TransformerQA
from .knowledge_manager import KnowledgeManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 模型配置
MODEL_CONFIG = {
    'MODEL_NAME': 'distilbert-base-cased-distilled-squad',  # 更小的、已针对SQuAD微调的模型
    'MODEL_PATH': os.path.join(settings.BASE_DIR, 'qa_model'),  # 模型保存/加载路径
    'KB_DIR': os.path.join(settings.BASE_DIR, 'knowledge_base'),  # 知识库路径放在qa_system主目录下
    'TEMP_MODEL_PATH': os.path.join(tempfile.gettempdir(), 'qa_temp_model')  # 临时模型保存路径
}

# 初始化全局对象
qa_model = None
knowledge_manager = None

def init_qa_system():
    """初始化问答系统"""
    global qa_model, knowledge_manager
    
    # 确保模型目录存在
    os.makedirs(MODEL_CONFIG['MODEL_PATH'], exist_ok=True)
    
    # 初始化知识库管理器
    logger.info("初始化知识库管理器...")
    knowledge_manager = KnowledgeManager()
    
    # 加载知识库文档
    if os.path.exists(MODEL_CONFIG['KB_DIR']):
        logger.info(f"从目录加载知识库文档: {MODEL_CONFIG['KB_DIR']}")
        knowledge_manager.load_documents_from_directory(MODEL_CONFIG['KB_DIR'])
        knowledge_manager.encode_documents()
    else:
        logger.warning(f"知识库目录不存在: {MODEL_CONFIG['KB_DIR']}")
    
    # 初始化问答模型
    logger.info("初始化问答模型...")
    qa_model = TransformerQA(model_name=MODEL_CONFIG['MODEL_NAME'])
    
    # 检查模型是否有效 - 不仅检查目录是否存在，还检查config.json是否存在
    model_config_path = os.path.join(MODEL_CONFIG['MODEL_PATH'], 'config.json')
    if os.path.exists(model_config_path):
        logger.info(f"尝试加载已训练的模型: {MODEL_CONFIG['MODEL_PATH']}")
        try:
            qa_model.load(MODEL_CONFIG['MODEL_PATH'])
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            logger.info(f"将使用预训练模型: {MODEL_CONFIG['MODEL_NAME']}")
    else:
        logger.warning(f"模型配置不存在，将使用预训练模型: {MODEL_CONFIG['MODEL_NAME']}")

# 第一次导入模块时初始化系统
init_qa_system()

@csrf_exempt
def qa_view(request):
    """
    处理问答请求的视图函数
    """
    if request.method == 'POST':
        try:
            # 获取问题内容
            question = request.POST.get('question', '').strip()
            
            if not question:
                logger.warning("接收到空问题")
                return JsonResponse({
                    'error': '问题不能为空',
                    'status': 'error'
                }, status=400)

            # 记录问题
            logger.info(f"接收到问题: {question}")
            
            # 检查知识库和模型是否已初始化
            global qa_model, knowledge_manager
            if knowledge_manager is None:
                logger.error("知识库未初始化，尝试重新初始化")
                knowledge_manager = KnowledgeManager()
                if knowledge_manager is None:
                    return JsonResponse({
                        'error': '无法初始化知识库管理器',
                        'status': 'error'
                    }, status=500)
            
            if qa_model is None:
                logger.error("问答模型未初始化，尝试加载预训练模型")
                try:
                    qa_model = TransformerQA(model_name="distilbert-base-cased-distilled-squad")
                except Exception as e:
                    logger.error(f"加载预训练模型失败: {str(e)}")
                    return JsonResponse({
                        'error': '无法加载问答模型，请稍后再试',
                        'status': 'error'
                    }, status=500)
            
            # 首先尝试在datas.txt中查找问题的精确匹配
            kb_file = os.path.join(MODEL_CONFIG['KB_DIR'], 'datas.txt')
            if os.path.exists(kb_file):
                try:
                    with open(kb_file, 'r', encoding='utf-8') as f:
                        data_content = f.read()
                    
                    # 搜索与当前问题相似的"问："段落
                    qa_pairs = []
                    parts = data_content.strip().split('问：')
                    parts = [p for p in parts if p.strip()]
                    
                    best_match = None
                    best_match_score = 0
                    best_answer = None
                    
                    for part in parts:
                        qa_split = part.strip().split('答：', 1)
                        if len(qa_split) == 2:
                            q = qa_split[0].strip()
                            a = qa_split[1].strip()
                            
                            # 使用改进的问题相似度计算
                            score = knowledge_manager._calculate_question_similarity(q, question)
                            
                            # 如果问题完全匹配
                            if q == question:
                                best_match = q
                                best_answer = a
                                best_match_score = 1.0
                                break
                            
                            # 如果找到更好的匹配
                            if score > best_match_score:
                                best_match = q
                                best_answer = a
                                best_match_score = score
                    
                    # 如果找到了相似度足够高的问题，直接返回对应答案
                    if best_match and best_match_score > 0.3:
                        logger.info(f"从datas.txt中找到匹配问题: {best_match}, 相似度: {best_match_score}")
                        
                        # 如果答案包含多个句子，只返回第一句
                        sentences = best_answer.split('。')
                        if len(sentences) > 1:
                            best_answer = sentences[0].strip() + '。'
                            logger.info(f"只返回第一句答案: {best_answer}")
                        
                        return JsonResponse({
                            'answer': best_answer,
                            'status': 'direct_match'
                        })
                        
                except Exception as e:
                    logger.error(f"读取datas.txt文件失败: {str(e)}")
            
            # 检查知识库是否为空
            if not knowledge_manager.documents:
                logger.warning("知识库为空，尝试重新加载")
                # 尝试重新加载知识库
                try:
                    knowledge_manager.load_documents_from_directory(MODEL_CONFIG['KB_DIR'])
                    if not knowledge_manager.documents:
                        # 如果知识库仍然为空，使用直接回答模式
                        logger.warning("知识库仍然为空，切换到直接回答模式")
                        # 直接使用模型回答问题，不依赖知识库
                        direct_answer = direct_answer_question(question)
                        return JsonResponse({
                            'answer': direct_answer,
                            'status': 'direct_answer'
                        })
                    knowledge_manager.encode_documents()
                except Exception as e:
                    logger.error(f"重新加载知识库失败: {str(e)}")
                    # 尝试直接回答
                    direct_answer = direct_answer_question(question)
                    return JsonResponse({
                        'answer': direct_answer,
                        'status': 'direct_answer'
                    })
            
            # 获取与问题相关的上下文
            try:
                context = knowledge_manager.get_context_for_question(question, top_k=1)  # 只获取最相关的一个
            except Exception as e:
                logger.error(f"获取上下文失败: {str(e)}")
                # 尝试直接回答
                direct_answer = direct_answer_question(question)
                return JsonResponse({
                    'answer': direct_answer,
                    'status': 'direct_answer'
                })
            
            if not context:
                logger.warning("未找到相关上下文，尝试直接回答")
                direct_answer = direct_answer_question(question)
                return JsonResponse({
                    'answer': direct_answer,
                    'status': 'direct_answer'
                })
            
            # 使用问答模型预测答案
            logger.info("使用问答模型预测答案...")
            try:
                answer_results = qa_model.predict(question, context)
            except Exception as e:
                logger.error(f"预测答案失败: {str(e)}")
                # 尝试直接回答
                direct_answer = direct_answer_question(question)
                return JsonResponse({
                    'answer': direct_answer,
                    'status': 'direct_answer'
                })
            
            if not answer_results:
                logger.warning("模型未能给出答案，尝试直接回答")
                direct_answer = direct_answer_question(question)
                return JsonResponse({
                    'answer': direct_answer,
                    'status': 'direct_answer'
                })
            
            # 获取最佳答案
            best_answer = answer_results[0]['text']
            
            # 检查并清理答案格式 - 移除问答格式标记
            if '答：' in best_answer:
                best_answer = best_answer.split('答：', 1)[-1].strip()
            if '问：' in best_answer:
                best_answer = best_answer.split('问：', 1)[-1].strip()
            
            # 如果答案包含多个句子，只返回第一句
            sentences = best_answer.split('。')
            if len(sentences) > 1:
                best_answer = sentences[0].strip() + '。'
                logger.info(f"模型预测答案太长，只返回第一句: {best_answer}")
            
            # 返回答案
            return JsonResponse({
                'answer': best_answer,
                'context': context,  # 可选：也返回上下文，便于调试或展示
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"处理请求时出错: {str(e)}", exc_info=True)
            return JsonResponse({
                'error': str(e),
                'answer': '对不起，处理您的请求时出现了错误。请稍后再试。',
                'status': 'error'
            }, status=500)
            
    # GET请求返回页面
    return render(request, 'qa.html')

def direct_answer_question(question):
    """不使用检索，直接回答基本问题"""
    question = question.lower().strip()
    
    # 尝试从datas.txt查找匹配问题
    kb_file = os.path.join(MODEL_CONFIG['KB_DIR'], 'datas.txt')
    if os.path.exists(kb_file):
        try:
            with open(kb_file, 'r', encoding='utf-8') as f:
                data_content = f.read()
            
            # 搜索问题匹配
            parts = data_content.strip().split('问：')
            parts = [p for p in parts if p.strip()]
            
            best_match = None
            best_match_score = 0
            best_answer = None
            
            # 导入计算相似度的函数
            from .knowledge_manager import KnowledgeManager
            temp_km = KnowledgeManager()
            
            for part in parts:
                qa_split = part.strip().split('答：', 1)
                if len(qa_split) == 2:
                    q = qa_split[0].strip().lower()
                    a = qa_split[1].strip()
                    
                    # 使用改进的相似度计算
                    score = temp_km._calculate_question_similarity(q, question)
                    
                    # 如果找到更好的匹配
                    if score > best_match_score:
                        best_match = q
                        best_answer = a
                        best_match_score = score
            
            # 如果找到了相似度足够高的问题，直接返回对应答案
            if best_match and best_match_score > 0.3:  # 使用较低的阈值以提高召回率
                logger.info(f"直接回答从datas.txt中找到匹配问题: {best_match}, 相似度: {best_match_score}")
                
                # 确保清理任何额外的标记
                if '答：' in best_answer:
                    best_answer = best_answer.split('答：', 1)[-1].strip()
                if '问：' in best_answer:
                    best_answer = best_answer.split('问：', 1)[-1].strip()
                
                # 移除所有可能的问答标记（针对包含多个问答对的情况）
                import re
                best_answer = re.sub(r'问：.*?答：', '', best_answer)
                
                # 如果答案包含多个句子，只返回第一句
                sentences = best_answer.split('。')
                if len(sentences) > 1:
                    first_sentence = sentences[0].strip() + '。'
                    logger.info(f"只返回第一句答案: {first_sentence}")
                    return first_sentence
                    
                return best_answer
                
        except Exception as e:
            logger.error(f"直接回答读取datas.txt文件失败: {str(e)}")
    
    # 常见问题的预设回答
    qa_pairs = {
        "你好": "你好！我是基于Transformer的AI小助手，有什么我可以帮助您的吗？",
        "你是谁": "我是基于Transformer的AI小助手，使用本地知识库和预训练模型来回答您的问题。",
        "什么是python": "Python是一种高级编程语言，具有简洁易读的语法，广泛应用于数据分析、人工智能、Web开发和自动化脚本等领域。",
        "什么是django": "Django是一个基于Python的Web框架，遵循MVC架构模式，内置ORM系统，自带管理后台，安全性高。",
        "什么是transformer": "Transformer是一种基于注意力机制的深度学习模型，最初由Google于2017年提出，用于自然语言处理任务。",
        "什么是问答系统": "问答系统是一种能够理解自然语言问题并给出相应答案的人工智能系统，通常包括问题理解、信息检索、答案生成和答案排序等步骤。"
    }
    
    # 使用相同的模糊匹配逻辑检查预设问题
    if temp_km:
        best_match = None
        best_match_score = 0
        best_answer = None
        
        for q, a in qa_pairs.items():
            score = temp_km._calculate_question_similarity(q, question)
            if score > best_match_score:
                best_match = q
                best_answer = a
                best_match_score = score
        
        if best_match and best_match_score > 0.5:  # 预设问题要求更高的匹配度
            return best_answer
    else:
        # 简单匹配作为后备
        for key, answer in qa_pairs.items():
            if key in question or question in key:
                return answer
    
    # 对于其他问题，给出一个通用回答
    return "对于这个问题，我需要查询知识库才能给出准确答案，但目前无法访问知识库。"

@csrf_exempt
def train_model_view(request):
    """
    训练模型的视图函数 - 支持上传一问一答数据集进行训练
    """
    global qa_model
    
    if request.method == 'POST':
        try:
            # 检查是否有上传的文件
            if 'qa_dataset' in request.FILES:
                # 处理上传的问答数据集
                uploaded_file = request.FILES['qa_dataset']
                logger.info(f"接收到上传的问答数据集: {uploaded_file.name}, 大小: {uploaded_file.size} 字节")
                
                # 创建临时目录保存上传的文件和模型
                temp_model_path = MODEL_CONFIG['TEMP_MODEL_PATH']
                os.makedirs(temp_model_path, exist_ok=True)
                
                # 确定文件扩展名
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                if not file_ext:  # 如果没有扩展名
                    if any(marker in uploaded_file.read(1024).decode('utf-8', errors='ignore') for marker in ["问：", "答："]):
                        file_ext = '.txt'  # 如果包含"问："和"答："，作为txt处理
                    else:
                        # 尝试解析为JSON
                        try:
                            uploaded_file.seek(0)
                            json.loads(uploaded_file.read().decode('utf-8', errors='ignore'))
                            file_ext = '.json'
                        except:
                            file_ext = '.txt'  # 默认作为txt处理
                
                # 保存上传的文件
                dataset_path = os.path.join(temp_model_path, f"qa_dataset_{int(time.time())}{file_ext}")
                with open(dataset_path, 'wb') as f:
                    uploaded_file.seek(0)
                    for chunk in uploaded_file.chunks():
                        f.write(chunk)
                
                logger.info(f"问答数据集已保存到: {dataset_path}")
                
                # 从train_model.py导入处理函数
                from .train_model import prepare_qa_pairs_data
                
                # 准备训练数据
                logger.info("准备训练数据...")
                train_questions, train_contexts, train_answers, val_questions, val_contexts, val_answers = \
                    prepare_qa_pairs_data(dataset_path)
                
                if not train_questions:
                    return JsonResponse({
                        'error': '数据集解析失败或没有有效的训练样本',
                        'status': 'error'
                    }, status=400)
                
                logger.info(f"成功解析 {len(train_questions)} 个训练样本和 {len(val_questions)} 个验证样本")
                
                # 设置较小的批量大小，避免内存问题
                batch_size = min(4, max(1, len(train_questions) // 10))  # 动态调整批量大小
                epochs = 1  # 一个训练轮次
                
                # 初始化模型
                if qa_model is None:
                    qa_model = TransformerQA(model_name="distilbert-base-cased-distilled-squad")
                
                # 训练模型
                logger.info(f"开始训练模型，批量大小: {batch_size}，训练轮次: {epochs}")
                try:
                    qa_model.train(
                        train_questions=train_questions,
                        train_contexts=train_contexts,
                        train_answers=train_answers,
                        val_questions=val_questions,
                        val_contexts=val_contexts,
                        val_answers=val_answers,
                        batch_size=batch_size,
                        epochs=epochs,
                        save_path=temp_model_path
                    )
                    
                    # 确保手动调用保存方法，确保生成pytorch_model.bin
                    if not os.path.exists(os.path.join(temp_model_path, "pytorch_model.bin")):
                        logger.warning("训练后未发现模型文件，尝试手动保存")
                        save_result = qa_model.save(temp_model_path)
                        if save_result:
                            logger.info("手动保存模型成功")
                        else:
                            logger.error("手动保存模型失败")
                            return JsonResponse({
                                'error': '保存模型失败，请联系管理员',
                                'status': 'error'
                            }, status=500)
                except Exception as e:
                    logger.error(f"训练模型失败: {str(e)}", exc_info=True)
                    return JsonResponse({
                        'error': f'训练过程中出错: {str(e)}',
                        'status': 'error'
                    }, status=500)
                
                # 检查模型文件是否生成
                model_file = os.path.join(temp_model_path, "pytorch_model.bin")
                config_file = os.path.join(temp_model_path, "config.json")
                
                if not os.path.exists(model_file) or not os.path.exists(config_file):
                    logger.error("训练完成但模型文件未生成")
                    return JsonResponse({
                        'error': '训练完成但模型文件未生成，请尝试使用较小的数据集或更小的批量大小',
                        'status': 'error'
                    }, status=500)
                
                # 尝试复制模型到正式目录
                try:
                    target_model_path = MODEL_CONFIG['MODEL_PATH']
                    os.makedirs(target_model_path, exist_ok=True)
                    
                    for file_name in os.listdir(temp_model_path):
                        if file_name.endswith('.json') or file_name.endswith('.bin') or file_name.endswith('.txt'):
                            src = os.path.join(temp_model_path, file_name)
                            dst = os.path.join(target_model_path, file_name)
                            try:
                                shutil.copy2(src, dst)
                                logger.info(f"已复制模型文件: {file_name}")
                            except Exception as e:
                                logger.warning(f"复制文件失败: {file_name}, 错误: {str(e)}")
                except Exception as e:
                    logger.error(f"复制模型文件失败: {str(e)}")
                
                # 重新加载模型
                try:
                    qa_model.load(temp_model_path)
                    logger.info("已重新加载训练好的模型")
                except Exception as e:
                    logger.error(f"重新加载模型失败: {str(e)}")
                
                return JsonResponse({
                    'success': True,
                    'message': f'模型训练完成，共处理了 {len(train_questions)} 个训练样本',
                    'status': 'success'
                })
                
            # 检查是否指定使用现有知识库文件
            elif request.POST.get('use_kb_file'):
                # 使用知识库中的datas.txt作为训练语料
                kb_file = os.path.join(MODEL_CONFIG['KB_DIR'], 'datas.txt')
                
                if not os.path.exists(kb_file):
                    return JsonResponse({
                        'error': '知识库中不存在datas.txt文件',
                        'status': 'error'
                    }, status=404)
                
                logger.info(f"使用知识库中的数据文件: {kb_file}")
                
                # 从train_model.py导入处理函数
                from .train_model import prepare_qa_pairs_data
                
                # 准备训练数据
                logger.info("准备训练数据...")
                train_questions, train_contexts, train_answers, val_questions, val_contexts, val_answers = \
                    prepare_qa_pairs_data(kb_file)
                
                if not train_questions:
                    return JsonResponse({
                        'error': '数据集解析失败或没有有效的训练样本',
                        'status': 'error'
                    }, status=400)
                
                logger.info(f"成功解析 {len(train_questions)} 个训练样本和 {len(val_questions)} 个验证样本")
                
                # 设置较小的批量大小，避免内存问题
                batch_size = min(4, max(1, len(train_questions) // 10))  # 动态调整批量大小
                epochs = 1  # 一个训练轮次
                
                # 创建临时目录用于保存模型
                temp_model_path = MODEL_CONFIG['TEMP_MODEL_PATH']
                os.makedirs(temp_model_path, exist_ok=True)
                
                # 初始化模型
                if qa_model is None:
                    qa_model = TransformerQA(model_name="distilbert-base-cased-distilled-squad")
                
                # 训练模型
                logger.info(f"开始训练模型，批量大小: {batch_size}，训练轮次: {epochs}")
                try:
                    qa_model.train(
                        train_questions=train_questions,
                        train_contexts=train_contexts,
                        train_answers=train_answers,
                        val_questions=val_questions,
                        val_contexts=val_contexts,
                        val_answers=val_answers,
                        batch_size=batch_size,
                        epochs=epochs,
                        save_path=temp_model_path
                    )
                    
                    # 确保手动调用保存方法，确保生成pytorch_model.bin
                    if not os.path.exists(os.path.join(temp_model_path, "pytorch_model.bin")):
                        logger.warning("训练后未发现模型文件，尝试手动保存")
                        save_result = qa_model.save(temp_model_path)
                        if save_result:
                            logger.info("手动保存模型成功")
                        else:
                            logger.error("手动保存模型失败")
                            return JsonResponse({
                                'error': '保存模型失败，请联系管理员',
                                'status': 'error'
                            }, status=500)
                    
                    # 复制模型到正式目录
                    target_model_path = MODEL_CONFIG['MODEL_PATH']
                    os.makedirs(target_model_path, exist_ok=True)
                    for file_name in os.listdir(temp_model_path):
                        if file_name.endswith('.json') or file_name.endswith('.bin'):
                            src = os.path.join(temp_model_path, file_name)
                            dst = os.path.join(target_model_path, file_name)
                            shutil.copy2(src, dst)
                            logger.info(f"已复制模型文件: {file_name}")
                    
                    # 重新加载模型
                    qa_model.load(temp_model_path)
                    
                    return JsonResponse({
                        'success': True,
                        'message': f'模型训练完成，共处理了 {len(train_questions)} 个训练样本',
                        'status': 'success'
                    })
                    
                except Exception as e:
                    logger.error(f"训练模型失败: {str(e)}", exc_info=True)
                    return JsonResponse({
                        'error': f'训练过程中出错: {str(e)}',
                        'status': 'error'
                    }, status=500)
                
            else:
                # 没有上传文件，使用预训练模型
                logger.info("没有上传数据集，使用预训练模型...")
                model_name = "distilbert-base-cased-distilled-squad"
                logger.info(f"正在加载预训练模型: {model_name}")
                
                qa_model = TransformerQA(model_name=model_name)
                
                # 验证模型是否可用
                test_question = "什么是Python?"
                test_context = "Python是一种高级编程语言，具有简洁易读的语法。"
                
                try:
                    test_result = qa_model.predict(test_question, test_context)
                    if test_result and len(test_result) > 0:
                        logger.info(f"模型测试成功，预测结果: {test_result[0]['text']}")
                    else:
                        logger.warning("模型测试没有返回结果")
                except Exception as e:
                    logger.error(f"模型测试失败: {str(e)}")
                
                return JsonResponse({
                    'success': True,
                    'message': '已加载预训练问答模型',
                    'model_name': model_name,
                    'status': 'success'
                })
                
        except Exception as e:
            logger.error(f"训练模型视图函数出错: {str(e)}", exc_info=True)
            return JsonResponse({
                'error': str(e),
                'status': 'error'
                }, status=500)
    
    return JsonResponse({
        'error': '只支持POST请求',
        'status': 'error'
    }, status=405)

@csrf_exempt
def update_knowledge_view(request):
    """
    更新知识库的视图函数
    """
    if request.method == 'POST':
        try:
            # 重新加载知识库
            logger.info("重新加载知识库...")
            knowledge_manager.documents = {}
            knowledge_manager.embeddings = {}
            knowledge_manager.document_chunks = {}
            knowledge_manager.chunk_embeddings = {}
            
            knowledge_manager.load_documents_from_directory(MODEL_CONFIG['KB_DIR'])
            knowledge_manager.encode_documents()

            return JsonResponse({
                'success': True,
                'message': '知识库已更新',
                'document_count': len(knowledge_manager.documents),
                'chunk_count': len(knowledge_manager.chunk_embeddings)
            })
            
        except Exception as e:
            logger.error(f"更新知识库时出错: {str(e)}", exc_info=True)
            return JsonResponse({
                'error': str(e)
            }, status=500)
            
    return JsonResponse({
        'error': '只支持POST请求'
    }, status=405)
