import os
import json
import logging
import tempfile
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, redirect
from .knowledge_manager import knowledge_manager
from .train_model import train_model, load_and_format_data
from .config import config
from .security import access_control
from .cache_manager import model_cache, document_cache

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_qa_system():
    """初始化问答系统"""
    try:
        # 初始化知识管理器
        success = knowledge_manager.load_documents_from_directory(config.get_model_config()['KB_DIR'])
        if success:
            logger.info("系统初始化成功！")
            return True
        else:
            logger.error("系统初始化失败")
            return False
            
    except Exception as e:
        logger.error(f"初始化问答系统时出错: {str(e)}")
        return False

def train_new_model():
    """训练新模型"""
    try:
        train_file = os.path.join(config.get_model_config()['KB_DIR'], 'train.json')
        test_file = os.path.join(config.get_model_config()['KB_DIR'], 'test.json')
        model_path = config.get_model_config()['MODEL_PATH']
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            logger.error("训练数据文件不存在")
            return False
            
        logger.info("开始训练新模型...")
        success = train_model(
            train_file, 
            test_file, 
            model_path, 
            config.get_model_config()['MODEL_NAME'],
            batch_size=config.get_model_config()['BATCH_SIZE'],
            learning_rate=config.get_model_config()['LEARNING_RATE'],
            epochs=config.get_model_config()['EPOCHS']
        )
        
        if success:
            logger.info("模型训练成功，重新初始化系统...")
            return init_qa_system()
        else:
            logger.error("模型训练失败")
            return False
            
    except Exception as e:
        logger.error(f"训练新模型时出错: {str(e)}")
        return False

def qa_page_view(request):
    """渲染qa.html页面"""
    return render(request, 'qa.html')

@csrf_exempt
def qa_view(request):
    """处理问答请求 (POST Only)"""
    if request.method != 'POST':
        return JsonResponse({'error': '此端点只支持POST请求'}, status=405)
        
    # 检查访问权限
    access_granted, message = access_control.check_access(request)
    if not access_granted:
        return JsonResponse({'error': message}, status=403)
        
    try:
        # 检查系统是否已初始化
        if not knowledge_manager.documents:
            logger.info("系统未初始化，尝试初始化...")
            if not init_qa_system():
                logger.error("系统初始化失败，无法处理请求。")
                return JsonResponse({'error': '系统初始化失败'}, status=500)

        data = json.loads(request.body)
        question = data.get('question', '').strip()
        
        if not question:
            return JsonResponse({'error': '问题不能为空'}, status=400)
            
        # 使用知识管理器生成答案
        answer = knowledge_manager.generate_answer(question)
        
        return JsonResponse({
            'answer': answer,
            'status': 'success'
        })
        
    except json.JSONDecodeError:
        logger.error("接收到的请求体不是有效的JSON")
        return JsonResponse({'error': '无效的请求格式，请发送JSON数据。'}, status=400)
    except Exception as e:
        logger.error(f"处理问题时出错: {str(e)}")
        return JsonResponse({'error': f'处理问题时出错: {str(e)}'}, status=500)

@csrf_exempt
def train_model_view(request):
    """触发模型训练的视图"""
    if request.method != 'POST':
        return JsonResponse({'error': '此端点只支持POST请求'}, status=405)
        
    # 检查访问权限
    access_granted, message = access_control.check_access(request)
    if not access_granted:
        return JsonResponse({'error': message}, status=403)
        
    try:
        # 检查是否有上传的文件
        if 'qa_dataset' in request.FILES:
            uploaded_file = request.FILES['qa_dataset']
            temp_file_path = os.path.join(config.get_model_config()['TEMP_MODEL_PATH'], uploaded_file.name)
            
            with open(temp_file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
                    
            train_file_path = temp_file_path
            test_file_path = os.path.join(config.get_model_config()['KB_DIR'], 'test.json')
            
            if not os.path.exists(test_file_path):
                return JsonResponse({'status': 'error', 'message': '测试数据集文件不存在'}, status=400)
                
            logger.info(f"使用上传文件进行模型训练: {train_file_path}")
            success = train_model(
                train_file_path, 
                test_file_path, 
                config.get_model_config()['MODEL_PATH'],
                config.get_model_config()['MODEL_NAME']
            )
            
        elif request.POST.get('use_kb_file') == 'true':
            train_file_path = os.path.join(config.get_model_config()['KB_DIR'], 'datas.txt')
            test_file_path = os.path.join(config.get_model_config()['KB_DIR'], 'test.json')
            
            if not os.path.exists(train_file_path):
                return JsonResponse({'status': 'error', 'message': '知识库datas.txt文件不存在'}, status=400)
            if not os.path.exists(test_file_path):
                return JsonResponse({'status': 'error', 'message': '测试数据集文件不存在'}, status=400)
                
            logger.info(f"使用知识库文件进行模型训练: {train_file_path}")
            success = train_model(
                train_file_path, 
                test_file_path, 
                config.get_model_config()['MODEL_PATH'],
                config.get_model_config()['MODEL_NAME']
            )
        else:
            train_file = os.path.join(config.get_model_config()['KB_DIR'], 'train.json')
            test_file = os.path.join(config.get_model_config()['KB_DIR'], 'test.json')
            
            if not os.path.exists(train_file) or not os.path.exists(test_file):
                return JsonResponse({'status': 'error', 'message': '默认训练或测试数据集文件不存在'}, status=400)
                
            logger.info(f"使用默认数据集进行模型训练: {train_file}")
            success = train_model(
                train_file, 
                test_file, 
                config.get_model_config()['MODEL_PATH'],
                config.get_model_config()['MODEL_NAME']
            )
            
        if success:
            # 训练成功后重新初始化系统
            init_qa_system()
            return JsonResponse({'status': 'success', 'message': '模型训练成功'})
        else:
            return JsonResponse({'status': 'error', 'message': '模型训练失败'}, status=500)
            
    except Exception as e:
        logger.error(f"训练模型时出错: {str(e)}")
        return JsonResponse({'status': 'error', 'message': f'训练模型时出错: {str(e)}'}, status=500)

@csrf_exempt
def update_knowledge_view(request):
    """更新知识库的视图"""
    if request.method != 'POST':
        return JsonResponse({'error': '只支持POST请求'}, status=405)
        
    # 检查访问权限
    access_granted, message = access_control.check_access(request)
    if not access_granted:
        return JsonResponse({'error': message}, status=403)
        
    try:
        # 重新加载知识库
        if knowledge_manager.load_documents_from_directory(config.get_model_config()['KB_DIR']):
            return JsonResponse({'status': 'success', 'message': '知识库更新成功'})
        else:
            return JsonResponse({'status': 'error', 'message': '知识库更新失败'}, status=500)
            
    except Exception as e:
        logger.error(f"更新知识库时出错: {str(e)}")
        return JsonResponse({'status': 'error', 'message': f'更新知识库时出错: {str(e)}'}, status=500)
