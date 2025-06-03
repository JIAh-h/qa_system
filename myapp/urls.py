from django.urls import path
from . import views

urlpatterns = [
    path('', views.qa_page_view, name='qa_page'),  # 将根路径指向渲染qa.html的视图
    path('qa/', views.qa_view, name='qa'),  # 问答接口 (POST Only)
    path('train/', views.train_model_view, name='train'),  # 训练模型接口
    path('update_kb/', views.update_knowledge_view, name='update_kb'),  # 更新知识库接口
]