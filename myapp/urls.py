from django.urls import path
from . import views

urlpatterns = [
    path('qa/', views.qa_view, name='qa'),
    path('train/', views.train_model_view, name='train_model'),
    path('update_kb/', views.update_knowledge_view, name='update_knowledge'),
]