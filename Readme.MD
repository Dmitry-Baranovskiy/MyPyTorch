🔬 My PyTorch Training Playground
Этот проект создан для практики с PyTorch: формирование батчей, создание архитектур, написание цикла обучения и метрик.
📁 Структура проекта
my-pytorch-project/ 
├── notebooks/ ← Один ноутбук — одна задача 
├── model.py ← Все архитектуры сетей 
├── data_utils.py ← Генерация и загрузка данных 
├── train.py ← Универсальный цикл обучения 
├── checkpoints/ ← Сохранённые модели (.pt) 
├── requirements.txt ← pip-зависимости 
└── README.md ← ты сейчас его читаешь 

🚀 Как начать
Клонируй проект или открой в VS Code
Установи зависимости:
pip install -r requirements.txt 
Запусти ноутбук в notebooks/ (например 01_linear_classification.ipynb)
В ноутбуке выбирай модель и данные:
from model import LinearClassifier from data_utils import get_2d_classification_data from train import train_model 
Обучи:
model = LinearClassifier() loader = get_2d_classification_data() train_model(model, loader, loss_fn, optimizer, device) 
🧪 Задачи
[x] 01 — Linear classification
[x] 02 — MNIST classification
[ ] 03 — Spiral classification
[ ] 04 — AutoEncoder (будет позже)
🧠 Цель
Наработать навык в 3 ключевых вещах:
Создание архитектур (model.py)
Подготовка данных (data_utils.py)
Цикл обучения и метрики (train.py)