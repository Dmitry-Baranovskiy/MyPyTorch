import torch
import torch.nn as nn

#print(nn.ReLU)


# === 1. Простая сеть для 2D бинарной классификации ===
class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid() # для ВСЕLoss
            
        )

    def forward(self, x):
        return self.net(x)
    

# === 2. Задача MNIST классификации ===
class MnistMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,10)

        )

    def forward(self, x):
        return self.net(x)
    

# == 3. Многоуровневая регрессия ==
class LocalizationMLP(nn.Module): 
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64), # Вход - десять признаков (сенсоры)
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # Выход - координаты x, y, z
        )
    
    def froward(self, x):
        return self.net(x )