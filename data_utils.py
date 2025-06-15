import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from torchvision import datasets, transforms

# Task 1: 2D Syntetic Classification
def get_2d_classification_data(batch_size=32):
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=42
    )

    X = torch.tensor(X, dtype = torch.float32)
    y = torch.tensor(y, dtype = torch.float32).view(-1, 1)

    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size = batch_size, shuffle=True)


# Task 2: MNIST Classification  
def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    val_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# Task 3: Multiple regression
from sklearn.datasets import make_regression
from torch.utils.data import DataLoader, TensorDataset
import torch


def get_localization_data(batch_size=64):
    X, y = make_regression(
        n_samples=1000,
        n_features=10, #Имитация данных с 10 сенсоров 
        n_targets=3, #Три предсказываемых координаты x, y, z
        noise=0.5,
        random_state=42
    )

    #Преобразуем в torch.Tensor
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size = batch_size, shuffle = True)