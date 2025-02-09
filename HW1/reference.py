"""
You will need to validate your NN implementation using PyTorch. You can use any PyTorch functional or modules in this code.

IMPORTANT: DO NOT change any function signatures
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class SingleLayerMLP(nn.Module):
    """constructing a single layer neural network with Pytorch"""

    def __init__(self, indim, outdim, hidden_dim=100):
        super(SingleLayerMLP, self).__init__()

        # Define the layers
        self.first_layer = nn.Linear(indim, hidden_dim)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(hidden_dim, outdim)
        self.double()

    def forward(self, x):
        """
        x shape (batch_size, indim)
        """

        x = self.first_layer(x)
        x = self.relu(x)
        x = self.hidden_layer(x)
        return x


class DS(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.length = len(X)
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx]
        return (x, y)

    def __len__(self):
        return self.length


def validate(loader):
    """takes in a dataloader, then returns the model loss and accuracy on this loader"""

    if loader == train_loader:
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = train_correct / train_total
        avg_train_loss = train_loss / len(loader)

        return avg_train_loss, train_accuracy
    else:
        model.eval()
        eval_correct = 0
        eval_total = 0
        eval_loss = 0.0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                eval_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                eval_total += labels.size(0)
                eval_correct += (predicted == labels).sum().item()
        eval_accuracy = eval_correct / eval_total
        avg_eval_loss = eval_loss / len(loader)

        return avg_eval_loss, eval_accuracy


if __name__ == "__main__":
    """The dataset loaders were provided for you.
    You need to implement your own training process.
    You need plot the loss and accuracies during the training process and test process.
    """

    indim = 60
    outdim = 2
    hidden_dim = 100
    lr = 0.01
    batch_size = 64
    epochs = 500
    print(f"device: {device}")

    # dataset
    Xtrain = pd.read_csv("./data/X_train.csv")
    Ytrain = pd.read_csv("./data/y_train.csv")
    scaler = MinMaxScaler()
    Xtrain = pd.DataFrame(scaler.fit_transform(
        Xtrain), columns=Xtrain.columns).to_numpy()
    Ytrain = np.squeeze(Ytrain)
    m1, n1 = Xtrain.shape
    # print(m1, n1)
    train_ds = DS(Xtrain, Ytrain)
    train_loader = DataLoader(train_ds, batch_size=batch_size)

    Xtest = pd.read_csv("./data/X_test.csv")
    Ytest = pd.read_csv("./data/y_test.csv").to_numpy()
    Xtest = pd.DataFrame(scaler.fit_transform(
        Xtest), columns=Xtest.columns).to_numpy()
    Ytest = np.squeeze(Ytest)
    m2, n2 = Xtest.shape
    # print(m1, n2)
    test_ds = DS(Xtest, Ytest)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # construct the model
    model = SingleLayerMLP(indim=indim, outdim=outdim).to(device)

    # construct the training process
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_loss_list = []
    test_loss_list = []

    train_acc_list = []
    test_acc_list = []

    for epoch in range(epochs):
        train_loss, train_acc = validate(loader=train_loader)
        test_loss, test_acc = validate(loader=test_loader)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(
            f"""Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f} , Train Acc {
                train_acc:.4f} | Test Loss: {test_loss:.4f} , Train Acc {test_acc:.4f}"""
        )

    # Plot Loss & Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))

    # fig.figure(figsize=(12, 5))
    ax1.plot(train_loss_list, label="Train Loss")
    ax1.plot(test_loss_list, label="Test Loss")
    ax1.set(xlabel="Epochs", ylabel="Loss")
    ax1.legend()

    ax2.plot(train_acc_list, label="Train Accuracy")
    ax2.plot(test_acc_list, label="Test Accuracy")
    ax2.set(xlabel="Epochs", ylabel="Accuracy")
    ax2.legend()

    plt.savefig("ref.png")
