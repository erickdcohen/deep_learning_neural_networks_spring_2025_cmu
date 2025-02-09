"""
You will need to implement a single layer neural network from scratch.

IMPORTANT: DO NOT change any function signatures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Transform(object):
    """
    This is the base class. You do not need to change anything.
    Read the comments in this class carefully.
    """

    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Compute and save the gradients wrt the parameters for step()
        Return grad_wrt_x which will be the grad_wrt_out for previous Transform
        """
        pass

    def step(self):
        """
        Apply gradients to update the parameters
        """
        pass

    def zerograd(self):
        """
        This is used to Reset the gradients.
        Usually called before backward()
        """
        pass


class ReLU(Transform):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (indim, batch_size)
        """
        self.x = x
        return x * (x > 0).float()

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        """

        return grad_wrt_out * (self.x > 0)


class LinearMap(Transform):
    def __init__(self, indim, outdim, lr=0.01):
        """
        indim: input dimension
        outdim: output dimension
        lr: learning rate
        """
        super(LinearMap, self).__init__()

        self.weights = torch.nn.init.kaiming_uniform_(torch.empty(
            (outdim, indim), dtype=torch.float32, requires_grad=True, device=device),
            nonlinearity='relu')

        # self.weights = 0.01 * \
        #     torch.rand((outdim, indim), dtype=torch.float64,
        #                requires_grad=True, device=device)

        self.bias = 0.01 * \
            torch.rand((outdim, 1), dtype=torch.float32,
                       requires_grad=True, device=device)

        self.lr = lr

    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (outdim, batch_size)
        """
        self.x = x
        self.logits = torch.matmul(self.weights, self.x) + self.bias
        return self.logits

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        return shape (indim, batch_size)
        """
        batch_size = grad_wrt_out.shape[1]

        # compute grad_wrt_weights
        self.grad_wrt_weights = torch.matmul(
            grad_wrt_out, self.x.T)

        # compute grad_wrt_bias
        self.grad_wrt_bias = torch.sum(
            grad_wrt_out, axis=1, keepdims=True)

        # compute & return grad_wrt_input
        self.grad_wrt_input = torch.matmul(self.weights.T, grad_wrt_out)

        return self.grad_wrt_input

    def step(self):
        """
        apply gradients calculated by backward() to update the parameters
        """
        with torch.no_grad():
            self.weights -= self.lr * self.grad_wrt_weights
            self.bias -= self.lr * self.grad_wrt_bias

        # with torch.no_grad():
        #     # Print before update
        #     print("Before Update: ", self.weights[0, 0].item())
        #     self.weights -= self.lr * self.grad_wrt_weights
        #     self.bias -= self.lr * self.grad_wrt_bias
        #     # Print after update
        #     print("After Update: ", self.weights[0, 0].item())


class SoftmaxCrossEntropyLoss(object):
    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are one-hot labels of given inputs
        logits and labels are in the shape of (num_classes, batch_size)
        returns loss as a scalar (i.e. mean value of the batch_size loss)
        """

        # Calculate softmax
        self.labels = labels
        self.exp_logits = torch.exp(logits)
        self.sm = self.exp_logits / \
            torch.sum(self.exp_logits, dim=0, keepdims=True)

        # Calculate cross entropy loss
        loss = -torch.mean(
            torch.sum(labels * torch.log(self.sm + 1e-10), dim=0)
        )
        return loss

    def backward(self):
        """
        return grad_wrt_logits shape (num_classes, batch_size)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        batch_size = self.labels.shape[1]
        grad_wrt_logits = (self.sm - self.labels) / batch_size
        return grad_wrt_logits

    def getAccu(self):
        """
        return accuracy here
        """
        preds = torch.argmax(self.sm, axis=0)
        labels = torch.argmax(self.labels, axis=0)
        accuracy = torch.mean(preds == labels)
        return accuracy


class SingleLayerMLP(Transform):
    """constructing a single layer neural network with the previous functions"""

    def __init__(self, indim, outdim, hidden_layer=100, lr=0.01):
        super(SingleLayerMLP, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hiddendim = hidden_layer
        self.lr = lr

        self.first_layer = LinearMap(
            indim=indim, outdim=self.hiddendim, lr=self.lr)
        self.activation = ReLU()
        self.hidden_layer = LinearMap(
            indim=self.hiddendim, outdim=outdim, lr=self.lr)

    def forward(self, x):
        """
        x shape (indim, batch_size)
        return the presoftmax logits shape(outdim, batch_size)
        """

        self.input = x
        self.z1 = self.first_layer.forward(x)
        self.a1 = self.activation.forward(self.z1)
        self.output = self.hidden_layer.forward(self.a1)

        return self.output

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        calculate the gradients wrt the parameters
        """
        # Backpropagate through output layer
        grad_hidden = self.hidden_layer.backward(grad_wrt_out)

        # Backpropagate through ReLU
        grad_z1 = grad_hidden * self.activation.backward(self.z1)

        # Backpropagate through first layer
        grad_input = self.first_layer.backward(grad_z1)

        return grad_input

    def step(self):
        """update model parameters"""

        self.first_layer.step()
        self.hidden_layer.step()


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


def labels2onehot(labels: np.ndarray):
    return np.array([[i == lab for i in range(2)] for lab in labels]).astype(int)


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
    Ytest = pd.read_csv("./data/y_test.csv")
    Xtest = pd.DataFrame(scaler.fit_transform(
        Xtest), columns=Xtest.columns).to_numpy()
    Ytest = np.squeeze(Ytest)
    m2, n2 = Xtest.shape
    # print(m1, n2)
    test_ds = DS(Xtest, Ytest)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = SingleLayerMLP(indim, outdim, hidden_dim, lr)
    loss_fn = SoftmaxCrossEntropyLoss()

    train_losses = []
    test_losses = []

    train_accuracies = []
    test_accuracies = []

    print(f"device {device}")

    for epoch in range(epochs):
        # Training phase
        train_correct = 0
        train_total = 0
        train_loss = 0.0

        for X_train, y_train in train_loader:
            # Move data to GPU and convert to correct format
            X_train = X_train.to(device).T.float()

            y_train = labels2onehot(y_train.numpy()).T
            y_train = torch.tensor(
                y_train, dtype=torch.float32, device=device)

            # Forward pass
            logits = model.forward(X_train)
            loss = loss_fn.forward(logits, y_train)
            train_loss += loss.item()

            # Backward pass and update
            model.backward(loss_fn.backward())

            model.step()

            # Compute accuracy
            train_correct += torch.sum(torch.argmax(logits, axis=0)
                                       == torch.argmax(y_train, axis=0)).item()
            train_total += y_train.shape[1]

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = train_correct / train_total
        train_accuracies.append(train_accuracy)

        # Evaluation phase
        test_correct = 0
        test_total = 0
        test_loss = 0.0

        with torch.no_grad():  # No gradients needed during testing
            for X_test, y_test in test_loader:
                X_test = X_test.to(device).T.float()
                y_test = labels2onehot(y_test.numpy()).T
                y_test = torch.tensor(
                    y_test, dtype=torch.float32, device=device)

                test_logits = model.forward(X_test)
                loss = loss_fn.forward(test_logits, y_test)
                test_loss += loss.item()

                test_correct += torch.sum(torch.argmax(test_logits, axis=0)
                                          == torch.argmax(y_test, axis=0)).item()
                test_total += y_test.shape[1]

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accuracy = test_correct / test_total
        test_accuracies.append(test_accuracy)

        if epoch % 20 == 0:
            print(
                f"""Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f},
                Train Acc: {train_accuracy:.4f} | Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}"""
            )

    # Plot Loss & Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(test_losses, label="Test Loss")
    ax1.set(xlabel="Epochs", ylabel="Loss")
    ax1.legend()

    ax2.plot(train_accuracies, label="Train Accuracy")
    ax2.plot(test_accuracies, label="Test Accuracy")
    ax2.set(xlabel="Epochs", ylabel="Accuracy")
    ax2.legend()

    plt.savefig("nn.png")
