"""
You will need to implement a single layer neural network from scratch.

IMPORTANT: DO NOT change any function signatures
"""

import numpy as np
import pandas as pd
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
        return np.maximum(0.0, x)

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        """
        return 0.0 if grad_wrt_out <= 0 else 1.0


class LinearMap(Transform):
    def __init__(self, indim, outdim, lr=0.01):
        """
        indim: input dimension
        outdim: output dimension
        lr: learning rate
        """
        super(LinearMap, self).__init__()
        rng = np.random.default_rng(seed=1)

        self.weights = 0.01 * rng.random((outdim, indim))
        # self.weights = 0.01 * \
        #     torch.rand((outdim, indim), dtype=torch.float64,
        #                requires_grad=True, device=device)
        self.bias = 0.01 * rng.random((outdim, 1))
        # self.bias = 0.01 * \
        #     torch.rand((outdim, 1), dtype=torch.float64,
        #                requires_grad=True, device=device)
        self.lr = lr

    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (outdim, batch_size)
        """
        return np.dot(self.weights, x) + self.bias

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        return shape (indim, batch_size)
        """
        # compute grad_wrt_weights

        # compute grad_wrt_bias

        # compute & return grad_wrt_input

        raise NotImplementedError()

    def step(self):
        """
        apply gradients calculated by backward() to update the parameters
        """
        raise NotImplementedError()


class SoftmaxCrossEntropyLoss(object):
    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are one-hot labels of given inputs
        logits and labels are in the shape of (num_classes, batch_size)
        returns loss as a scalar (i.e. mean value of the batch_size loss)
        """

        # Calculate softmax
        exp_logits = np.exp(logits)
        sm = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Calculate cross entropy loss
        return -np.mean(
            np.sum(labels * np.log(sm), axis=1)
        )

    def backward(self):
        """
        return grad_wrt_logits shape (num_classes, batch_size)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        raise NotImplementedError()

    def getAccu(self):
        """
        return accuracy here
        """
        raise NotImplementedError()


class SingleLayerMLP(Transform):
    """constructing a single layer neural network with the previous functions"""

    def __init__(self, indim, outdim, hidden_layer=100, lr=0.01):
        super(SingleLayerMLP, self).__init__()
        raise NotImplementedError()

    def forward(self, x):
        """
        x shape (indim, batch_size)
        return the presoftmax logits shape(outdim, batch_size)
        """
        raise NotImplementedError()

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        calculate the gradients wrt the parameters
        """
        raise NotImplementedError()

    def step(self):
        """update model parameters"""
        raise NotImplementedError()


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

    # TEMP test
    x = Xtrain[:batch_size]
    y = Ytrain[:batch_size]

    x = LinearMap(indim=indim, outdim=hidden_dim).forward(x.T)

    relu = ReLU()
    x = relu.forward(x)

    x = LinearMap(hidden_dim, outdim).forward(x)

    scel = SoftmaxCrossEntropyLoss()

    loss = scel.forward(x.T, labels2onehot(y))
    print(loss)

    # construct the model
    # raise NotImplementedError()
    # construct the training process
    # raise NotImplementedError()
