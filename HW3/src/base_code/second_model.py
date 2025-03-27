import torch
import torchmetrics
from lightning import LightningModule
from torch import nn
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ProteinClassifierHF(LightningModule):
    def __init__(self, BASE_MODEL_NAME, n_classes=25):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME, num_labels=n_classes)

        # Replace the classifier layer with the correct number of output classes
        self.model.classifier = torch.nn.Linear(
            self.model.config.hidden_size, n_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass",
                                                                 num_classes=n_classes)
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass",
                                                                   num_classes=n_classes)
        self.val_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=n_classes)

    def forward(self, x):

        ids = self.tokenizer(x, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(
            self.device).to(self.dtype)

        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return output.logits

    def training_step(self, batch, batch_idx):
        '''
        calculate output --> loss --> training accuracy and save to self.log
        return loss
        '''
        X, y = batch
        output = self.forward(X)
        loss = self.criterion(output, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        '''
        make predictions and calculate validation accuracy/F1 score and save to self.log
        '''
        X, y = batch
        output = self.forward(X)
        loss = self.criterion(output, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        '''
        return optimizer for the model
        '''
        return Adam(self.parameters())
