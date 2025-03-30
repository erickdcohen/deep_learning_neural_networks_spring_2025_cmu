import torch
import torchmetrics
from lightning import LightningModule
from torch import nn
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ProteinClassifierHF(LightningModule):

    def __init__(self, BASE_MODEL_NAME, n_classes=25):
        super().__init__()

        # set tokenizer from pretrained model using AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME,
            do_lower_case=False,
        )

        # set model from automodelforseqclasss
        self.model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=n_classes,
            problem_type="multi_label_classification"
        )

        # set criterion
        self.criterion = nn.CrossEntropyLoss()

        # metrics for measuring performance
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass",
                                                                 num_classes=n_classes)
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass",
                                                                   num_classes=n_classes)
        self.val_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=n_classes)

    def forward(self, x):
        ids = self.tokenizer(x, add_special_tokens=True,
                             padding="longest", return_tensors="pt")
        input_ids = ids['input_ids'].to(self.device)
        attention_mask = ids['attention_mask'].to(
            self.device).to(self.dtype)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        return outputs.logits

    def training_step(self, batch, batch_idx):
        '''
        calculate output --> loss --> training accuracy and save to self.log
        return loss
        '''
        # Unpack batch
        X, y = batch

        # Get model output
        logits = self.forward(X)

        # Compute loss
        loss = self.criterion(logits, y)

        # Compute predictions for accuracy
        preds = torch.argmax(logits, dim=1)

        # Update and log training accuracy
        train_acc = self.train_accuracy(preds, y)

        # Logging
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        self.log("train_accuracy", train_acc, on_step=True,
                 on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        '''
        make predictions and calculate validation accuracy/F1 score and save to self.log
        '''
        # Unpack batch
        X, y = batch

        # Get model output
        logits = self.forward(X)

        # Compute loss
        loss = self.criterion(logits, y)

        # Compute predictions
        preds = torch.argmax(logits, dim=1)

        # Update and log validation metrics
        val_acc = self.val_accuracy(preds, y)

        # Logging
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", val_acc, on_step=False,
                 on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        '''
        return optimizer for the model
        '''
        return Adam(self.parameters())
