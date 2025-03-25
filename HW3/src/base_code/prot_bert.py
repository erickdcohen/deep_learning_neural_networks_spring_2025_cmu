import torch
import torchmetrics
from lightning import LightningModule
from torch import nn
from torch.optim import Adam
from transformers import BertTokenizer, BertGenerationEncoder


class ProteinClassifier(LightningModule):
    def __init__(self, n_classes=25):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(
            "Rostlab/prot_bert", do_lower_case=False)
        self.embedder = BertGenerationEncoder.from_pretrained(
            "Rostlab/prot_bert")
        dmodel = 1024
        self.model = nn.Linear(dmodel, n_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass",
                                                                 num_classes=n_classes)
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass",
                                                                   num_classes=n_classes)
        self.val_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=n_classes)

    def forward(self, x):
        lengths = torch.tensor([len(i) for i in x]).to(self.device)
        ids = self.tokenizer(x, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(
            self.device).to(self.dtype)
        with torch.no_grad():
            embeddings = self.embedder(input_ids=input_ids,
                                       attention_mask=attention_mask).last_hidden_state
        embeddings = embeddings.sum(dim=1)/lengths.view(-1, 1)
        logits = self.model(embeddings)
        return logits

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
