from lightning import Trainer
from datamodule import PAFDatamodule
from second_model import ProteinClassifierHF

BASE_MODEL_NAME = "oohtmeel/Bert_protein_classifier"

if __name__ == "__main__":
    n_classes = 25
    datamodule = PAFDatamodule("data/datafiles", batch_size=32)
    model = ProteinClassifierHF(
        BASE_MODEL_NAME=BASE_MODEL_NAME, n_classes=n_classes)
    trainer = Trainer(max_epochs=10)
    trainer.fit(model=model, datamodule=datamodule)
