from lightning import Trainer
from datamodule import PAFDatamodule
from prot_bert import ProteinClassifier

if __name__ == "__main__":
    n_classes = 25
    datamodule = PAFDatamodule("data/datafiles", batch_size=32)
    model = ProteinClassifier(n_classes=n_classes)
    trainer = Trainer(max_epochs=100)
    trainer.fit(model=model, datamodule=datamodule)
