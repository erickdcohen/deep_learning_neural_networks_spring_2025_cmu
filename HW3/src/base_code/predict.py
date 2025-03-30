from lightning import Trainer
from datamodule import PAFDatamodule
from prot_bert import ProteinClassifier
import os
import pandas as pd
import numpy as np
import pickle
import torch

if __name__ == "__main__":
    test_data = pd.read_csv("data/datafiles/test_data.csv")

    classes = pickle.load(
        open("data/datafiles/selected_families.pkl", "rb"))

    n_classes = len(classes)  # 25
    datamodule = PAFDatamodule("data/datafiles", batch_size=32)
    model = ProteinClassifier.load_from_checkpoint(
        "lightning_logs/version_5/checkpoints/epoch=9-step=10410.ckpt", n_classes=n_classes)
    trainer = Trainer()

    preds = trainer.predict(model=model, datamodule=datamodule)

    all_preds = torch.cat([pred for pred in preds])

    predicted_class_indices = torch.argmax(all_preds, dim=1)

    # Convert the predicted indices to class names using the 'classes' list
    predicted_class_names = [classes[pred.item()]
                             for pred in predicted_class_indices]

    preds_df = pd.DataFrame({
        "sequence_name": test_data["sequence_name"],
        "family_id": predicted_class_names,
    })

    if not os.path.isdir("output"):
        os.mkdir("output")

    preds_df.to_csv("output/bert_baseline_output.csv", index=False)
    print("Predictions saved to output.")
