from lightning import Trainer
from datamodule import PAFDatamodule
from second_model import ProteinClassifierHF
import os
import pandas as pd
import numpy as np
import pickle
import torch

BASE_MODEL_NAME = "Rocketknight1/esm2_t6_8M_UR50D"
CHKPTS_PATH = "lightning_logs/checkpoints/"
torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    test_data = pd.read_csv("data/datafiles/test_data.csv")

    classes = pickle.load(
        open("data/datafiles/selected_families.pkl", "rb"))

    n_classes = 25
    datamodule = PAFDatamodule("data/datafiles", batch_size=32)

    chkpts = os.listdir(CHKPTS_PATH)

    model = ProteinClassifierHF.load_from_checkpoint(
        CHKPTS_PATH + "esm2_t6_8M_UR50D-epoch=00-val_loss=0.01.ckpt",
        n_classes=n_classes,
        BASE_MODEL_NAME=BASE_MODEL_NAME
    )
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

    preds_df.to_csv("output/model_2_output.csv", index=False)
    print("Predictions saved to output.")
