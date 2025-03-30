from lightning import Trainer
from datamodule import PAFDatamodule
from second_model import ProteinClassifierHF
import torch
import os
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# BASE_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
# BASE_MODEL_NAME = "ShuklaGroupIllinois/LazDEF_ESM2"
BASE_MODEL_NAME = "Rocketknight1/esm2_t6_8M_UR50D"
# BASE_MODEL_NAME = "Rostlab/prot_bert"

torch.set_float32_matmul_precision('medium')
CHKPTS_PATH = "lightning_logs/checkpoints/"


def main() -> None:
    n_classes = 25
    datamodule = PAFDatamodule("data/datafiles", batch_size=32)

    # Model Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHKPTS_PATH,
        # filename='base-classifier-{epoch:02d}-{val_loss:.2f}',
        filename=BASE_MODEL_NAME.rsplit(
            '/', 1)[-1] + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # Early Stopping Callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode='min'
    )

    model = ProteinClassifierHF(
        BASE_MODEL_NAME=BASE_MODEL_NAME,
        n_classes=n_classes,
    )

    trainer = Trainer(
        max_epochs=5,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
        ],
        enable_progress_bar=True,
        accelerator="auto",
        devices="auto",
        strategy="auto"
    )

    print(model.model.config)  # Check model configuration

    print(model.model.classifier)  # Verify classifier layer

    # If you want to resume training from a checkpoint
    chkpts = os.listdir(CHKPTS_PATH)
    if chkpts:
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=os.path.join(CHKPTS_PATH, chkpts[-1])
        )
    else:
        trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
