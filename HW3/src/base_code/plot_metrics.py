"""
Plots training metrics
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PATH_TO_BASE_METRICS_FILE = "lightning_logs/version_5/metrics.csv"
PATH_TO_METRICS_FILE = "lightning_logs/version_17/metrics.csv"


def main() -> None:
    # base_metrics_df = pd.read_csv(PATH_TO_BASE_METRICS_FILE)
    # plt.plot("step", "train_loss", data=base_metrics_df)
    # plt.title("Base Protein Bert Training Loss")
    # plt.show()

    pretrained_metrics_df = pd.read_csv(PATH_TO_METRICS_FILE)
    print(pretrained_metrics_df)

    plt.plot("step", "train_loss_step",
             data=pretrained_metrics_df, label="train_loss", color='g')
    plt.title("Pretrained ESM Model")
    plt.show()


if __name__ == "__main__":
    main()
