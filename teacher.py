import os
import shutil
import subprocess
import time
import matplotlib.pyplot as plt
import pandas as pd

# === CONFIGURATION ===
YOLO_DIR = "/home/lucy/Desktop/brassai_projekt/yolov5"  # Path to cloned yolov5 repo
DATA_YAML = "../data.yaml"  # Path to dataset YAML
EPOCHS = 100
BATCH_SIZE = 16
NAME = "Agriculture_CNN"
IMG_SIZE = 640
PROJECT = "runs/train"

# === STEP 1: Launch Training ===
def train_yolo():
    os.chdir(YOLO_DIR)
    cmd = [
        "python3", "train.py",
        f"--data={DATA_YAML}",
        f"--img={IMG_SIZE}",
        f"--batch={BATCH_SIZE}",
        f"--epochs={EPOCHS}",
        f"--name={NAME}",
        "--exist-ok"  # Overwrite if exists
    ]
    print("🚀 Starting training...")
    subprocess.run(cmd)
    print("✅ Training completed.")

# === STEP 2: Monitor and Save Results ===
def monitor_results():
    results_path = os.path.join(YOLO_DIR, PROJECT, NAME, "results.csv")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results not found at {results_path}")
    
    # Read CSV
    df = pd.read_csv(results_path)

    # Plotting loss and metrics
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(df['train/box_loss'], label='Box Loss')
    axs[0, 0].plot(df['train/obj_loss'], label='Obj Loss')
    axs[0, 0].set_title("Training Loss")
    axs[0, 0].legend()

    axs[0, 1].plot(df['metrics/precision(B)'], label='Precision')
    axs[0, 1].plot(df['metrics/recall(B)'], label='Recall')
    axs[0, 1].set_title("Precision and Recall")
    axs[0, 1].legend()

    axs[1, 0].plot(df['metrics/mAP_0.5(B)'], label='mAP@0.5')
    axs[1, 0].plot(df['metrics/mAP_0.5:0.95(B)'], label='mAP@0.5:0.95')
    axs[1, 0].set_title("mAP")
    axs[1, 0].legend()

    for ax in axs.flat:
        ax.set_xlabel("Epoch")

    plt.tight_layout()
    output_plot = os.path.join(YOLO_DIR, PROJECT, NAME, "training_summary.png")
    plt.savefig(output_plot)
    print(f"📊 Saved training summary to {output_plot}")

# === STEP 3: Backup Best Weights ===
def save_best_weights():
    best_path = os.path.join(YOLO_DIR, PROJECT, NAME, "weights", "best.pt")
    backup_path = os.path.join(YOLO_DIR, f"{NAME}_best.pt")
    shutil.copy(best_path, backup_path)
    print(f"💾 Backed up best weights to {backup_path}")

# === MAIN FLOW ===
if __name__ == "__main__":
    train_yolo()
    monitor_results()
    save_best_weights()
