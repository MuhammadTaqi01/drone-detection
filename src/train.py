from ultralytics import YOLO
import os
import yaml
import matplotlib.pyplot as plt
import pandas as pd

def train_model(
    data_yaml="config/drone_data.yaml",
    model_size="yolov8n",       # n=nano, s=small, m=medium
    epochs=50,
    img_size=640,
    batch_size=16,
    project_name="drone_detection"
):
    print("=" * 60)
    print("YOLOv8 Drone Detection — Training Started")
    print("=" * 60)

    model = YOLO(f"{model_size}.pt")

    print(f"\nModel      : {model_size}")
    print(f"Epochs     : {epochs}")
    print(f"Image Size : {img_size}")
    print(f"Batch Size : {batch_size}")
    print(f"Dataset    : {data_yaml}\n")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name=project_name,
        patience=10,           
        save=True,           
        plots=True,             
        verbose=True,
        conf=0.25,
        iou=0.45,
        device="cpu",           
        workers=4,
        lr0=0.01,               
        lrf=0.001,              
        momentum=0.937,
        weight_decay=0.0005,
        augment=True,           
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    print("\nTraining Complete!")
    print(f"Results saved to: runs/detect/{project_name}/")
    print(f"Best model: runs/detect/{project_name}/weights/best.pt")

    os.makedirs("models", exist_ok=True)
    import shutil
    shutil.copy(
        f"runs/detect/{project_name}/weights/best.pt",
        "models/best.pt"
    )
    print("Best model copied to models/best.pt")

    return results

def plot_training_results(project_name="drone_detection"):
    results_csv = f"runs/detect/{project_name}/results.csv"

    if not os.path.exists(results_csv):
        print("Results CSV not found!")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("YOLOv8 Drone Detection — Training Results", fontsize=16, fontweight='bold')

    axes[0, 0].plot(df['train/box_loss'], label='Train Box Loss', color='blue')
    axes[0, 0].plot(df['val/box_loss'], label='Val Box Loss', color='orange')
    axes[0, 0].set_title('Box Loss')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Epoch')

    axes[0, 1].plot(df['train/cls_loss'], label='Train Cls Loss', color='blue')
    axes[0, 1].plot(df['val/cls_loss'], label='Val Cls Loss', color='orange')
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Epoch')

    axes[0, 2].plot(df['train/dfl_loss'], label='Train DFL Loss', color='blue')
    axes[0, 2].plot(df['val/dfl_loss'], label='Val DFL Loss', color='orange')
    axes[0, 2].set_title('DFL Loss')
    axes[0, 2].legend()
    axes[0, 2].set_xlabel('Epoch')

    axes[1, 0].plot(df['metrics/precision(B)'], color='green')
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel('Epoch')

    axes[1, 1].plot(df['metrics/recall(B)'], color='red')
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel('Epoch')

    axes[1, 2].plot(df['metrics/mAP50(B)'], label='mAP@0.5', color='purple')
    axes[1, 2].plot(df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='brown')
    axes[1, 2].set_title('mAP Score')
    axes[1, 2].legend()
    axes[1, 2].set_xlabel('Epoch')

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_curves.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Training curves saved to results/training_curves.png")

if __name__ == "__main__":
    results = train_model(
        data_yaml="config/drone_data.yaml",
        model_size="yolov8n",
        epochs=50,
        batch_size=16
    )
    plot_training_results()