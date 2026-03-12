from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

def evaluate_model(
    model_path="models/best.pt",
    data_yaml="config/drone_data.yaml",
    img_size=640,
    conf=0.5,
    iou=0.5
):
    print("=" * 60)
    print("YOLOv8 Drone Detection — Model Evaluation")
    print("=" * 60)

    model = YOLO(model_path)

    print("\nRunning evaluation on test set...")
    metrics = model.val(
        data=data_yaml,
        imgsz=img_size,
        conf=conf,
        iou=iou,
        split='test',
        plots=True,
        verbose=True
    )

    precision = float(metrics.box.p[0]) if len(metrics.box.p) > 0 else 0
    recall    = float(metrics.box.r[0]) if len(metrics.box.r) > 0 else 0
    map50     = float(metrics.box.map50)
    map50_95  = float(metrics.box.map)
    f1_score  = 2 * (precision * recall) / (precision + recall + 1e-6)

    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(f"Precision     : {precision:.4f}  ({precision*100:.2f}%)")
    print(f"Recall        : {recall:.4f}  ({recall*100:.2f}%)")
    print(f"F1-Score      : {f1_score:.4f}  ({f1_score*100:.2f}%)")
    print(f"mAP@0.5       : {map50:.4f}  ({map50*100:.2f}%)")
    print(f"mAP@0.5:0.95  : {map50_95:.4f}  ({map50_95*100:.2f}%)")
    print("=" * 60)

    plot_metrics_chart(precision, recall, f1_score, map50, map50_95)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mAP50': map50,
        'mAP50_95': map50_95
    }


def plot_metrics_chart(precision, recall, f1, map50, map50_95):
    metrics_names  = ['Precision', 'Recall', 'F1-Score', 'mAP@0.5', 'mAP@0.5:0.95']
    metrics_values = [precision, recall, f1, map50, map50_95]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('YOLOv8 Drone Detection — Model Performance',
                 fontsize=14, fontweight='bold')

    bars = ax1.bar(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='80% threshold')

    for bar, val in zip(bars, metrics_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax1.legend(fontsize=9)
    ax1.tick_params(axis='x', rotation=15)

    angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
    metrics_values_radar = metrics_values + [metrics_values[0]]
    angles += angles[:1]

    ax2 = plt.subplot(122, projection='polar')
    ax2.plot(angles, metrics_values_radar, 'o-', linewidth=2, color='#2196F3')
    ax2.fill(angles, metrics_values_radar, alpha=0.25, color='#2196F3')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics_names, size=9)
    ax2.set_ylim(0, 1)
    ax2.set_title('Performance Radar', pad=20)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=7)
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/evaluation_metrics.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Metrics chart saved to results/evaluation_metrics.png")


def compare_confidence_thresholds(
    model_path="models/best.pt",
    image_path="dataset/images/test/sample.jpg"
):
    model = YOLO(model_path)
    thresholds = [0.3, 0.5, 0.7, 0.9]

    import cv2
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Detection at Different Confidence Thresholds", fontsize=14, fontweight='bold')

    for ax, conf in zip(axes, thresholds):
        results = model(image_path, conf=conf, verbose=False)
        for result in results:
            img = result.plot()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            num_det = len(result.boxes) if result.boxes else 0
            ax.imshow(img_rgb)
            ax.set_title(f"Conf={conf}\nDrones={num_det}", fontweight='bold')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig("results/confidence_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Comparison saved to results/confidence_comparison.png")


if __name__ == "__main__":
    evaluate_model(
        model_path="models/best.pt",
        data_yaml="config/drone_data.yaml"
    )