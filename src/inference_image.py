from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path

def detect_drones_in_image(
    image_path,
    model_path="models/best.pt",
    conf_threshold=0.5,
    save_output=True
):
    print(f"\n Detecting drones in: {image_path}")
    model = YOLO(model_path)
    results = model(image_path, conf=conf_threshold)
    
    for result in results:
        
        annotated_img = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            print(f"✅ Found {len(boxes)} drone(s)!")
            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"   Drone {i+1}: Confidence={conf:.2f}, "
                      f"BBox=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        
        else:
            print("❌ No drones detected.")
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_rgb)
        plt.title("YOLOv8 Drone Detection Result", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        if save_output:
            os.makedirs("results/images", exist_ok=True)
            filename = Path(image_path).stem
            output_path = f"results/images/{filename}_detected.jpg"
            cv2.imwrite(output_path, annotated_img)
            print(f"Saved to: {output_path}")
        plt.tight_layout()
        plt.savefig("results/images/detection_plot.png", dpi=150)
        plt.show()
    
    return results

def detect_batch_images(
    folder_path,
    model_path="models/best.pt",
    conf_threshold=0.5
):
    print(f"\n Processing folder: {folder_path}")

    model = YOLO(model_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    image_files = [
        f for f in Path(folder_path).iterdir()
        if f.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images\n")

    total_detections = 0
    results_summary = []

    for img_path in image_files:
        results = model(str(img_path), conf=conf_threshold, verbose=False)

        for result in results:
            num_drones = len(result.boxes) if result.boxes else 0
            total_detections += num_drones
            results_summary.append({
                'image': img_path.name,
                'drones_detected': num_drones
            })

            annotated = result.plot()
            os.makedirs("results/images/batch", exist_ok=True)
            cv2.imwrite(f"results/images/batch/{img_path.stem}_detected.jpg", annotated)

        print(f"  {img_path.name}: {num_drones} drone(s) detected")

    print(f"\n Summary:")
    print(f"Total Images  : {len(image_files)}")
    print(f"Total Drones  : {total_detections}")
    print(f"Avg per image : {total_detections/max(len(image_files),1):.2f}")
    print(f"Saved to      : results/images/batch/")

    return results_summary

if __name__ == "__main__":
    detect_drones_in_image(
        image_path="sample/image/drone_img.png",
        model_path="models/best.pt",
        conf_threshold=0.5
    )