from ultralytics import YOLO
import cv2
import os
import time
from pathlib import Path

def detect_drones_in_video(
    video_path,
    model_path="models/best.pt",
    conf_threshold=0.5,
    save_output=True,
    show_live=True
):
    print("=" * 60)
    print("   YOLOv8 Drone Detection — Video Inference")
    print("=" * 60)
    print(f"Video  : {video_path}")
    print(f"Model  : {model_path}")
    print(f"Conf   : {conf_threshold}")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"\nResolution : {width}x{height}")
    print(f"FPS        : {fps}")
    print(f"Frames     : {total_frames}")
    print(f"Duration   : {duration:.1f}s\n")

    output_path = None
    if save_output:
        os.makedirs("results/videos", exist_ok=True)
        filename = Path(video_path).stem
        output_path = f"results/videos/{filename}_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_detections = 0
    detection_frames = 0
    start_time = time.time()

    print("Starting detection...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model(frame, conf=conf_threshold, verbose=False)

        num_drones = 0
        for result in results:
            annotated_frame = result.plot()

            if result.boxes is not None:
                num_drones = len(result.boxes)
                total_detections += num_drones
                if num_drones > 0:
                    detection_frames += 1

            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0

            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)

            cv2.putText(annotated_frame,
                        f"Frame: {frame_count}/{total_frames}",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(annotated_frame,
                        f"FPS: {current_fps:.1f}",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(annotated_frame,
                        f"Drones: {num_drones}",
                        (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,0,255) if num_drones > 0 else (255,255,255), 2)
            cv2.putText(annotated_frame,
                        f"Model: YOLOv8",
                        (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            if show_live:
                cv2.imshow("Drone Detection - YOLOv8 | Press Q to quit", annotated_frame)

            if save_output:
                out.write(annotated_frame)

        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% | Frame {frame_count}/{total_frames} "
                  f"| Drones this frame: {num_drones}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user")
            break

    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()

    elapsed_total = time.time() - start_time
    avg_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
    detection_rate = (detection_frames / frame_count * 100) if frame_count > 0 else 0

    print("\n" + "=" * 60)
    print("DETECTION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Frames Processed : {frame_count}")
    print(f"Frames with Drones     : {detection_frames}")
    print(f"Detection Rate         : {detection_rate:.1f}%")
    print(f"Total Drone Detections : {total_detections}")
    print(f"Processing Time        : {elapsed_total:.1f}s")
    print(f"Average FPS            : {avg_fps:.1f}")
    if output_path:
        print(f"Output Saved To        : {output_path}")
    print("=" * 60)

    return {
        'frames_processed': frame_count,
        'detection_frames': detection_frames,
        'detection_rate': detection_rate,
        'total_detections': total_detections,
        'avg_fps': avg_fps,
        'output_path': output_path
    }


def detect_from_youtube(youtube_url, model_path="models/best.pt"):
    try:
        import yt_dlp
    except ImportError:
        print("Install yt-dlp: pip install yt-dlp")
        return

    print(f"Downloading YouTube video...")
    ydl_opts = {
        'format': 'best[height<=720]',
        'outtmpl': 'temp_video.%(ext)s',
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        filename = ydl.prepare_filename(info)

    print(f"Downloaded: {filename}")
    detect_drones_in_video(filename, model_path=model_path)


if __name__ == "__main__":
    detect_drones_in_video(
        video_path="sample/image/drone_video.mp4",   
        model_path="models/best.pt",
        conf_threshold=0.5,
        save_output=True,
        show_live=True
    )