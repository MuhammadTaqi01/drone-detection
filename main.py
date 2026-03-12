import argparse
from src.train import train_model, plot_training_results
from src.inference_image import detect_drones_in_image, detect_batch_images
from src.inference_video import detect_drones_in_video
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8-Based Drone/UAV Detection System"
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'detect_image', 'detect_video', 'evaluate'],
        required=True,
        help='Mode to run'
    )
    parser.add_argument('--source', type=str, help='Image or video file path')
    parser.add_argument('--model',  type=str, default='models/best.pt', help='Model path')
    parser.add_argument('--conf',   type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--epochs', type=int,   default=50,  help='Training epochs')

    args = parser.parse_args()

    if args.mode == 'train':
        print("Starting Training...")
        train_model(epochs=args.epochs)
        plot_training_results()

    elif args.mode == 'detect_image':
        if not args.source:
            print("Please provide --source image path")
            return
        detect_drones_in_image(args.source, args.model, args.conf)

    elif args.mode == 'detect_video':
        if not args.source:
            print("Please provide --source video path")
            return
        detect_drones_in_video(args.source, args.model, args.conf)

    elif args.mode == 'evaluate':
        evaluate_model(args.model)

if __name__ == "__main__":
    main()