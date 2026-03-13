import cv2
import time
import argparse
from ultralytics import YOLO

def benchmark_models(models_list, video_path, show=False, save=False):
    """
    Benchmarks a list of YOLO models (preferably .engine files) against a video.
    """
    for model_path in models_list:
        print(f"\n{'='*50}")
        print(f"Benchmarking Model: {model_path}")
        print(f"{'='*50}")
        
        try:
            model = YOLO(model_path, task='detect')
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            continue
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_vid = None
        
        if save:
            out_name = f"output_{model_path.split('/')[-1].split('.')[0]}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_vid = cv2.VideoWriter(out_name, fourcc, 30.0, (frame_width, frame_height))
            print(f"Saving output to {out_name}")

        frames_processed = 0
        total_inference_time = 0.0

        # Run Warmup (TensorRT usually takes a bit longer on the very first frame)
        print("Warming up model...")
        ret, frame = cap.read()
        if ret:
            model(frame, verbose=False)
        
        print("Starting Benchmark...")
        start_time_real = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Measure pure inference time
            t1 = time.time()
            results = model(frame, verbose=False)
            t2 = time.time()
            
            total_inference_time += (t2 - t1)
            frames_processed += 1
            
            # Rendering and visualization
            if show or save:
                annotated_frame = results[0].plot()
                if save and out_vid:
                    out_vid.write(annotated_frame)
                if show:
                    cv2.imshow(f"Inference: {model_path}", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        end_time_real = time.time()
        cap.release()
        if save and out_vid:
            out_vid.release()
        if show:
            cv2.destroyAllWindows()
            
        actual_time = end_time_real - start_time_real
        if frames_processed > 0:
            avg_infer_fps = frames_processed / total_inference_time
            avg_real_fps = frames_processed / actual_time
            print(f"\nResults for {model_path}:")
            print(f"Total Frames:        {frames_processed}")
            print(f"Total Time (Real):   {actual_time:.2f} s")
            print(f"Avg Pure Inference:  {avg_infer_fps:.2f} FPS")
            print(f"Avg Real-world FPS:  {avg_real_fps:.2f} FPS (Includes video decoding)")
        else:
            print("No frames processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark YOLO models on Jetson.")
    parser.add_argument("--models", nargs="+", required=True, help="List of model paths (.engine preferred) to benchmark")
    parser.add_argument("--video", type=str, required=True, help="Path to the test video file")
    parser.add_argument("--show", action="store_true", help="Display video during inference (Warning: reduces FPS!)")
    parser.add_argument("--save", action="store_true", help="Save the annotated output video")
    
    args = parser.parse_args()
    benchmark_models(args.models, args.video, show=args.show, save=args.save)
