# -*- coding: utf-8 -*-
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

class CafeMonitoring:
    def __init__(self, model_path, video_path, output_path):
        # Initialize YOLO
        self.model = YOLO(model_path)
        
        # Initialize DeepSORT
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            embedder='mobilenet'
        )
        
        # Video paths
        self.video_path = video_path
        self.output_path = output_path
        
        # Tracking data storage
        self.tracks_data = {}
        self.completed_tracks = {}
        
        # Colors for visualization
        self.colors = {
            'Barista': (0, 255, 0),  # Green
            'person': (255, 0, 0)    # Blue
        }
    
    def process_video(self):
        # Initialize video capture
        cap = cv2.VideoCapture(self.video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer
        out = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        # Process frames with progress bar
        pbar = tqdm(total=total_frames, desc='Processing video')
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Detect objects
            results = self.model.predict(frame, conf=0.5, show=False)
            
            # Process detections
            detections = []
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    
                    # DeepSORT format: [x1,y1,w,h], conf, class
                    w = x2 - x1
                    h = y2 - y1
                    detections.append(([x1,y1,w,h], conf, class_name))
            
            # Update tracks
            tracks = self.tracker.update_tracks(detections, frame=frame)
            
            # Process and visualize tracks
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_name = track.det_class  # Menggunakan det_class 
                
                # Store track data
                if track_id not in self.tracks_data:
                    self.tracks_data[track_id] = {
                        'class_name': class_name,
                        'start_frame': frame_count,
                        'positions': []
                    }
                
                self.tracks_data[track_id]['positions'].append({
                    'frame': frame_count,
                    'bbox': ltrb
                })
                
                # Draw visualization
                x1, y1, x2, y2 = map(int, ltrb)
                color = self.colors.get(class_name, (0, 255, 0))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Calculate duration
                duration = (frame_count - self.tracks_data[track_id]['start_frame']) / fps
                
                # Draw labels
                if class_name == "Barista": #you can use any class name you want based on your YOLO Object Class
                    label = f"#{track_id} {class_name}"
                    info = f"Work: {duration:.1f}s"
                else:
                    label = f"#{track_id} Customer" #you can use any class name you want based on your YOLO Object Class
                    info = f"Wait: {duration:.1f}s"
                
                cv2.putText(frame, label, (x1, y1-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, info, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            # Write frame
            out.write(frame)
            pbar.update(1)
        
        # Clean up
        pbar.close()
        cap.release()
        out.release()
        
        # Save tracking data
        self.save_tracking_data()
    
    def save_tracking_data(self):
        # Prepare data for CSV
        tracking_records = []
        
        for track_id, data in self.tracks_data.items():
            record = {
                'track_id': track_id,
                'class_name': data['class_name'],
                'start_frame': data['start_frame'],
                'total_frames': len(data['positions']),
                'duration_seconds': len(data['positions']) / 30  # Assuming 30 fps
            }
            tracking_records.append(record)
        
        # Save to CSV
        df = pd.DataFrame(tracking_records)
        df.to_csv('data/output/tracking_results.csv', index=False)
        
        # Print summary
        print("\nTracking Summary:")
        print(f"Total Baristas: {len(df[df['class_name'] == 'Barista'])}") #you can use any class name you want based on your YOLO Object Class
        print(f"Total Customers: {len(df[df['class_name'] == 'person'])}") #you can use any class name you want based on your YOLO Object Class
        print(f"Average Customer Wait Time: {df[df['class_name'] == 'person']['duration_seconds'].mean():.2f} seconds")

def main():
    # Initialize and run tracking
    tracker = CafeMonitoring(
        model_path='data/models/best.pt',
        video_path='data/input/cafe.mp4',
        output_path='data/output/output.mp4'
    )
    
    tracker.process_video()

if __name__ == "__main__":
    main()
