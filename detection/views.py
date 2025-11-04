# detection/views.py
from django.shortcuts import render
from .models import DetectionResult
from django.http import StreamingHttpResponse
from django.conf import settings
from django.core.files import File
import cv2  # <-- This is the correct module name
from ultralytics import YOLO
import threading
from datetime import datetime
import os

# --- Load the model once when the server starts ---
MODEL_PATH = r"C:\Users\yadav\OneDrive\Desktop\violence_detection\dataset\yolo_small_weights.pt"
VIOLENCE_CLASS_NAME = 'violence'
CONFIDENCE_THRESHOLD = 0.50
try:
    print("Loading YOLO model for website stream...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Helper function to save to database ---
def save_detection_event(source_name, confidence, video_path, video_filename):
    print("DATABASE: Saving event...")
    try:
        with open(video_path, 'rb') as f_video:
            django_file = File(f_video, name=video_filename)
            DetectionResult.objects.create(
                video_name=source_name,
                is_violent=True,
                confidence_score=confidence,
                video_file=django_file
            )
        print("DATABASE: Event and video snippet saved successfully.")
    except Exception as e:
        print(f"DATABASE: Error saving to database: {e}")

# --- Main video stream generator (with recording) ---
def video_stream_generator():
    if model is None:
        print("Model not loaded. Cannot start video stream.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # --- 💡💡💡 THIS IS THE CORRECTED CODE 💡💡💡 ---
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # --- End of fix ---

    if fps == 0 or fps < 5: fps = 20

    source_name = "Webcam (Website)"
    video_writer = None
    current_video_path = None
    current_video_filename = None
    highest_confidence_in_event = 0
    violence_event_active = False 
    
    print("Starting video stream and detection...")
    
    snippet_folder = os.path.join(settings.MEDIA_ROOT, 'violence_snippets')
    os.makedirs(snippet_folder, exist_ok=True)
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame.")
                break
            
            results = model(frame, verbose=False) 
            r = results[0]
            annotated_frame = r.plot() 

            fight_detected = False
            highest_confidence_this_frame = 0
            
            for box in r.boxes:
                confidence = box.conf.item()
                if confidence >= CONFIDENCE_THRESHOLD:
                    class_id = int(box.cls.item())
                    detected_class_name = model.names[class_id]
                    if detected_class_name == VIOLENCE_CLASS_NAME:
                        fight_detected = True
                        if confidence > highest_confidence_this_frame:
                            highest_confidence_this_frame = confidence
            
            if fight_detected:
                text = f"🚨 VIOLENCE DETECTED! ({highest_confidence_this_frame*100:.2f}%)"
                cv2.putText(annotated_frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                
                if highest_confidence_this_frame > highest_confidence_in_event:
                    highest_confidence_in_event = highest_confidence_this_frame

                if not violence_event_active:
                    violence_event_active = True
                    print(f"DATABASE: New violence event detected! Starting recording...")
                    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    current_video_filename = f"snippet_{timestamp_str}.mp4"
                    current_video_path = os.path.join(snippet_folder, current_video_filename)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                    video_writer = cv2.VideoWriter(current_video_path, fourcc, fps, (frame_width, frame_height))
                
                if video_writer:
                    video_writer.write(frame) 
            else:
                if violence_event_active:
                    print("DATABASE: Violence event ended. Saving video and resetting flag.")
                    violence_event_active = False
                    if video_writer:
                        video_writer.release()
                        video_writer = None

                    if current_video_path:
                        save_detection_event(source_name, highest_confidence_in_event, current_video_path, current_video_filename)
                    
                    current_video_path = None
                    current_video_filename = None
                    highest_confidence_in_event = 0
            
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                print("Failed to encode frame.")
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        print("Stopping video stream...")
        cap.release()
        
        if violence_event_active:
            print("DATABASE: Stream stopped during a violence event. Saving final snippet...")
            if video_writer:
                video_writer.release()
                video_writer = None
            if current_video_path:
                save_detection_event(source_name, highest_confidence_in_event, current_video_path, current_video_filename)


def show_results(request):
    """
    This view displays the main page with the list of saved events.
    """
    all_results = DetectionResult.objects.all().order_by('-timestamp')
    context = {
        'results': all_results
    }
    return render(request, 'detection/results.html', context)

def video_stream(request):
    """
    This view streams the video feed from the generator.
    """
    return StreamingHttpResponse(video_stream_generator(), 
                                 content_type='multipart/x-mixed-replace; boundary=frame')