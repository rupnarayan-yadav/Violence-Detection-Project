
import cv2
from ultralytics import YOLO
import os  
import sys 
import django 
# import argparse # We are not using argparse anymore
from datetime import datetime
from django.conf import settings # To get MEDIA_ROOT
from django.core.files import File # To save the file to the model

# --- DJANGO SETUP BLOCK ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web_app.settings')
django.setup()

# --- IMPORT YOUR MODEL ---
from detection.models import DetectionResult

# ====================================================================
# --- 1. CONFIGURATION ---
# ====================================================================

# 💡 SET YOUR SOURCE HERE: 'WEBCAM' or 'VIDEO_FILE'
INPUT_SOURCE = 'WEBCAM' 

# --- Model and Path Settings ---
model_path = r"C:\Users\yadav\OneDrive\Desktop\violence_detection\dataset\yolo_small_weights.pt"
video_file_path = r"C:\Users\yadav\OneDrive\Desktop\Violence_Detection\sample_video\fight.mp4" # Path for VIDEO_FILE

# --- Detection Parameters ---
VIOLENCE_CLASS_NAME = 'violence' # Make sure this matches the class name in the model
CONFIDENCE_THRESHOLD = 0.50 

# ====================================================================
# --- 2. MODEL LOADING ---
# ====================================================================
try:
    model = YOLO(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ====================================================================
# --- 3. VIDEO CAPTURE SETUP ---
# ====================================================================

if INPUT_SOURCE == 'WEBCAM':
    cap = cv2.VideoCapture(0) # Use webcam 0
    source_name = "Webcam"
    is_webcam = True
elif INPUT_SOURCE == 'VIDEO_FILE':
    cap = cv2.VideoCapture(video_file_path)
    source_name = f"Video File: {os.path.basename(video_file_path)}"
    is_webcam = False
else:
    print("Error: INPUT_SOURCE must be 'WEBCAM' or 'VIDEO_FILE'.")
    exit()

if not cap.isOpened():
    print(f"Error: Could not open {source_name}.")
    exit()

# --- Get video properties for saving ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Handle webcam FPS being 0 or very low
if fps == 0 or fps < 5:
    fps = 20  # Assume 20 FPS for webcam or problematic video
    
print(f"{source_name} opened successfully. Press 'q' to quit.")

# --- Video writer objects ---
video_writer = None
current_video_path = None
current_video_filename = None
highest_confidence_in_event = 0

# ====================================================================
# --- 💡 NEW: HELPER FUNCTION TO SAVE TO DATABASE ---
# ====================================================================
def save_detection_event(source_name, confidence, video_path, video_filename):
    """
    Helper function to save the detection result to the database.
    """
    print("DATABASE: Saving event...")
    try:
        # Open the video file that was just saved
        with open(video_path, 'rb') as f_video:
            django_file = File(f_video, name=video_filename)
            
            # Create the database record
            DetectionResult.objects.create(
                video_name=source_name,
                is_violent=True,
                confidence_score=confidence,
                video_file=django_file
            )
        
        print("DATABASE: Event and video snippet saved successfully.")
        
    except Exception as e:
        print(f"DATABASE: Error saving to database: {e}")

# ====================================================================
# --- 4. MAIN PROCESSING LOOP ---
# ====================================================================

violence_event_active = False 
print("Starting detection loop. Will save new events to database...")

while True:
    success, frame = cap.read()
    
    if not success:
        if not is_webcam: # If it's a video file, loop it
            print("End of video stream. Restarting video.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop the video
            continue
        else: # If it's a webcam, break
            print("Failed to grab frame (Webcam issue).")
            break

    # --- DETECTION INFERENCE ---
    results = model(frame, verbose=False) 
    r = results[0]
    annotated_frame = r.plot() # Get frame with boxes drawn

    fight_detected = False
    highest_confidence_this_frame = 0
    
    # --- LOOP THROUGH BOXES ---
    for box in r.boxes:
        confidence = box.conf.item()
        class_id = int(box.cls.item())
        detected_class_name = model.names[class_id]
        
        # --- 💡💡💡 ADDED THIS PRINT STATEMENT 💡💡💡 ---
        print(f"DEBUG: Detected '{detected_class_name}' with {confidence*100:.2f}% confidence")
        
        if confidence >= CONFIDENCE_THRESHOLD:
            # Check if the detected class is the one we care about
            if detected_class_name == VIOLENCE_CLASS_NAME:
                fight_detected = True
                if confidence > highest_confidence_this_frame:
                    highest_confidence_this_frame = confidence
    
    # --- VIDEO RECORDING & DATABASE LOGIC ---
    if fight_detected:
        text = f"🚨 VIOLENCE DETECTED! ({highest_confidence_this_frame*100:.2f}%)"
        cv2.putText(annotated_frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        
        # if highest_confidence_this_frame > highest_confidence_in_event:
        #     highest_confidence_in_event = highest_confidence_in_event

        if highest_confidence_this_frame > highest_confidence_in_event:
            highest_confidence_in_event = highest_confidence_this_frame # <-- FIX

        if not violence_event_active:
            # --- START A NEW RECORDING ---
            violence_event_active = True
            print(f"DATABASE: New violence event detected! Starting recording...")
            
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            current_video_filename = f"snippet_{timestamp_str}.mp4"
            current_video_path = os.path.join(settings.MEDIA_ROOT, 'violence_snippets', current_video_filename)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video_writer = cv2.VideoWriter(current_video_path, fourcc, fps, (frame_width, frame_height))
        
        if video_writer:
            video_writer.write(frame) # Write original frame

    else:
        # No violence detected in this frame
        if violence_event_active:
            # --- STOP RECORDING & SAVE TO DATABASE ---
            print("DATABASE: Violence event ended. Saving video and resetting flag.")
            violence_event_active = False
            if video_writer:
                video_writer.release()
                video_writer = None

            # --- Use helper function to save ---
            if current_video_path:
                save_detection_event(
                    source_name, 
                    highest_confidence_in_event, 
                    current_video_path, 
                    current_video_filename
                )
            
            # Reset for the next event
            current_video_path = None
            current_video_filename = None
            highest_confidence_in_event = 0
            
    # Display the frame
    cv2.imshow("Violence Detection", annotated_frame)
    
    wait_time = 1
    
    if cv2.waitKey(wait_time) & 0xFF == ord("q"):
        break

# ====================================================================
# --- 5. CLEANUP (MODIFIED) ---
# ====================================================================
cap.release()
cv2.destroyAllWindows()

# --- CHECK IF SCRIPT EXITED DURING A VIOLENCE EVENT ---
if violence_event_active:
    print("DATABASE: Script exited during a violence event. Saving final snippet...")
    if video_writer:
        video_writer.release()
        video_writer = None
    
    # --- Save the final event to the database ---
    if current_video_path:
        save_detection_event(
            source_name, 
            highest_confidence_in_event, 
            current_video_path, 
            current_video_filename
        )
else:
    # This is safe cleanup if the script exits normally
    if video_writer:
        video_writer.release()
        print("Video writer released on exit (no active event).")

print("Capture released and windows closed.")