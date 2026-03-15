import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import winsound
import urllib.request
import os

# --- MEDIAPIPE TASKS SETUP ---
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
EAR_THRESHOLD = 0.25               # Eye Aspect Ratio threshold (below this means eyes are closed)
CLOSED_EYES_TIME_THRESHOLD = 3.0   # Time in seconds the eyes must be closed to trigger the alarm
ALARM_REPEAT_INTERVAL = 2.0        # Time in seconds between repeating alarm beeps

MAR_THRESHOLD = 0.5                # Mouth Aspect Ratio threshold (above this means yawning)
YAWN_COUNT_LIMIT = 3               # Number of continuous yawns to trigger the drink water alert
YAWN_TIME_WINDOW = 60.0            # Time window (in seconds) to count consecutive yawns; resets if exceeded
WATER_ALARM_DURATION = 5.0         # How long to show the "Drink Water" warning on screen

# Continuous Alarm Configuration
DROWSY_BEEP_LIMIT = 5              # Number of repeated beeps before initiating continuous alarm

# Model Path for Face Landmarker
MODEL_PATH = "face_landmarker.task"

# Face landmarks indices according to MediaPipe Face Mesh
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
# Mouth landmarks: 78 (Left corner), 308 (Right corner), 13 (Top inner lip), 14 (Bottom inner lip)
MOUTH_INDICES = [78, 308, 13, 14]

# Global state to prevent overlapping threads
alarm_playing = False
last_alarm_time = 0.0
water_alarm_playing = False

# Global state for continuous alarm
continuous_alarm_playing = False
beep_count = 0

def download_model():
    """Downloads the MediaPipe Face Landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading FaceLandmarker model... This may take a moment.")
        try:
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                MODEL_PATH
            )
            print("Download complete!")
        except Exception as e:
            print(f"Failed to download model: {e}")
            exit(1)

def calculate_eye_aspect_ratio(eye_landmarks):
    """
    Calculates the Eye Aspect Ratio (EAR) based on 6 landmark points.
    """
    v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    
    if h == 0:
        return 0.0
    
    ear = (v1 + v2) / (2.0 * h)
    return ear

def calculate_mouth_aspect_ratio(mouth_landmarks):
    """
    Calculates the Mouth Aspect Ratio (MAR) based on 4 landmark points (left, right, top, bottom).
    """
    h = np.linalg.norm(np.array(mouth_landmarks[0]) - np.array(mouth_landmarks[1]))
    v = np.linalg.norm(np.array(mouth_landmarks[2]) - np.array(mouth_landmarks[3]))
    
    if h == 0:
        return 0.0
    return v / h

def trigger_beep_alarm():
    """
    Plays a short repeating beep alarm sound using the laptop speaker.
    Increments the global beep_count to track if they are ignoring the alarm.
    """
    global alarm_playing, last_alarm_time, beep_count
    
    current_time = time.time()
    
    # Only thread a new alarm if one isn't playing and the interval has passed
    if not alarm_playing and (current_time - last_alarm_time) >= ALARM_REPEAT_INTERVAL:
        last_alarm_time = current_time
        alarm_playing = True
        beep_count += 1
        
        def play_sound():
            global alarm_playing
            winsound.Beep(2500, 500) # Play a beep 2500 Hz for half a sec (500 ms)
            alarm_playing = False
            
        t = threading.Thread(target=play_sound)
        t.daemon = True
        t.start()

def trigger_continuous_alarm():
    """
    Plays an infinite continuous, abrasive alarm loop.
    This runs continuously until continuous_alarm_playing goes False.
    """
    global continuous_alarm_playing
    if not continuous_alarm_playing:
        continuous_alarm_playing = True
        
        def play_continuous_sound():
            global continuous_alarm_playing
            while continuous_alarm_playing:
                # Play rapid, loud alternating frequencies
                winsound.Beep(3000, 300)
                winsound.Beep(2000, 300)
                
        t = threading.Thread(target=play_continuous_sound)
        t.daemon = True
        t.start()

def stop_continuous_alarm():
    """
    Flags the continuous alarm thread to terminate.
    """
    global continuous_alarm_playing
    continuous_alarm_playing = False

def trigger_water_alarm():
    """
    Plays 3 continuous beep sounds to warn the driver to drink water.
    """
    global water_alarm_playing
    if not water_alarm_playing:
        water_alarm_playing = True
        
        def play_sound():
            global water_alarm_playing
            for _ in range(3):
                winsound.Beep(2000, 300) # Play beep 2000Hz for 300ms
                time.sleep(0.1)
            water_alarm_playing = False
            
        t = threading.Thread(target=play_sound)
        t.daemon = True
        t.start()

def get_facial_landmarks(face_landmarks, frame_width, frame_height):
    """
    Extracts the pixel coordinates for the eyes and mouth based on MediaPipe face mesh landmarks.
    """
    right_eye_coords = []
    left_eye_coords = []
    mouth_coords = []
    
    for idx in RIGHT_EYE_INDICES:
        point = face_landmarks[idx]
        x, y = int(point.x * frame_width), int(point.y * frame_height)
        right_eye_coords.append((x, y))
        
    for idx in LEFT_EYE_INDICES:
        point = face_landmarks[idx]
        x, y = int(point.x * frame_width), int(point.y * frame_height)
        left_eye_coords.append((x, y))
        
    for idx in MOUTH_INDICES:
        point = face_landmarks[idx]
        x, y = int(point.x * frame_width), int(point.y * frame_height)
        mouth_coords.append((x, y))
        
    return right_eye_coords, left_eye_coords, mouth_coords

def detect_face(rgb_frame, detector):
    """
    Uses MediaPipe to detect the face landmarks in the given RGB image.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    return detection_result

def main():
    global beep_count, continuous_alarm_playing

    # 1. Download Model
    download_model()
    
    # 2. Initialize the built-in laptop camera (Index 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Tracking states
    eyes_closed_start_time = None
    
    # Yawn tracking states
    yawn_count = 0
    is_yawning = False
    last_yawn_time = 0.0
    show_water_warning_until = 0.0

    # 3. Setup MediaPipe Face Landmarker Options
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1)
        
    # 4. Initialize the FaceLandmarker detector
    with vision.FaceLandmarker.create_from_options(options) as detector:
        print("Starting Driver Alert System (Drowsiness & Yawning)... Press 'ESC' to exit.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the frame horizontally to display like a mirror
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape

            # Convert to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 5. Detect Face and Landmarks
            detection_result = detect_face(rgb_frame, detector)

            if detection_result.face_landmarks:
                # We only track the first face detected
                face_landmarks = detection_result.face_landmarks[0]
                
                # --- DRAW EXPLICIT FACE LANDMARKS AND MESH ---
                for landmark in face_landmarks:
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)
                    # Draw microscopic white dots for the 468 facial contour landmarks
                    cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

                # 6. Extract Eye & Mouth Landmarks
                right_eye, left_eye, mouth = get_facial_landmarks(face_landmarks, frame_width, frame_height)
                
                # Highlight the eye and mouth regions heavily with green dots
                for point in right_eye + left_eye + mouth:
                    cv2.circle(frame, point, 2, (0, 255, 0), -1)

                # 7. Calculate Aspect Ratios (EAR & MAR)
                ear_right = calculate_eye_aspect_ratio(right_eye)
                ear_left = calculate_eye_aspect_ratio(left_eye)
                avg_ear = (ear_right + ear_left) / 2.0
                
                mar = calculate_mouth_aspect_ratio(mouth)

                # 8. Check Yawning continuously
                current_time = time.time()
                
                # Reset yawn count if user hasn't yawned in a long time (YAWN_TIME_WINDOW)
                if yawn_count > 0 and (current_time - last_yawn_time) > YAWN_TIME_WINDOW:
                    yawn_count = 0
                
                if mar > MAR_THRESHOLD:
                    if not is_yawning:
                        is_yawning = True
                else:
                    if is_yawning:
                        # Mouth went from open to closed -> Yawn completed
                        is_yawning = False
                        yawn_count += 1
                        last_yawn_time = current_time
                        
                        if yawn_count >= YAWN_COUNT_LIMIT:
                            trigger_water_alarm()
                            show_water_warning_until = current_time + WATER_ALARM_DURATION
                            yawn_count = 0 # Reset count after triggering alarm

                # 9. Check drowsiness continuously
                is_drowsy = False
                if avg_ear < EAR_THRESHOLD:
                    # Eyes are closed
                    if eyes_closed_start_time is None:
                        eyes_closed_start_time = time.time()
                        
                    elapsed_time = time.time() - eyes_closed_start_time
                    
                    if elapsed_time >= CLOSED_EYES_TIME_THRESHOLD:
                        is_drowsy = True
                else:
                    # Eyes are OPEN
                    eyes_closed_start_time = None
                    is_drowsy = False
                    # Instantly reset beep counts and stop continuous alarm if eyes open
                    beep_count = 0
                    if continuous_alarm_playing:
                        stop_continuous_alarm()

                # 10. Display Visual Alerts based on Priority
                # Water Warning (highest priority full screen display)
                if current_time < show_water_warning_until:
                    # Blue warning rectangle for drinking water
                    cv2.rectangle(frame, (0, 0), (frame_width, frame_height), (255, 100, 0), 10)
                    cv2.putText(frame, "WARNING: DRINK WATER!", (50, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 100, 0), 4)
                    
                    # Keep eye status indicator active even during water warn
                    status_color = (0, 0, 255) if is_drowsy else (0, 255, 0)
                    status_text = "Status: Driver Drowsy" if is_drowsy else "Status: Driver Alert"
                    cv2.putText(frame, status_text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                    
                    if is_drowsy:
                        if beep_count >= DROWSY_BEEP_LIMIT:
                            trigger_continuous_alarm()
                        else:
                            trigger_beep_alarm()
                        
                elif is_drowsy:
                    # System triggers alarm overlapping checks on the 2 sec interval
                    if beep_count >= DROWSY_BEEP_LIMIT:
                        # Driver ignored 5 beeps, trigger infinite continuous alarm
                        trigger_continuous_alarm()
                        # Override UI for Extreme Danger
                        cv2.putText(frame, "Status: CRITICAL FATIGUE!", (30, 80), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        cv2.rectangle(frame, (0, 0), (frame_width, frame_height), (0, 0, 255), -1) # Solid Red Screen Flash
                    else:
                        trigger_beep_alarm()
                        cv2.putText(frame, "Status: Driver Drowsy", (30, 80), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.rectangle(frame, (0, 0), (frame_width, frame_height), (0, 0, 255), 10)
                                
                    # UI Main Drowsiness Warning Text Overlay
                    # To remain visible through transparent/solid flashes, draw text vividly
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255) if beep_count >= DROWSY_BEEP_LIMIT else (0, 0, 255), 4)

                else:
                    # Driver is Alert and fine
                    # Small Status Indicator showing Driver Normal/Alert in Top Left
                    cv2.putText(frame, "Status: Driver Alert", (30, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the computed EAR, MAR, and Yawn Count explicitly for judges
                cv2.putText(frame, f"EAR: {avg_ear:.2f} | MAR: {mar:.2f}", (30, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Yawns: {yawn_count} | Beep Level: {beep_count}", (frame_width - 320, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 11. Render output on screen
            cv2.imshow('Driver Drowsiness Detection System', frame)

            # 12. Handle Exit
            # Program runs continuously until user presses ESC (ASCII 27)
            key = cv2.waitKey(5) & 0xFF
            if key == 27: 
                break

    # Once ESC is pressed, completely release resources and stop any continuous audio
    stop_continuous_alarm()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
