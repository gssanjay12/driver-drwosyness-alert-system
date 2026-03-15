# Driver Drowsiness Detection System

This project develops an AI-powered driver safety system that monitors the driver using a camera and computer vision techniques. It detects drowsiness and yawning in real time and triggers a beep alarm to alert the driver and help prevent accidents. Built specifically for hackathon prototype demonstrations.

## Features
- Real-time face tracking and face mesh processing using **MediaPipe**.
- Detects the driver's **Eye Aspect Ratio (EAR)** using **OpenCV** to track eyelid closure.
- Checks if the eyes are closed continuously for **more than 3 seconds**.
- Triggers a loud built-in **system alarm** via multiple-threaded `winsound` in Python (Windows specific).
- Provides a clean, informative UI visualizing:
  - Complex Face Mesh Layout
  - Exact green tracking dots along the inner and outer eye contours
  - Eye Status (`OPEN` vs `CLOSED`)
  - Warning Text Overlays in vibrant red for the `DROWSINESS ALERT!`

## Prerequisites & Installation

To run this application, ensure you have Python 3.8+ installed on your computer.

### Step 1: Install Required Libraries
Open your terminal (or command prompt) and run this command:
```bash
pip install opencv-python mediapipe numpy
```

*(Note: `time`, `threading`, and `winsound` are built directly into standard Python on Windows, so you do not need to install them)*

## Setup and Running the Project

1. Verify your physical laptop webcam is unblocked and operable.
2. Open a terminal inside this project directory (`driver-alert-sys`).
3. Run the Python application:
```bash
python main.py
```

## How It Works

1. `detect_face()` - Processes the camera stream converting it from BGR to RGB, leveraging `mediapipe.solutions.face_mesh` for pinpoint accuracy.
2. `detect_eyes()` - Finds explicitly 6 data points specifically around the Left Eye and 6 on the Right Eye.
3. `calculate_eye_aspect_ratio()` - Formulates mathematically the vertical length versus horizontal length. If it drops below `EAR_THRESHOLD=0.25`, the eyes are calculated as closed.
4. `trigger_alarm()` - Non-blocking implementation via `threading.Thread` calling the system's `winsound.Beep(2500, 1000)` which creates a very loud, sharp internal computer alert noise.

To **quit** the application, ensure the newly popped up video frame is deeply focused/clicked on, and press your `ESC` key.
