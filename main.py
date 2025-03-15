import pathlib
import cv2
from deepface import DeepFace

# Get the path to the Haar cascade file
cascade_path = pathlib.Path(__file__).parent / "haarcascade_frontalface_default.xml"

# Load the Haar cascade
if not cascade_path.exists():
    print(f"Error: Haar cascade file not found at {cascade_path}")
    exit(1)

clf = cv2.CascadeClassifier(str(cascade_path))
if clf.empty():
    print("Error: Failed to load Haar cascade classifier.")
    exit(1)

# Initialize the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access the camera.")
    exit(1)
else:
    print("Camera initialized successfully.")

frame_count = 0  # Counter to limit DeepFace processing frequency
dominant_emotion = "Unknown"  # 🔹 Initialize variable

# Main loop
while camera.isOpened():
    ret, frame = camera.read()
    if not ret or frame is None:
        print("Error: Failed to capture frame from camera.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar cascade
    faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:  # 🔹 Process emotion detection only if a face is detected
        for (x, y, width, height) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)

            # Process emotion detection every 10 frames (to reduce lag)
            if frame_count % 10 == 0:
                face_roi = frame[y:y + height, x:x + width]  # Extract face ROI

                try:
                    # Perform emotion detection
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    dominant_emotion = result[0]['dominant_emotion']  # Extract emotion
                except Exception as e:
                    print(f"Error in DeepFace: {e}")
                    dominant_emotion = "Unknown"

            # Display emotion text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, dominant_emotion, (x, y - 10), font, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Emotion Detection", frame)

    frame_count += 1  # Increment frame counter

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()
