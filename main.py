import pathlib
import cv2

# Get the path to the Haar cascade file
cascade_path = pathlib.Path(__file__).parent / "haarcascade_frontalface_default.xml"

# Load the Haar cascade
if not cascade_path.exists():
    print(f"Error: Haar cascade file not found at {cascade_path}")
    exit(1)  # Exit the program if the file does not exist

clf = cv2.CascadeClassifier(str(cascade_path))
if clf.empty():  # Check if the cascade was loaded successfully
    print("Error: Failed to load Haar cascade classifier.")
    exit(1)

# Initialize the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():  # Check if the camera is successfully opened
    print("Error: Could not access the camera.")
else:
    print("Camera initialized successfully.")

# Main loop
while camera.isOpened():
    ret, frame = camera.read()  # Capture a frame
    if not ret or frame is None:  # Check if the frame was captured successfully
        print("Error: Failed to capture frame from camera.")
        break  # Exit the loop gracefully if no frame is captured

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar cascade
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles around detected faces
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)

    # Display the frame in a window
    cv2.imshow("Faces", frame)

    # Exit the program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()