import os
import cv2
import time

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Create a folder for saving images if it doesn't exist
folder = "images"
if not os.path.exists(folder):
    os.makedirs(folder)

# Time interval between captures
interval = 5

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Create a filename based on the current time
        filename = os.path.join(folder, f"{time.time()}.jpg")
        # Save the captured frame to the folder
        cv2.imwrite(filename, frame)

    # Wait for the specified interval
    time.sleep(interval)

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
