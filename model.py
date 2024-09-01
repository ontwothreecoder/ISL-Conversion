import cv2
import torch
import numpy as np

# Load the custom YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='Final_best-5.pt')

# Open a connection to the webcam (default camera index is 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img_rgb)

    # Convert results to numpy format for OpenCV
    annotated_frame = np.asarray(results.render()[0])

    #img_rgb_1 = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Display the resulting frame
    cv2.imshow('Object Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

print("Code Completed!")