import cv2
import torch
import numpy as np
import pyttsx3
import datetime

# Initialize TTS engine
engine = pyttsx3.init()

# Load the custom YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='DemoWordsbest-6.pt')

# Open a connection to the webcam (default camera index is 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to save detected objects to a text file
def save_to_file(objects, filename='detected_objects.txt'):
    try:
        with open(filename, 'a') as file:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            file.write(f'{timestamp} - Detected objects: {", ".join(objects)}\n')
    except Exception as e:
        print(f"Error writing to file: {e}")

def speak_objects(objects):
    if not objects:
        return  # Do nothing if no objects are detected
    speech = "Detected objects: "
    for obj in objects:
        speech += obj + ", "
    engine.say(speech)
    engine.runAndWait()

previous_objects = set()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Convert BGR to RGB for the model
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img_rgb)

    # Convert results to numpy format for OpenCV
    annotated_frame = np.asarray(results.render()[0])

    # Convert the annotated frame from RGB to BGR for OpenCV display
    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Extract detected objects
    detections = results.xyxy[0].numpy()
    current_objects = set()
    for detection in detections:
        class_id = int(detection[5])
        confidence = detection[4]
        if confidence > 0.5:
            current_objects.add(results.names[class_id])
    
    # Print detected objects to the terminal
    if current_objects:
        detected_objects_str = ', '.join(current_objects)
        print("Detected objects:", detected_objects_str)
        
        # Save detected objects to file
        save_to_file(current_objects)
        
        # Speak detected objects if there are new ones
        new_objects = current_objects - previous_objects
        if new_objects:
            speak_objects(new_objects)
    
    # Update previous objects
    previous_objects = current_objects
    
    # Display the resulting frame
    cv2.imshow('Object Detection', annotated_frame_bgr)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

print("Code Completed!")
