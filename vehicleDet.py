import cv2
from sort import *
import math
import numpy as np
from ultralytics import YOLO
import cvzone
from tkinter import *


# Opens the video file 'VideoEXAMPLE.mp4' for reading.
cap = cv2.VideoCapture('VideoEXAMPLE1.mp4')
screen_width, screen_height = 1920, 1080
# Step 1: Get screen resolution
# Initializes the YOLO object detection model using the weights file "yolov8n.pt".
model = YOLO("yolov8n.pt")

# Read the class names from 'classes.txt' file and store them in the 'classNames' list.
classNames = []
with open('classes.txt','r') as f:
    classNames = f.read().splitlines()

##TAKES BOUNDING BOX COORDINATES AND ACCURACY
tracker = Sort(max_age=20)

# frame by frame detection tool.
while True:
    # Reads a frame from the video capture object and stores it in the variable 'frame'.
    ret, frame = cap.read()
    # If the frame is not read successfully (end of the video or an error occurred while reading).
    if not ret:
        cap = cv2.VideoCapture('VideoEXAMPLE1.mp4') # Reopens the video file to restart video reading.
        continue

    detections = np.empty((0,5))


    # Performs object detection on the current frame using the YOLO model.
    result = model(frame, stream=1)
    # Iterates through the result, which contains detection information for the frame.
    for data in result:
        # Extracts the detected bounding boxes from the result.
        boxes = data.boxes
        # Iterates through each detected bounding box.
        for box in boxes:
            # Extracts the coordinates of the bounding box (top-left and bottom-right)
            x1, y1, x2, y2 = box.xyxy[0]

            # Extracts the confidence score (accuracy) of the detected object.
            accuracy = box.conf[0]
            # Extracts the class index of the detected object.
            classinx = box.cls[0]
            # Converts the confidence score to a percentage (rounded up).
            accuracy = math.ceil(accuracy * 100)
            # Converts the class index to an integer
            classinx = int(classinx)

            # Get the name of the detected object using the class index.
            objectdetect = classNames[classinx]
            # If the detected object is a 'car', 'bus', 'motorcycle', or 'bicycle', and the accuracy is above 70%.
            # Score-Threshold techqnique to make sure the detection tool works well
            if objectdetect == 'car' or objectdetect =='truck' or  objectdetect =='motorcycle' or objectdetect =='bicycle' and accuracy > 70 :
                newDetections = np.array([x1,y1,x2,y2,accuracy])
                #Sends our new detections to the detections array we have defined above.
                detections = np.vstack((detections, newDetections))
                #contains id and coordinates
                resultsOfTracker = tracker.update(detections)
                for result in resultsOfTracker:

                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # Draw a rectangle box using rgb values (yellow).

                    newDetections = np.array([x1,y1,x2,y2,accuracy])


                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                    # Put text showing the class name and accuracy percentage near the top-left corner of the bounding box.
                    cvzone.putTextRect(frame,
                                       f'{classNames[classinx]} {accuracy}%',
                                       [x1 - 8, y1 - 12],
                                       thickness=2,
                                       scale=1.5)

                for result in resultsOfTracker:
                    x1, y1, x2, y2, id = map(int, result)

                    # Draw the rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

                    # Calculate text position at the bottom of the rectangle
                    text = f'id#:{[id]}'
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]  # Adjusted font scale
                    text_x = x1
                    text_y = y2 + text_size[1] + 3  # Adjusted vertical spacing

                    # Draw the text with a smaller font size
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)



    window_x = (screen_width - frame.shape[1]) // 2
    window_y = (screen_height - frame.shape[0]) // 2

    #Create a window and display the frame
    cv2.namedWindow('Detector', cv2.WINDOW_NORMAL)  # Create a resizable window
    cv2.moveWindow('Detector', window_x, window_y-150)  # Position the window
    cv2.imshow('Detector', frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press Esc key to exit
        break


