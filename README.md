
# Vehicle Detection and Tracking using YOLO and SORT

This project demonstrates how to detect and track vehicles in a video stream using the YOLO (You Only Look Once) object detection model and the SORT (Simple Online and Realtime Tracking) algorithm. Combining deep learning-based object detection and online object tracking techniques accurately identifies and tracks vehicle movement. 







## Confidence Score Thresholding

The Score-Threshold technique was used to make sure that the vehicles being tracked were of accurate type (eg: Truck, car, etc) (>70%).



## Use Case:
**Traffic Optimization:**

Traffic Flow: By tracking the movement of vehicles, Using this system, you can analyze traffic flow and identify congestion points. with this algorithm, you have access to the vehicle type and identity assigned. with this information during congestion, a thorough analysis could be made to identify the reason for the anomaly. The system can determine the percentage of time each lane remains occupied by vehicles, aiding in optimizing signal timings.

_Problem:_ During congestion most vehicles under analysis are identified as "Trucks".

_Solution:_ A HOV (High Occupancy Vehicle) Lane has to be built into the highway for trucks in order to optimize traffic flow.


## Getting Started
__Prerequisites:__

Python 3.x

OpenCV

NumPy

ultralytics (for YOLO)

cvzone (for drawing text and rectangles)

YOLOv8n Model - Download and place in the project directory.

Video file (e.g., Vid3.mp4) for testing.

__Download the requirments.txt file to get started.__


## RESULT
<img width="802" alt="Screenshot 2023-08-28 at 1 29 43 AM" src="https://github.com/OfficalGlitch/VehicleDetectionTool/assets/77417270/f1d16d25-cfed-4142-b393-add79a4d525b">

<img width="222" alt="screenshot" src="https://github.com/OfficalGlitch/VehicleDetectionTool/assets/77417270/380e7bd9-4138-4631-bb37-178a47935153">


## License
SORT is released under the GPL License (refer to the LICENSE file for details) to promote the open use of the tracker and future improvements.
