# Drone-FaceTracking
https://youtu.be/kwGHcA9aoC8

## Details

In order to track a person indoors using a drone, a face recognition algorithm together with a controller is used. A face detector was chosen over a pedestrian detector as it is more reliable in tight spaces. This is because a human face is easier visible by a drone than a full human body, where the drone has to be further away from its target to detect it.
As a base to control the DJI Tello, the example script in the Python Tello Library named DJITelloPy is used. The script is used to set up the connection between the Tello and a PC, get the video feed of the drone and control the Tello using the keyboard of a PC.
To detect a face, a pre-trained deep-learning neural network (DNN) from the open source python image processing library OpenCV is used. This DNN takes a frame from the drone as a BGR-image and outputs the bounding boxes of the detected faces.
As a first step, once the DNN has been loaded, the frame from the Tello is passed through a blob feature detection algorithm. These blob locations are then passed onto the DNN which looks for the combination of blobs that best resembles a face.
The DNN outputs the left upper and right lower point of the bounding boxes together with the confidences of the detections. If the confidence is above a certain threshold, the coordinates of the bounding box are stored. 
It can be noted that the detection is accepted if it is lower then a certain confidence. That is because the DNN does not really output the confidence of the detection but rather the likely-hood of it being a false-positive, 1-confidence.