# py C:\Users\benva\Desktop\UT\2A\2D_and_3D_Scene_Analysis\Drone\detect_faces_video.py

# import the necessary packages
from __future__ import print_function
#from imutils.video import VideoStream
import numpy as np
import argparse
from imutils import paths
import imutils
import time
import cv2

# construct the argument parse and parse the arguments (to change confidence interval if wanted)
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

prototxt = r"C:\Users\benva\Desktop\UT\2A\2D_and_3D_Scene_Analysis\Drone\deploy.prototxt.txt"
model = r"C:\Users\benva\Desktop\UT\2A\2D_and_3D_Scene_Analysis\Drone\res10_300x300_ssd_iter_140000.caffemodel"

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt,model)

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID') #AVI
out = cv2.VideoWriter(r'C:\Users\benva\Desktop\UT\2A\2D_and_3D_Scene_Analysis\Drone\Video_output\outputvideo.avi', fourcc, 30.0, (400, 300)) #20 fps and 320x240 frame
time.sleep(2.0)


# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 240 pixels
	ret, frame = cap.read()
	frame = imutils.resize(frame, width=240)
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
	cv2.circle(frame,(int(w/2), int(h/2)), 10, (0, 0, 255), 1)
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	
	BstartX, BstartY, BendX, BendY, max_size, Bconfidence = [0, 0, 0, 0, 0, 0]
	

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < args["confidence"]:
			continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		
		#Find closest face
		test_area = (endX-startX)*(endY-startY)
		if test_area>=max_size:
			max_size = test_area
			BstartX = startX
			BstartY = startY
			BendX = endX
			BendY = endY
			Bconfidence = confidence
 	# draw the bounding box of the face along with the associated
	# probability
	text = "{:.2f}%".format(Bconfidence * 100)
	y = BstartY - 10 if BstartY - 10 > 10 else BstartY + 10
	cv2.rectangle(frame, (BstartX, BstartY), (BendX, BendY),(0, 0, 255), 2)
	cv2.circle(frame, (int(BstartX + (BendX-BstartX)/2), BstartY + int((BendY-BstartY)/2)), 10, (0, 255, 0), 1)
	area = (BendX - BstartX)*(BendY - BstartY)
	cv2.putText(frame, text, (BstartX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	diff_x = w/2 - (BstartX + (BendX-BstartX)/2)
	diff_y = h/2 - (BstartY + (BendY-BstartY)/2)
	cv2.putText(frame,'%d, %d, %d' %(diff_x, diff_y, area), (0,20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),4,cv2.LINE_AA)

	# # show the output frame
	out.write(frame)
	cv2.imshow('frame', frame)
	key = cv2.waitKey(1) & 0xFF
 
	#  if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cap.release()
out.release()
cv2.destroyAllWindows()