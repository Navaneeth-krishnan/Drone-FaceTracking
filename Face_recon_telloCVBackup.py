from __future__ import print_function
from djitellopy import Tello
import cv2
import pygame
from pygame.locals import *
import numpy as np
import argparse
from imutils import paths
import imutils
import time
import datetime
import math

# Speed of the drone
S = 60
# Frames per second of the pygame window display
FPS = 25




class FrontEnd(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations
            - W and S: Up and down.
    """

    def __init__(self):
        # Init pygame
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])
        global w
        global h
        w = 960
        h = 720
		
        global ref_area #for scaling from pixels to centimeters
        global ref_dist # from camera to face
        global ref_width # of human face
        global ref_w_px # of human face in pixels
        ref_area = 182224 #pixels squared
        ref_dist = 30 # centimeters
        ref_width = 14 # centimeters
        ref_w_px = 295

        global pt 
        pt = datetime.datetime.now() #Past Time

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10
        
        global Override
        Override = True
        self.send_rc_control = False

        # create update timer
        pygame.time.set_timer(USEREVENT + 1, 50)

    def run(self):
	
        global pt

        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        frame_read = self.tello.get_frame_read()

        should_stop = False
        while not should_stop:

            for event in pygame.event.get():
                if event.type == USEREVENT + 1:
                    self.update()
                elif event.type == QUIT:
                    should_stop = True
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == KEYUP:
                    self.keyup(event.key)
            if frame_read.stopped:
                frame_read.stop()
                cv2.destroyAllWindows()
                break

# IMAGE PROCESSING:
            self.screen.fill([0, 0, 0])
            frame = frame_read.frame
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
            cv2.circle(frame,(int(w/2), int(h/2)), 10, (0, 0, 255), 1)
            
            net.setInput(blob)
            detections = net.forward()
	
            BstartX, BstartY, BendX, BendY, max_size, Bconfidence = [0, 0, 0, 0, 0, 0]
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                #print('confidence = %d' %(confidence))
                if confidence < args["confidence"]:
                #if confidence < 0.5:
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
            text = "{:.2f}%".format(Bconfidence * 100)
            y = BstartY - 10 if BstartY - 10 > 10 else BstartY + 10
            cv2.rectangle(frame, (BstartX, BstartY), (BendX, BendY),(0, 0, 255), 2)
            cv2.circle(frame, (int(BstartX + (BendX-BstartX)/2), BstartY + int((BendY-BstartY)/2)), 10, (0, 255, 0), 1)
            area = (BendX - BstartX)*(BendY - BstartY)
            cv2.putText(frame, text, (BstartX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            diff_x = w/2 - (BstartX + (BendX-BstartX)/2)
            diff_y = h/2 - (BstartY + (BendY-BstartY)/2)
            abs_diff = math.sqrt(diff_x**2 + diff_y**2)
            diff_area = 20000 - area
            cv2.putText(frame,'%d, %d, %d cm' %(diff_x/ref_w_px*ref_width, diff_y/ref_w_px*ref_width, diff_area/ref_area*ref_dist), (0,20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),4,cv2.LINE_AA)
            #cv2.imshow('frame', frame)
            #frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame = np.flipud(frame)
            frame_py = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame_py, (0, 0))
            pygame.display.update()
			
            ct = datetime.datetime.now() # get current time
            dtime = ct - pt
            dt = dtime.total_seconds()	# get time difference	
            pt = ct
			
            K = 0.0026
			
            if Override == False:
                if BstartX == 0 and BstartY == 0: 	
                    self.reset()
                else:				
                    #print("Best are %d and %d" %(BstartX,BstartY))    
					
                    self.yaw_velocity = -int(np.sign(diff_x)*math.ceil(100-100*math.exp(-K*abs(diff_x)))) # Clockwise turn positive
                    self.up_down_velocity = int(np.sign(diff_y)*math.ceil(100-100*math.exp(-K*abs(diff_y)))) # Up mvt positive
                    self.for_back_velocity = int(np.sign(diff_area)*math.ceil(100-100*math.exp(-K*abs(diff_area/100)))) # Fwd mvt positive


                    if self.up_down_velocity > 100:
                        self.up_down_velocity = 10
                    elif self.up_down_velocity < -100:
                        self.up_down_velocity = -0	
                    if self.left_right_velocity > 100:
                        self.left_right_velocity = 10
                    elif self.left_right_velocity < -100:
                         self.left_right_velocity = -10
                    if self.for_back_velocity > 100:
                        self.for_back_velocity = 100
                    elif self.for_back_velocity < -100:
                         self.for_back_velocity = -100
                    if self.yaw_velocity > 100:
                        self.yaw_velocity = 100
                    elif self.yaw_velocity < -100:
                         self.yaw_velocity = -100
                    self.update()
					
                    print("%f, %d, %d" %(dt, abs_diff, diff_area))

			
            out.release()
            time.sleep(1 / FPS)

        # Call it always before finishing. I deallocate resources.
        self.tello.end()

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw counter clockwise velocity
            self.yaw_velocity = S

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        global Override
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
            #print("Velocity updated")
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
            time.sleep(2)
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False
        elif key == pygame.K_SPACE: 
            if not Override:
                #global Override
                Override = True
                self.reset()
                print("User Now in Control")
            else:
                #global Override
                Override = False
                print("Now Face Tracking")

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)
									   
    def reset(self):
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0	


def main():
    frontend = FrontEnd()

    # run frontend
    frontend.run()

def init_face_recon():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    prototxt = r"C:\Users\benva\Desktop\UT\2A\2D_and_3D_Scene_Analysis\Drone\deploy.prototxt.txt"
    model = r"C:\Users\benva\Desktop\UT\2A\2D_and_3D_Scene_Analysis\Drone\res10_300x300_ssd_iter_140000.caffemodel"
# load our serialized model from disk
    print("[INFO] loading model...")
    global net
    net = cv2.dnn.readNetFromCaffe(prototxt,model)

# initialize the video stream and allow the camera sensor to warmup
#print("[INFO] starting video stream...")
#cap = cv2.VideoCapture(0)
    global fourcc 
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') #AVI
    global out 
    out = cv2.VideoWriter(r'C:\Users\benva\Desktop\UT\2A\2D_and_3D_Scene_Analysis\Drone\Video_output\outputvideo.avi', fourcc, 25.0, (960, 720)) #25 fps and 960x720 frame
    time.sleep(2.0)


if __name__ == '__main__':
    init_face_recon()
    main()
