# https://makernambo.com/123

# MIT License
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image
 
import cv2
import numpy as np
from jetbot import Robot
import time
 
# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen
 
 
def gstreamer_pipeline(
    capture_width=160,
    capture_height=120,
    display_width=160,
    display_height=120,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
 
 
def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            cv2.imshow("CSI Camera", img)
            
#####################################################################################################3
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
            blur = cv2.GaussianBlur(gray,(5,5),0)
            
            ret,thresh1 = cv2.threshold(blur,130,255,cv2.THRESH_BINARY_INV)
            
            mask = cv2.erode(thresh1, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            cv2.imshow('mask',mask)
        
            contours,hierarchy = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)
            
            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)
                
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                
                if cx >= 95 and cx <= 125:              
                    print("Turn Left!")
                    robot.left(0.8)
                elif cx >= 39 and cx <= 65:
                    print("Turn Right")
                    robot.right(0.8)
                else:
                    print("go")
                    robot.forward(0.8)
#####################################################################################################3

            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera") 
 
if __name__ == "__main__":
    show_camera()