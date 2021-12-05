import face_recognition
import cv2
import numpy as np
from jetbot import Robot
import time

robot = Robot()

def linetracing():
    print("linetracing 998")
    robot.stop()
    time.sleep(0.5)
    # MIT License
    # Copyright (c) 2019 JetsonHacks
    # See license
    # Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
    # NVIDIA Jetson Nano Developer Kit using OpenCV
    # Drivers for the camera and OpenCV are included in the base image
   
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
        webcam = cv2.VideoCapture(1)
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
                       
                   
                    # Reading the video from the
                    # webcam in image frames
                    _, imageFrame = webcam.read()
                   
                    # Convert the imageFrame in
                    # BGR(RGB color space) to
                    # HSV(hue-saturation-value)
                    # color space
                    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
                   
                    # Set range for green color and
                    # define mask
                    green_lower = np.array([25, 52, 72], np.uint8)
                    green_upper = np.array([102, 255, 255], np.uint8)
                    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
               
                    # Set range for green color and
                    # define mask
                    green_lower = np.array([25, 52, 72], np.uint8)
                    green_upper = np.array([102, 255, 255], np.uint8)
                    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
                   
                     # Morphological Transform, Dilation
                    # for each color and bitwise_and operator
                    # between imageFrame and mask determines
                    # to detect only that particular color
                    kernal = np.ones((5, 5), "uint8")
                   
                    # For green color
                    green_mask = cv2.dilate(green_mask, kernal)
                    res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                                mask = green_mask)
                   
                    # Creating contour to track green color
                    contours, hierarchy = cv2.findContours(green_mask,
                                                        cv2.RETR_TREE,
                                                        cv2.CHAIN_APPROX_SIMPLE)
                   
                    for pic, contour in enumerate(contours):
                        area = cv2.contourArea(contour)
                        if(area > 300):
                            x, y, w, h = cv2.boundingRect(contour)
                            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                                    (x + w, y + h),
                                                    (0, 255, 0), 2)
                           
                            cv2.putText(imageFrame, "Green Colour", (x, y),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0, (0, 255, 0))
                            robot.stop()
                            time.sleep(0.5)  
                           
                    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                                   
                   
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

    show_camera()

def main():
    # This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
    # other example, but it includes some basic performance tweaks to make things run a lot faster:
    #   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
    #   2. Only detect faces in every other frame of video.

    # PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
    # OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
    # specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(1)

    # Load a sample picture and learn how to recognize it.
    obama_image = face_recognition.load_image_file("obama.jpg")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    # Load a second sample picture and learn how to recognize it.
    biden_image = face_recognition.load_image_file("biden.jpg")
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

    # Load a second sample picture and learn how to recognize it.
    goeun_image = face_recognition.load_image_file("hwanggim.jpg")
    goeun_face_encoding = face_recognition.face_encodings(goeun_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        obama_face_encoding,
        biden_face_encoding,
        goeun_face_encoding
    ]
    known_face_names = [
        "Barack Obama",
        "Joe Biden",
        "Hwang Go Eun"
    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use namethe known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            if(name == "Hwang Go Eun"):
                print("Goeun is here")
                video_capture.release()
                cv2.destroyAllWindows()
                linetracing()

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()