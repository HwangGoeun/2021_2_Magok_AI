import cv2
import numpy as np
from jetbot import Robot
import time
import face_recognition

robot = Robot()

####################################################################################################################

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

    # Load a third sample picture and learn how to recognize it.
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

                # Or instead, use the known face with the smallest distance to the new face
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

            if name == "Hwang Go Eun":
                video_capture.release()
                linetracing()

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

####################################################################################################################

def linetracing():
    camera = cv2.VideoCapture(0)
    camera.set(3,160) 
    camera.set(4,120)

    while( camera.isOpened() ):
        ret, frame = camera.read()
        frame = cv2.flip(frame,-1)
        cv2.imshow('normal',frame)
        
        crop_img =frame[60:120, 0:160]
        
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
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
                motor.forward(0.8)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()