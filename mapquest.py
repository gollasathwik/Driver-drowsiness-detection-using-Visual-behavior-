import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk, ImageDraw
from tkinter.filedialog import askopenfilename
import numpy as np
import imutils
import time
import cv2
import os
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import playsound
import time
import dlib
import smtplib
import requests
import smtplib
import telegram
import pyttsx3
import asyncio
import telebot
import geopy
import geocoder
from geopy.geocoders import Nominatim
import requests

import math
window = tk.Tk()
window.title("DROWSINESS DETECTION SYSTEM")
window.geometry('1900x800')

# Load the image file and create a label to display it
bg_image = tk.PhotoImage(file="pc3.png")
bg_label = tk.Label(window, image=bg_image)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create a label for the text and set its compound option to display the text on the image
message = tk.Label(window, text="DROWSINESS DETECTION SYSTEM", fg="white", bg="black",
                   width=30, height=2, font=('times', 30, 'bold'), compound=tk.CENTER)
message.place(relx=0.995, rely=0.05, anchor=tk.NE)


# One time initialization
engine = pyttsx3.init()
geopy.geocoders.options.default_user_agent = "dda"

# Set properties _before_ you add things to say
engine.setProperty('rate', 125)    # Speed percent (can go over 100)
engine.setProperty('volume', 1)  # Volume 0-1

def duration(start_time):
    """
    Function to calculate duration and format it as hh:mm:ss
    """
    current_time = time.time()
    duration = int(current_time - start_time)
    formatted_duration = time.strftime('%H:%M:%S', time.gmtime(duration))
    return formatted_duration
def getheadpose(frame,shape,size):
    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                                (shape[33, :]),     # Nose tip
                                (shape[8,  :]),     # Chin
                                (shape[36, :]),     # Left eye left corner
                                (shape[45, :]),     # Right eye right corne
                                (shape[48, :]),     # Left Mouth corner
                                (shape[54, :])      # Right mouth corner
                            ], dtype="double")
     
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner                     
                            ])

    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
    #print ("Camera Matrix :\n {0}".format(camera_matrix))

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
     
    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
     
    for p in image_points:
        cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    cv2.line(frame, p1, p2, (255,0,0), 2)

    # calculate the head tilt angle
    rmat, _ = cv2.Rodrigues(rotation_vector)
    proj = np.dot(rmat, np.array([0, 0, 1]))
    angle = math.atan2(proj[1], proj[0]) * 180 / math.pi
    # apply threshold for head tilt angle
    '''if abs(angle) > 25:
        print("Head tilted!")
    else:
        print("Head straight")
'''
    return p1,p2,angle

def mouth_aspect_ratio(mouth):
    # Compute the euclidean distances between the three sets
    # of vertical mouth landmarks (x, y)-coordinates
    A = np.linalg.norm(mouth[13] - mouth[19])
    B = np.linalg.norm(mouth[14] - mouth[18])
    C = np.linalg.norm(mouth[15] - mouth[17])

    # Compute the euclidean distance between the horizontal
    # mouth landmarks (x, y)-coordinates
    D = np.linalg.norm(mouth[12] - mouth[16])

    # Compute the mouth aspect ratio
    mar = (A + B + C) / (2 * D)

    # Return the mouth aspect ratio
    return mar


def sound_alarm(path):
        # play an alarm sound
        playsound.playsound(path)
        
def eye_aspect_ratio(eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        return ear
 



def drowsiness():
    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold for to set off the
    # alarm
    EYE_AR_THRESH = 0.20
    EYE_AR_CONSEC_FRAMES = 6
    MOUTH_AR_THRESH = 0.2
    MOUTH_AR_CONSECUTIVE_FRAMES = 5
    YAWN_COUNT = 0
    # initialize the frame counter as well as a boolean used to
    # indicate if the alarm is going off
    cnt = 0
    COUNTER = 0
    ALARM_ON = False
    MOUTH_COUNTER = 0
    YELLOW_COLOR = (0, 255, 255)
    frame_count = 0
    start_time = time.time()
    

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    # start the video stream thread
    print("[INFO] Starting Video")
    vs = VideoStream(0).start()
    time.sleep(1.0)
    while True:
            # grab the frame from the threaded video file stream, resize
            # it, and convert it to grayscale
            # channels)
            frame = vs.read()
            size = frame.shape
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_count += 1

    # calculate the time elapsed since the start of the loop
            elapsed_time = time.time() - start_time

    # calculate the frame rate
            frame_rate = frame_count / elapsed_time

            # detect faces in the grayscale frame
            rects = detector(gray, 0)
            if(len(rects) < 1):
                cv2.putText(frame, "Alert! Look at Camera", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                engine.say("Look at Camera")
                engine.runAndWait()
            else:
                cv2.putText(frame, "", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # loop over the face detections
            for rect in rects:
                    # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    # extract the left and right eye coordinates, then use the
                    # coordinates to compute the eye aspect ratio for both eyes
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    mouth = shape[mStart:mEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)
                    mar = mouth_aspect_ratio(mouth)
                    

                    # average the eye aspect ratio together for both eyes
                    ear = (leftEAR + rightEAR) / 2.0

                    # compute the convex hull for the left and right eye, then
                    # visualize each of the eyes and mouth
                    mouthHull = cv2.convexHull(mouth)
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [mouthHull], -1, YELLOW_COLOR, 1)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                    # check to see if the eye and mouth aspect ratio is below the 
                    # threshold, and if so, increment the blink frame counter
                    if mar > MOUTH_AR_THRESH:
                            MOUTH_COUNTER += 1
                            #print(MOUTH_COUNTER)
                            if MOUTH_COUNTER >= MOUTH_AR_CONSECUTIVE_FRAMES:
                                    YAWN_COUNT += 1
                                    MOUTH_COUNTER = 0
                                    if YAWN_COUNT > 5:
                                        engine.say("You seem tired!")
                                        engine.runAndWait()
                                        #audioalert()
                                        bot = telegram.Bot(token='6171813184:AAFNowB4Bj0HYXfIWsitr8GXUFvuQLitrhw')
                                        chat_id = '857465702'
                                        async def send_telegram_alert():
                                          
                                          

# Get the driver's location using MapQuest Geocoding API
                                             g = geocoder.ip('me')
                                             location = g.latlng
                                             lat = location[0]
                                             lon = location[1]
    
    # Reverse geocode the driver's location using MapQuest Geocoding API
                                             url = f'http://www.mapquestapi.com/geocoding/v1/reverse?key=TDpjUUr2daJWq216f1umRzxThcUyGseo&location={lat},{lon}'
                                             response = requests.get(url).json()
                                             address = response['results'][0]['locations'][0]['street'] + ', ' + response['results'][0]['locations'][0]['adminArea5'] + ', ' + response['results'][0]['locations'][0]['adminArea3']
    
    # Create the message to send via Telegram
                                             location_url = f'https://www.google.com/maps/search/?api=1&query={lat},{lon}'
                                             message_text = "Driver is feeling drowsy. Please take necessary actions."
                                             message_text += f"\n\nDriver's current location: {address}."
                                             message_text += f"\n\nGoogle Maps location URL: {location_url}"
    
    # Send the message via Telegram
                                             await bot.send_message(chat_id=chat_id, text=message_text)


                                        asyncio.run(send_telegram_alert())
                              
                    else:
                            MOUTH_COUNTER = 0
                            
                    if ear < EYE_AR_THRESH:
                            COUNTER += 1

                            # if the eyes were closed for a sufficient number of frames
                            # then sound the alarm
                            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                engine.say("stay alert...!!")
                                engine.runAndWait()
                                bot = telegram.Bot(token='6171813184:AAFNowB4Bj0HYXfIWsitr8GXUFvuQLitrhw')
                                chat_id = '857465702'
                                async def send_telegram_alert():
                                             

# Get the driver's location using MapQuest Geocoding API
# Get the driver's location using MapQuest Geocoding API
                                             g = geocoder.ip('me')
                                             location = g.latlng
                                             lat = location[0]
                                             lon = location[1]
    
    # Reverse geocode the driver's location using MapQuest Geocoding API
                                             url = f'http://www.mapquestapi.com/geocoding/v1/reverse?key=TDpjUUr2daJWq216f1umRzxThcUyGseo&location={lat},{lon}'
                                             response = requests.get(url).json()
                                             address = response['results'][0]['locations'][0]['street'] + ', ' + response['results'][0]['locations'][0]['adminArea5'] + ', ' + response['results'][0]['locations'][0]['adminArea3']
    
    # Create the message to send via Telegram
                                             location_url = f'https://www.google.com/maps/search/?api=1&query={lat},{lon}'
                                             message_text = "Driver is feeling drowsy. Please take necessary actions."
                                             message_text += f"\n\nDriver's current location: {address}."
                                             message_text += f"\n\nGoogle Maps location URL: {location_url}"
    
    # Send the message via Telegram
                                             await bot.send_message(chat_id=chat_id, text=message_text)

                                asyncio.run(send_telegram_alert())
                                        #asyncio.run(send_telegram_alert("Driver is feeling drowsy. Please take necessary actions."))
# Replace the message text with your own message
                    # threshold, so reset the counter and alarm
                    else:
                            COUNTER = 0
                            #ALARM_ON = False
                    p1, p2, angle = getheadpose(frame, shape, frame.shape[:2])
                    #print("pitch" + str(p1[0]) + "yaw" +  str(p2[0]))
                    pitch = p1[0]        
                    
                    '''if abs(angle) < 25 or abs(angle) > 141:
                          
#if abs(angle) < 25:
                          cv2.putText(frame, "Head tilted!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                          cv2.putText(frame, "", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    '''
                    if pitch >150 and pitch<210:

                        cnt += 1
                        if cnt > 15:
                            engine.say("Alert...!!")
                            engine.runAndWait()
                            #audioalert()
                              
                        cv2.putText(frame, "looking right".format(ear), (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        engine.say("looking right!")
                        engine.runAndWait()
                    elif pitch >270 and pitch <290:

                        cv2.putText(frame, "looking left".format(ear), (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        engine.say("looking left!")
                        engine.runAndWait()
                        
                    # draw the computed eye aspect ratio on the frame to help
                    # with debugging and setting the correct eye aspect ratio
                    # thresholds and frame counters
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "YAWN COUNT:{}".format(YAWN_COUNT), (270, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    #cv2.putText(frame, 'Duration: {}'.format(duration(start_time)), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    #cv2.putText(frame, f"Frame rate: {frame_rate:.2f}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, "Head Tilt: {:.2f}".format(p1[0]), (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2),cv2.line(frame, p1, p2, (0, 255, 0), 2)



            # show the frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                    break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    print("[INFO] Cleaning all")
    print("[INFO] Closed")    

def on_closing():
    from tkinter import messagebox
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()
window.protocol("WM_DELETE_WINDOW", on_closing)      


button3 = tk.Button(window, text=" Detector",command=drowsiness,fg="white"  ,bg="black"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 20, ' italic bold '))
button3.place(x=1020, y=350)

size = (100, 100)
color = 'black'
img = Image.new('RGB', size, color)
draw = ImageDraw.Draw(img)
draw.ellipse((0, 0, size[0], size[1]), fill='red')
photo = ImageTk.PhotoImage(img)
quitWindow = Button(window, text="Quit", fg="white", bg="black", bd=0, highlightthickness=0, command=on_closing, font=('times', 15, 'bold'), image=photo, compound="center")
quitWindow.place(x=1390, y=650)


window.mainloop()
print("[INFO] Closing ALL")
print("[INFO] Closed")