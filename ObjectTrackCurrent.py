#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 2/3 compatibility

#Modified facetrack.py to work for selected object in one window

from __future__ import print_function
#---------------------
import socket
import cv2
import StringIO
import time
import numpy as np
import math
import sys
from matplotlib import pyplot as plt
#from common import clock, draw_str
from random import randint

TCP_IP = '10.0.0.10'
TCP_PORT = 12345
BUFFER_SIZE = 24

RxC=1530 
RyC=1563
LxC=1425
LyC=1440 
NeckC=1530

Rx = RxC
Ry = RyC
Lx = LxC
Ly = LyC
Neck = NeckC
count = 0
 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))
s.send('{} {} {} {} {}'.format(RxC, RyC, LxC, LyC, NeckC))

#camera matrix
rightCameraMatrix=np.matrix('571.60331566 0. 313.63057696; 0. 570.47045797 235.60499504; 0. 0. 1.')

rightDistortionCoefficients=np.matrix('1.07918650e-01 -5.70919923e-01 3.44280019e-03 -5.47543586e-05 6.26097694e-01')

leftCameraMatrix=np.matrix('573.25102289 0. 311.17096872; 0. 572.37191522 241.59517615; 0. 0. 1.')

leftDistortionCoefficients=np.matrix('1.15207434e-01 -7.18058054e-01 1.91498382e-03 -5.18108238e-04 9.36403908e-01')    

midScreenX = (320)
midScreenY = (240)
midScreenWindow = 20  # acceptable 'error' for the center of the screen.

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

font = cv2.FONT_HERSHEY_SIMPLEX

def create_window():
    global cap, time, undistort_map1, rectify_map1, undistort_map2, rectify_map2
    #cap = cv2.VideoCapture('/home/callum/opencv/ProjOwl/Owl.mp4')
    cap = cv2.VideoCapture('http://10.0.0.10:8080/stream/video.mjpeg')
    ret, Frame = cap.read()
    imgR = rightFrame = Frame[0:480,  0:640]    #right eye
    imgL = leftFrame = Frame[0:480,  640:1280]  #left eye

    #Get new camera matrix
    #cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha[, newImgSize[,
    #                               centerPrincipalPoint]]) → retval, validPixROI
    hR,  wR = imgR.shape[:2]
    newcameramtxR, roiR=cv2.getOptimalNewCameraMatrix(rightCameraMatrix,rightDistortionCoefficients,(wR,hR),0,(wR,hR))
    hL,  wL = imgL.shape[:2]
    newcameramtxL, roiL=cv2.getOptimalNewCameraMatrix(leftCameraMatrix,leftDistortionCoefficients,(wL,hL),0,(wL,hL))
    
    #cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type[, map1[, map2]]) → map1, map2
    undistort_map1, rectify_map1 = cv2.initUndistortRectifyMap(cameraMatrix = leftCameraMatrix, distCoeffs = leftDistortionCoefficients,
                                                               R = None, newCameraMatrix = newcameramtxL, size = (wL,hL), m1type = cv2.CV_16SC2)
    undistort_map2, rectify_map2 = cv2.initUndistortRectifyMap(cameraMatrix = rightCameraMatrix, distCoeffs = rightDistortionCoefficients,
                                                               R = None, newCameraMatrix = newcameramtxR, size = (wR,hR), m1type = cv2.CV_16SC2)
    
    for i in range(0, 10):  #required to allow enough light into the camer as the first frame was to dark
        ret, Frame = cap.read()
        imgR = rightFrame = Frame[0:480,  0:640]    #right eye
        imgL = leftFrame = Frame[0:480,  640:1280]  #left eye

        time.sleep(0.1)
        unDistL = cv2.remap(imgL, undistort_map1, rectify_map1, cv2.INTER_LINEAR)
        unDistR = cv2.remap(imgR, undistort_map2, rectify_map2, cv2.INTER_LINEAR)
        
    cv2.destroyAllWindows()
    track(unDistL, 'L') #select tracking type and select object for tracking


def track(frame, side):
    global unDistR, unDistL, trackerR, trackerL, tracker_type, bboxL, bboxR
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW'] # need to decide on best tracker to use
    tracker_type = tracker_types[2]
    bboxL = np.matrix ('0 0 0 0')
    bboxR = np.matrix ('0 0 0 0')
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            trackerR = cv2.TrackerBoosting_create()
            trackerL = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            trackerR = cv2.TrackerMIL_create()
            trackerL = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            trackerR = cv2.TrackerKCF_create()
            trackerL = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            trackerR = cv2.TrackerTLD_create()
            trackerL = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            trackerR = cv2.TrackerMedianFlow_create()
            trackerL = cv2.TrackerMedianFlow_create()
 
    # select object to track
    if side == 'L':
        bbox = cv2.selectROI(frame, False)
        bboxL = bbox
        print ('Left bounding box', bboxL)
    elif side == 'R':
        bbox = cv2.selectROI(frame, False)
        bboxR = bbox
        print ('Right bounding box', bboxR)
        
    # start the tracker with selected object
    okR = trackerR.init(frame, bbox)
    okL = trackerL.init(frame, bbox)
    
    cv2.destroyAllWindows()
    
def image_process():
    global unDistLG, unDistRG, unDistL, unDistR, imgR, imgL
    ret, Frame = cap.read()
    imgR = rightFrame = Frame[0:480,  0:640]    #right eye
    imgL = leftFrame = Frame[0:480,  640:1280]  #left eye
    
    #cv2.remap(src, map1, map2, interpolation[, dst[, borderMode[, borderValue]]]) → dst
    unDistL = cv2.remap(imgL, undistort_map1, rectify_map1, cv2.INTER_LINEAR)
    unDistR = cv2.remap(imgR, undistort_map2, rectify_map2, cv2.INTER_LINEAR)

    object_newframe(unDistL, 'L')
    object_newframe(unDistR, 'R')

    eye_track(unDistL, unDistR, bboxL, bboxR)
    return unDistL, unDistR

def object_newframe(Frame, side):
    global bboxL, bboxR

    # Start timer
    timer = cv2.getTickCount()
    # Update tracker
    if side == 'L':
        ok, bbox = trackerL.update(Frame)
        bboxL = bbox
        if ok:  # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(Frame, p1, p2, (255,0,0), 2, 1)
        
        else :
            # Tracking failure
            cv2.putText(Frame, "Tracking failure detected", (100,80), font, 0.75,(0,0,255),2)
    elif side == 'R':
        ok, bbox = trackerR.update(Frame)
        bboxR = bbox
        if ok:  # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(Frame, p1, p2, (255,0,0), 2, 1)
        
        else :
            # Tracking failure
            cv2.putText(Frame, "Tracking failure detected", (100,80), font, 0.75,(0,0,255),2)
        
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
    # Display tracker type on frame
    cv2.putText(Frame, tracker_type + " Tracker", (100,20), font, 0.75, (50,170,50),2);
     
    # Display FPS on frame
    cv2.putText(Frame, "FPS : " + str(int(fps)), (100,50), font, 0.75, (50,170,50), 2);
    return bboxL, bboxR
    
def eye_track(imgL, imgR, rectsL, rectsR):
    global Rx, Ry, Lx, Ly, Neck, count
    #calculates direction to move from location of the rectangle from tracker
    x1R = rectsR[0]
    y1R = rectsR[1]
    x2R = rectsR[2]
    y2R = rectsR[3]
    midTargetXR = x1R+(x2R/2)
    midTargetYR = y1R+(y2R/2)
    midTargetR = [midTargetXR, midTargetYR]
    print('midtarget right', midTargetR)
    hR,  wR = imgR.shape[:2]
    print(hR, wR)

    x1L = rectsL[0]
    y1L = rectsL[1]
    x2L = rectsL[2]
    y2L = rectsL[3]
    midTargetXL = x1L+(x2L/2)
    midTargetYL = y1L+(y2L/2)
    midTargetL = [midTargetXL, midTargetYL]
    print('midtarget left', midTargetL)
    hL,  wL = imgL.shape[:2]
    print(hL, wL)
                
    if midTargetR == [0.0, 0.0]: #if no target found use value from other eye to try and find target
        Rx = Lx
        Ry = Ly
    else:
        if(midTargetXR > (midScreenX + midScreenWindow)):  #is the face on the left of the mid line?
            Rx -= 10

        elif(midTargetXR < (midScreenX - midScreenWindow)): #is the face on the right of the mid line?
            Rx += 10
                        
        if(midTargetYR > (midScreenY + midScreenWindow)): #is the face above the mid line?
            Ry -= 10
                        
        elif(midTargetYR < (midScreenY - midScreenWindow)): #is the face below the midline?
            Ry += 10

    if midTargetL == [0.0, 0.0]: #if no target found use value from other eye to try and find target
        Lx = Rx
        Ly = Ry
    else:
        if(midTargetXL < (midScreenX - midScreenWindow)):  #is the face on the left of the mid line?
            Lx += 10

        elif(midTargetXL > (midScreenX + midScreenWindow)): #is the face on the right of the mid line?
            Lx -= 10
                        
        if(midTargetYL > (midScreenY + midScreenWindow)): #is the face above the mid line?
            Ly += 10
                        
        elif(midTargetYL < (midScreenY - midScreenWindow)): #is the face below the midline?
            Ly -= 10

    if midTargetL == [0.0, 0.0]:
        if midTargetR == [0.0, 0.0]:
            pass
    else:
        if count == 3:
            count = 0
            if Rx - RxC & Lx + LxC < 0: 
                #Neck += ((Rx - RxC)+(Lx - LxC))/2
                Neck += 3
            elif Rx - RxC & Lx + LxC > 0:
                #Neck -= ((Rx - RxC)+(Lx - LxC))/2
                Neck += 3
            else:
                #range_calc(Rx, Lx)  # if the owl is faceing directly at the object attempt to calc range
                pass
    s.send('{} {} {} {} {}'.format(Rx, Ry, Lx, Ly, Neck))
    count += 1


def face_search(): #crude method of searching for target
    global Neck
    Neck = randint(1110, 1940)

def range_calc(Rx,Lx): # attempts to calculate the range using trig and angle of eyes not very good.....
    theta1 = math.degrees(90 - ((Lx-LxC)*0.113))
    theta2 = math.degrees(90 - ((Rx-RxC)*0.113))
    h1 = (math.sin(theta1)*math.sin(theta1)/math.sin(theta1+theta2))*65
    h2 = (math.sin(theta2)*math.sin(theta2)/math.sin(theta1+theta2))*65
    dist = ((h1+h2)/2)
    cv2.putText(visR, str(dist), (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(visL, str(dist), (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    

if __name__ == '__main__':
    
    import sys, getopt
    create_window()

    while True:
        image_process()
        cv2.imshow("Tracking L", unDistL) # show tracking frame
        cv2.imshow("Tracking R", unDistR) # show tracking frame


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    s.close()

    cv2.destroyAllWindows()
