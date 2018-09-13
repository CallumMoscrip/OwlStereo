#!/usr/bin/env python
# Python 2/3 compatibility
from __future__ import print_function
#---------------------
import socket
import cv2
import StringIO
import time
import numpy as np
import math
from matplotlib import pyplot as plt
from common import clock, draw_str, StatValue
from video import create_capture
from random import randint
from multiprocessing.pool import ThreadPool
from collections import deque

TCP_IP = '10.0.0.10'
TCP_PORT = 12345
BUFFER_SIZE = 24
#s.close()

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
 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))
s.send('{} {} {} {} {}'.format(RxC, RyC, LxC, LyC, NeckC))

#camera matrix
rightCameraMatrix=np.matrix('571.60331566 0. 313.63057696; 0. 570.47045797 235.60499504; 0. 0. 1.')

rightDistortionCoefficients=np.matrix('1.07918650e-01 -5.70919923e-01 3.44280019e-03 -5.47543586e-05 6.26097694e-01')

leftCameraMatrix=np.matrix('573.25102289 0. 311.17096872; 0. 572.37191522 241.59517615; 0. 0. 1.')

leftDistortionCoefficients=np.matrix('1.15207434e-01 -7.18058054e-01 1.91498382e-03 -5.18108238e-04 9.36403908e-01')    

image_scale = 2

midScreenX = (640/2)
midScreenY = (480/2)
midScreenWindow = 20  # acceptable 'error' for the center of the screen.
midfaceL = None
midfaceR = None
font = cv2.FONT_HERSHEY_SIMPLEX



def face_search():
    global Neck
    Neck = randint(1110, 1940)

def range_calc(Rx,Lx):
    theta1 = math.degrees(90 - ((Lx-LxC)*0.113))
    theta2 = math.degrees(90 - ((Rx-RxC)*0.113))
    h1 = (math.sin(theta1)*math.sin(theta1)/math.sin(theta1+theta2))*65
    h2 = (math.sin(theta2)*math.sin(theta2)/math.sin(theta1+theta2))*65
    dist = ((h1+h2)/2)
    cv2.putText(unDistR, str(dist), (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(unDistL, str(dist), (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
def eye_track(imgL, imgR, rectsL, rectsR):
    #calculates direction to move from location of the rectangle from haarcascade rectangle
    global Rx, Ry, Lx, Ly, Neck, midfaceL, midFaceXL, midFaceYL, midfaceR, midFaceXR, midFaceYR
    for x1, y1, x2, y2 in rectsR:
        cv2.putText(unDistR, str(rects), (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        midFaceXR = x1+((x2-x1)/2)
        midFaceYR = y1+((y2-y1)/2)
        midfaceR = [midFaceXR, midFaceYR]
        
    for x1, y1, x2, y2 in rectsL:
        cv2.putText(unDistL, str(rects), (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        midFaceXL = x1+((x2-x1)/2)
        midFaceYL = y1+((y2-y1)/2)
        midfaceL = [midFaceXL, midFaceYL]
                
        if np.any(midfaceR != None):
            if(midFaceXR < (midScreenX - midScreenWindow)):  #is the face on the left of the mid line?
                Rx += 4
                #time.sleep(0.01)

            elif(midFaceXR > (midScreenX + midScreenWindow)): #is the face on the right of the mid line?
                Rx -= 4
                time.sleep(0.01)
                        
            if(midFaceYR < (midScreenY - midScreenWindow/2)): #is the face above the mid line?
                Ry += 4
                #time.sleep(0.01)
                        
            elif(midFaceYR > (midScreenY + midScreenWindow/2)): #is the face below the midline?
                Ry -= 4
                #time.sleep(0.01)
        
        if np.any(midfaceL != None):
            if(midFaceXL < (midScreenX - midScreenWindow)):  #is the face on the left of the mid line?
                Lx += 4
                #time.sleep(0.01)

            elif(midFaceXL > (midScreenX + midScreenWindow)): #is the face on the right of the mid line?
                Lx -= 4
                #time.sleep(0.01)
                        
            if(midFaceYL < (midScreenY - midScreenWindow/2)): #is the face above the mid line?
                Ly -= 4
                #time.sleep(0.01)
                        
            elif(midFaceYL > (midScreenY + midScreenWindow/2)): #is the face below the midline?
                Ly += 4
                #time.sleep(0.01)
                    
    if Rx - RxC & Lx - LxC < 0:
        Neck += 3
    elif Rx - RxC & Lx - LxC > 0:
        Neck -= 3
    else:
        #range_calc(Rx, Lx)
        pass

    if np.any(midfaceL == None):
        #face_search()
        pass
          
            
def image_process():
    global unDistLG, unDistRG, unDistL, unDistR
    ret, Frame = cap.read()
    imgR = rightFrame = Frame[0:480,  0:640]    #right eye
    imgL = leftFrame = Frame[0:480,  640:1280]  #left eye
        
    #Get new camera matrix
    hR,  wR = imgR.shape[:2]
    newcameramtxR, roiR=cv2.getOptimalNewCameraMatrix(rightCameraMatrix,rightDistortionCoefficients,(wR,hR),0,(wR,hR))
    hL,  wL = imgL.shape[:2]
    newcameramtxL, roiL=cv2.getOptimalNewCameraMatrix(leftCameraMatrix,leftDistortionCoefficients,(wL,hL),0,(wL,hL))

    unDistL = cv2.remap(imgL, undistort_map1, rectify_map1, cv2.INTER_LINEAR)
    unDistR = cv2.remap(imgR, undistort_map2, rectify_map2, cv2.INTER_LINEAR)

    # Test to reduce load---------------------------------------------------
    #unDistRG = imgR
    #unDistLG = imgL
    # Test to reduce load---------------------------------------------------

    #convert to greyscale
    unDistRG=cv2.cvtColor(unDistR, cv2.COLOR_BGR2GRAY)
    unDistLG=cv2.cvtColor(unDistL, cv2.COLOR_BGR2GRAY)

    unDistRG = cv2.equalizeHist(unDistRG)
    unDistLG = cv2.equalizeHist(unDistLG)
    return unDistRG
    return unDistLG


def create_window():
    global cap, time
    #cap = cv2.VideoCapture('C:\Users\callum\OneDrive - University of Plymouth\AINT513\CMcurrent\Owl.mp4')
    cap = cv2.VideoCapture('http://10.0.0.10:8080/stream/video.mjpeg')
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

def detect(img, cascade):
    global rects
    rects = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=4, minSize=(20, 20), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    global x1, x2, y1, y2
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (318, 238), (322, 242), color, 4)
        
def find_faces():
    global rectR, rectL
    timer = cv2.getTickCount()
    rectR = detect(unDistRG, faces)
    rectL = detect(unDistLG, faces)
    #visR = unDistR.copy()
    #visL = unDistL.copy()
    #rectR = rectsR
    #rectL = rectsL
    
    draw_rects(unDistR, rectR, (0, 255, 0))
    draw_rects(unDistL, rectL, (0, 255, 0))

    eye_track(unDistL, unDistR, rectL, rectR)
    
    '''if not eyes.empty():
        for x1R, y1R, x2R, y2R in rectsR:
            roiR = unDistRG[y1R:y2R, x1R:x2R]
            vis_roiR = visR[y1R:y2R, x1R:x2R]
            subrectsR = detect(roiR.copy(), eyes)
            draw_rects(vis_roiR, subrectsR, (255, 0, 0))
               
    if not eyes.empty():
        for x1L, y1L, x2L, y2L in rectsL:
            roiL = unDistLG[y1L:y2L, x1L:x2L]
            vis_roiL = visL[y1L:y2L, x1L:x2L]
            subrectsL = detect(roiL.copy(), eyes)
            draw_rects(vis_roiL, subrectsL, (255, 0, 0))'''
    
    s.send('{} {} {} {} {}'.format(Rx, Ry, Lx, Ly, Neck))
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    # Display tracker type on frame
    cv2.putText(unDistR, "FPS : " + str(int(fps)), (100,50), font, 0.75, (50,170,50), 2);


if __name__ == '__main__':
    
    import sys, getopt
    create_window()

    args, video_src = getopt.getopt(sys.argv[1:], '', ['faces=', 'eyes-faces='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    faces_fn = args.get('--cascade', "C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml")
    eyes_fn  = args.get('--nested-cascade', "C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml")

    faces = cv2.CascadeClassifier(faces_fn)
    eyes = cv2.CascadeClassifier(eyes_fn)
    while True:
        image_process()
        find_faces()       

        #draw_str(visR, (20, 20), 'time: %.1f ms' % (dt*1000))
        #draw_str(visL, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('facedetect right', unDistR)
        cv2.imshow('facedetect left', unDistL)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    
    cv2.destroyAllWindows()
