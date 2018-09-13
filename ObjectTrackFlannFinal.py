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
midScreenWindow = 40  # acceptable 'error' for the center of the screen.

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

font = cv2.FONT_HERSHEY_SIMPLEX

def create_window():
    global cap, time, undistort_map1, rectify_map1, undistort_map2, rectify_map2
    cap = cv2.VideoCapture('http://10.0.0.10:8080/stream/video.mjpeg') #define video for capture
    ret, Frame = cap.read()                     #read a frane from video
    imgR = rightFrame = Frame[0:480,  0:640]    #right eye
    imgL = leftFrame = Frame[0:480,  640:1280]  #left eye

    #Get new camera matrix
    hR,  wR = imgR.shape[:2]
    newcameramtxR, roiR=cv2.getOptimalNewCameraMatrix(rightCameraMatrix,rightDistortionCoefficients,(wR,hR),0,(wR,hR))
    hL,  wL = imgL.shape[:2]
    newcameramtxL, roiL=cv2.getOptimalNewCameraMatrix(leftCameraMatrix,leftDistortionCoefficients,(wL,hL),0,(wL,hL))
    
    #cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type[, map1[, map2]])
    undistort_map1, rectify_map1 = cv2.initUndistortRectifyMap(cameraMatrix = leftCameraMatrix, distCoeffs = leftDistortionCoefficients,
                                                               R = None, newCameraMatrix = newcameramtxL, size = (wL,hL), m1type = cv2.CV_16SC2)
    undistort_map2, rectify_map2 = cv2.initUndistortRectifyMap(cameraMatrix = rightCameraMatrix, distCoeffs = rightDistortionCoefficients,
                                                               R = None, newCameraMatrix = newcameramtxR, size = (wR,hR), m1type = cv2.CV_16SC2)
    
    for i in range(0, 10):  #spin up camera, required to allow enough light into the camera as the first frame was to dark
        ret, Frame = cap.read()
        time.sleep(0.1)

        imgR = rightFrame = Frame[0:480,  0:640]    #right eye
        imgL = leftFrame = Frame[0:480,  640:1280]  #left eye

        time.sleep(0.1)
        unDistL = cv2.remap(imgL, undistort_map1, rectify_map1, cv2.INTER_LINEAR) #remap pixels using maps from undistortrectify
        unDistR = cv2.remap(imgR, undistort_map2, rectify_map2, cv2.INTER_LINEAR) #function to undistort the image
     
    Obj_Select()

def Obj_Select():
    global img1, surf, kp1, des1, flann, pts
    
    while True:
        ret, Frame = cap.read()
        time.sleep(0.1)

        imgR = rightFrame = Frame[0:480,  0:640]    #right eye
        imgL = leftFrame = Frame[0:480,  640:1280]  #left eye
        unDistR = cv2.remap(imgR, undistort_map2, rectify_map2, cv2.INTER_LINEAR)
        unDistL = cv2.remap(imgL, undistort_map1, rectify_map1, cv2.INTER_LINEAR)

        cv2.putText(unDistL, "Select the object for tracking.", (10,20), font, 0.75,(0,0,255),2)
        cv2.putText(unDistL, "Press space when ready.", (10,50), font, 0.75,(0,0,255),2)
        cv2.imshow("target", unDistL)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.destroyAllWindows()
            break

    roi = cv2.selectROI(unDistL, False) #select the region of interest to get keypoints from
    x,y,w,h = roi
    unDistL = unDistL[y:y+h, x:x+w] #crop the original image so only ROI is shown
    
    cv2.destroyAllWindows()
    '''#-------------------------------------------------------------------------------------------
    unDistL = cv2.imread('object1.jpg',1) #yorkshire tea box for testing
    #-------------------------------------------------------------------------------------------'''
    img1 = cv2.cvtColor(unDistL, cv2.COLOR_BGR2GRAY) #convert to grey scale
    

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)  #Required for SURF and SIFT matching
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    surf = cv2.xfeatures2d.SURF_create(500)
    surf.setUpright(True)
    
    kp1, des1 = surf.detectAndCompute(img1, None)
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    
    cv2.destroyAllWindows()   
 
def matcher(Frame, side):
    global kp1, kp2L, kp2R, des1, des2L, des2R, midTargetXL, midTargetYL, midTargetXR, midTargetYR
    MIN_MATCH_COUNT = 10
    if side == 'L':
        img2L = Frame 
        kp2L, des2L = surf.detectAndCompute(img2L,None)
        matchesL = flann.knnMatch(des1,des2L,k=2)
        # store all the good matches as per Lowe's ratio test.
        goodL = []
        for m,n in matchesL:
            if m.distance < 0.7*n.distance:
                goodL.append(m)
                
        #------------------------------------------------------------------------
        #need to search all good matches and get cooridinates. find Center coordinate to feed to eyetracking
        # Initialize lists
        list_kp1 = []
        list_kp2L = []

        # For each match...
        for mat in goodL:

            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2L_idx = mat.trainIdx

            # x - columns
            # y - rows
            # Get the coordinates
            (x1,y1) = kp1[img1_idx].pt
            (x2,y2) = kp2L[img2L_idx].pt

            # Append to each list
            list_kp1.append((x1, y1))
            list_kp2L.append((x2, y2))
        sortedP = np.sort(list_kp2L, axis=0)
        srtX =[]
        srtY = []
        for x,y in sortedP:
            srtX.append(x)
            srtY.append(y)
        midTargetXL = np.mean(srtX) #the X coord center of the matched keypoints
        midTargetYL = np.mean(srtY) #the Y coord center of the matched keypoints 
        #----------------------------------------------------------------------
        if len(goodL)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in goodL ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2L[m.trainIdx].pt for m in goodL ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            #h,w = img1.shape
            #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            img2 = cv2.polylines(Frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        else:
            print('Not enough matches are found on left - %d/%d' % (len(goodL),MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(matchColor = (0,255,0),
                         singlePointColor = (255,0,0),
                         matchesMask = matchesMask,
                         flags = 0)

        #img3L = cv2.drawMatchesKnn(img1,kp1,Frame,kp2L,goodL,None,**draw_params) #knn good matches
        img3L = cv2.drawMatches(img1,kp1,Frame,kp2L,goodL, outImg=Frame, flags=0)
        
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
         
        # Display FPS on frame
        cv2.putText(Frame, "FPS : " + str(int(fps)), (10,10), font, 0.75, (50,170,50), 2);
        cv2.imshow("target"+side, Frame)
        #cv2.imshow("key "+side, img3L) #show keypoint matches


    elif side == 'R':
        img2R = Frame 
        kp2R, des2R = surf.detectAndCompute(img2R,None)
        matchesR = flann.knnMatch(des1,des2R,k=2)
        # store all the good matches as per Lowe's ratio test.
        goodR = []
        for m,n in matchesR:
            if m.distance < 0.7*n.distance:
                goodR.append(m)
        #------------------------------------------------------------------------
        #need to search all good matches and get cooridinates. find Center coordinate to feed to eyetracking
        # Initialize lists
        list_kp1 = []
        list_kp2R = []

        # For each match...
        for mat in goodR:

            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2R_idx = mat.trainIdx

            # x - columns
            # y - rows
            # Get the coordinates
            (x1,y1) = kp1[img1_idx].pt
            (x2,y2) = kp2R[img2R_idx].pt

            # Append to each list
            list_kp1.append((x1, y1))
            list_kp2R.append((x2, y2))
        sortedP = np.sort(list_kp2R, axis=0)
        srtX =[]
        srtY = []
        for x,y in sortedP:
            srtX.append(x)
            srtY.append(y)
        midTargetXR = np.mean(srtX) #the X coord center of the matched keypoints
        midTargetYR = np.mean(srtY) #the Y coord center of the matched keypoints

        #----------------------------------------------------------------------

        if len(goodR)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in goodR ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2R[m.trainIdx].pt for m in goodR ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            #h,w = img1.shape
            #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            img2 = cv2.polylines(Frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        else:
            print('Not enough matches are found on right - %d/%d' % (len(goodR),MIN_MATCH_COUNT))
            matchesMask = None

        draw_params = dict(matchColor = (0,255,0),
                         singlePointColor = (255,0,0),
                         matchesMask = matchesMask,
                         flags = 0)

        #img3R = cv2.drawMatchesKnn(img1,kp1,Frame,kp2R,goodR,None,**draw_params)
        img3R = cv2.drawMatches(img1,kp1,Frame,kp2R,goodR, outImg=Frame, flags=0)
        
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
         
        # Display FPS on frame
        cv2.putText(Frame, "FPS : " + str(int(fps)), (20,20), font, 0.75, (50,170,50), 2);        
        cv2.imshow("target"+side, Frame)
        #cv2.imshow("key "+side, img3R) #show key point matches


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
    global unDistLG, unDistRG, unDistL, unDistR, imgR, imgL, timer
    # Start timer
    timer = cv2.getTickCount()
    ret, Frame = cap.read()
    imgR = rightFrame = Frame[0:480,  0:640]    #right eye
    imgL = leftFrame = Frame[0:480,  640:1280]  #left eye
    
    #cv2.remap(src, map1, map2, interpolation[, dst[, borderMode[, borderValue]]])
    unDistL = cv2.remap(imgL, undistort_map1, rectify_map1, cv2.INTER_LINEAR)
    unDistR = cv2.remap(imgR, undistort_map2, rectify_map2, cv2.INTER_LINEAR)

    unDistRG=cv2.cvtColor(unDistR, cv2.COLOR_BGR2GRAY)
    unDistLG=cv2.cvtColor(unDistL, cv2.COLOR_BGR2GRAY)

    matcher(unDistLG, 'L')
    matcher(unDistRG, 'R')
    eye_track()
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
    
def eye_track():
    global Rx, Ry, Lx, Ly, Neck, count
    #calculates direction to move from the midTarget which is the center of the distribution of matches
    
    if midTargetXR < (midScreenX + midScreenWindow):  #is the face on the left of the mid line?
        Rx += 15

    elif midTargetXR > (midScreenX - midScreenWindow): #is the face on the right of the mid line?
        Rx -= 10
                    
    if midTargetYR > (midScreenY + midScreenWindow): #is the face above the mid line?
        Ry -= 10
                    
    elif midTargetYR < (midScreenY - midScreenWindow): #is the face below the midline?
        Ry += 10


    if midTargetXL > (midScreenX - midScreenWindow):  #is the face on the left of the mid line?
        Lx -= 15

    elif midTargetXL < (midScreenX + midScreenWindow): #is the face on the right of the mid line?
        Lx += 10
                    
    if midTargetYL > (midScreenY + midScreenWindow): #is the face above the mid line?
        Ly += 10
                    
    elif midTargetYL < (midScreenY - midScreenWindow): #is the face below the midline?
        Ly -= 10

    if count == 2: #only move the neck after set cycles of eye movement
        count = 0
        if Rx - RxC < 0: 
            Neck += 10
        elif Rx - RxC > 0:
            Neck -= 10
        else:
            pass
    s.send('{} {} {} {} {}'.format(Rx, Ry, Lx, Ly, Neck))
    count += 1

if __name__ == '__main__':
    
    import sys, getopt
    create_window()

    while True:
        image_process()
                              
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord(' '):  #if space pressed reselect the object to track
            cv2.destroyAllWindows()
            Obj_Select()

    cap.release()
    cv2.destroyAllWindows()
    s.close()

