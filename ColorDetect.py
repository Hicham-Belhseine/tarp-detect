#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File name: "ColorDetect.py"
Date created: 11/24/2017
Date last modified: 1/12/2018
Python Version: 3.6

A simple script for detecting and tracking tarps/squares based on there on
their color.
"""

__author__ = "Hicham Belhseine"
__email__ = "hichambelhseine@gmail.com"

import cv2
import imutils
from imutils.video import VideoStream
import numpy as np
import matplotlib.pyplot as plt;

plt.rcdefaults()
import matplotlib.pyplot as plt
import threading


class ThreadImage:
    # Class for creating a parallel process that gets an image while
    # another image is being processed. Speeds up program ~3x if previous
    # program used cv2.VideoCapture, or ~1.5x if program used imutils VideoStream

    def __init__(self, sourceCam=0):
        # init that initializes camera and gets first image for initial use
        self.stream = cv2.VideoCapture(sourceCam)

        grabbed, self.frame = self.stream.read()

        self.stopped = False

    def startThread(self):
        # start the thread
        threading.Thread(target=self.update, args=()).start()

        return self

    def update(self):
        # Will infinitely loop until stopThread function called
        # and will continually pull images from the camera
        while True:

            if self.stopped:
                return

            grabbed, self.frame = self.stream.read()

    def read(self):
        # get the last image from camera
        return self.frame.copy()

    def stopThread(self):
        # stop the thread and stop grabbing images
        self.stopped = True


def detectSquare(contour):
    # Takes a contour and analyzes it using OpenCV to approximate the number
    # sides of the contour passed to the function.

    # The perimeter of the contour is found and then an epsilon is found.
    # Epsilon is the maximum distance from the actual contour to approximate
    # contour that is going to be created. This allows for shape detection
    # in the event that the shape has any slight curves or edges, or is slightly
    # covered. The contour approximation is the contour that approximately covers
    # the region of the contour and finally the contour area is found for
    # calculations below
    perimeter = cv2.arcLength(contour, True)
    epsilon = .02 * perimeter
    contourApproximation = cv2.approxPolyDP(contour, epsilon, True)

    # finding the contour area to test against height and width in to filter
    # out rectangles and squished squares so the only shapes detected are squares.
    # It is important to note that the width and height returned boundingRect
    # is the explicitly the height and width of the rectangle bounding a shape with
    # four vertices. A rectangle contour sloping at an angle of 45deg can be bounded
    # by a rectangle with the same height and width, so we must also use area in
    # order to filter out these shapes.
    contourArea = cv2.contourArea(contour)
    (x, y, w, h) = cv2.boundingRect(contourApproximation)

    # var is the variance between height and width (to test if it is a rect) and
    # areaVar is the variance between area of the bounding rectangle and the actual
    # contour area.
    var = w / h
    areaVar = contourArea / (w * h)

    # If statement that filters the shapes depending on ranges for which
    # a square should exist. Note: these may be changed depending on
    # factors such as angle.
    if (var >= .7 and var <= 1.3) and (areaVar >= .3):
        return 'Square'
    else:
        return 'Ignore'


def findSquare(contourList, image, color, ratio):
    # Finds all the detected contours and analyzes them for squares
    if len(contourList) == 0:
        return 0
    else:
        contour = max(contourList, key=cv2.contourArea)

    shape = detectSquare(contour)
    # If the shape is not square then loop to the next contour and check if
    # it's a square. Else, find its center and map it on the image.
    if shape != 'Square':
        return 0
    else:
        moment = cv2.moments(contour)

        # Since some contours can be really small a divide by zero error exception can be
        # thrown, so an if statement is used to pass over them.
        if moment["m00"] != 0:
            # cX is the CoM from left to right, and cY is the CoM from top to bottom
            cX = int((moment["m10"] / moment["m00"]) * ratio)
            cY = int((moment["m01"] / moment["m00"]) * ratio)

        # Here we find the location of the contour in the original image
        # before any resizing so that nothing is offset
        contour = contour.astype("float")
        contour *= ratio
        contour = contour.astype("int")

        # The size and location of a circle that will be used to surround the
        # squares is found below
        (circleX, circleY), radius = cv2.minEnclosingCircle(contour)
        center = (int(circleX), int(circleY))
        radius = int(radius)

        # A circle is drawn directly on top of the square so that it surrounds it
        image = cv2.circle(image, center, radius, (0, 255, 0), 2)
        cv2.putText(image, color + ' ' + shape, (cX, cY), cv2.FONT_HERSHEY_DUPLEX, 0.5, (20, 255, 255), 1)

        return 1


def showLag(frame, performance):
    # function to show column chart that shows all lag

    lagList = []

    y_pos = np.arange(len(frame))

    for key, value in performance.items():
        avg = round((sum(value) / len(value)), 2)
        lagList.append(avg)

    plt.bar(y_pos, lagList, align='center', alpha=0.5)
    plt.xticks(y_pos, frame)
    plt.ylabel('Time Frame')
    plt.title('Lag of Program')

    plt.show()


def lagDetect(lagID, timeOne, performance):
    # gets a total lag time from user defined time frame and appends it to a list
    # that is later used to find average lag
    timeTwo = cv2.getTickCount()

    lag = 1000 * ((timeTwo - timeOne) / cv2.getTickFrequency())

    performance[lagID].append(lag)

    return timeTwo


########################################################################################################################

# frame and performance for lag measurement
frame = ('Req Img', 'Img Manip', 'Find Cnt', 'Find Squares', 'Show Img', 'Wait For Key')
performance = {'Req Img': [],
               'Img Manip': [],
               'Find Cnt': [],
               'Find Squares': [],
               'Show Img': [],
               'Wait For Key': []
               }

# Initialize filters for the colors so the thresholded images only show
# the three tarps

# Pink Ranges
rangeMinOne = np.array([145, 121, 35], dtype=np.uint8)
rangeMaxOne = np.array([182, 240, 147], dtype=np.uint8)

# Blue Ranges
rangeMinTwo = np.array([84, 133, 28], dtype=np.uint8)
rangeMaxTwo = np.array([141, 255, 245], dtype=np.uint8)

# Yellow Ranges
rangeMinThree = np.array([23, 134, 108], dtype=np.uint8)
rangeMaxThree = np.array([44, 239, 169], dtype=np.uint8)

# Images saved so that all frames with 3 tarps are saved
imagesSaved = 0

imageStream = ThreadImage(sourceCam=0)
imageStream.startThread()

while True:
    timeOne = cv2.getTickCount()

    image = imageStream.read()

    # Resize the image gotten from the camera so that parsing is easier and faster,
    # It is noticed that at smaller resolutions, smaller squares seem to be harder to
    # detect. A ratio is kept so the image can be rescaled after parsing
    resizedImage = imutils.resize(image, width=400)
    ratio = image.shape[0] / float(resizedImage.shape[0])

    timeone = lagDetect('Req Img', timeOne, performance)

    # New image, so there should be zero squares detected so far.
    squaresDetected = 0

    # Image converted to HSV for easy color filtering.
    imageHSV = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2HSV)
    imageHSV = cv2.medianBlur(imageHSV, 5)

    # Creating thresholded images based in the the three ranges initialized above
    thresholdOne = cv2.inRange(imageHSV, rangeMinOne, rangeMaxOne)
    thresholdTwo = cv2.inRange(imageHSV, rangeMinTwo, rangeMaxTwo)
    thresholdThree = cv2.inRange(imageHSV, rangeMinThree, rangeMaxThree)

    # image manipulation techniques.
    kernelOne = np.ones((5, 5), np.uint8)
    kernelTwo = np.ones((5, 5), np.uint8)
    kernelThree = np.ones((5, 5), np.uint8)

    maskOne = cv2.morphologyEx(thresholdOne, cv2.MORPH_OPEN, kernelOne)
    maskOne = cv2.morphologyEx(maskOne, cv2.MORPH_CLOSE, kernelOne)
    maskOne = cv2.erode(maskOne, None, iterations=2)

    maskTwo = cv2.morphologyEx(thresholdTwo, cv2.MORPH_OPEN, kernelTwo)
    maskTwo = cv2.morphologyEx(maskTwo, cv2.MORPH_CLOSE, kernelTwo)
    maskTwo = cv2.erode(maskTwo, None, iterations=2)

    maskThree = cv2.morphologyEx(thresholdThree, cv2.MORPH_OPEN, kernelThree)
    maskThree = cv2.morphologyEx(maskThree, cv2.MORPH_CLOSE, kernelThree)
    maskThree = cv2.erode(maskThree, None, iterations=2)

    timeOne = lagDetect('Img Manip', timeOne, performance)

    # Create a list of contours represented on each thresholded image. Ideally, the ranges should
    # be tight enough so at most, 1 contour should be present per image
    contoursOne = cv2.findContours(maskOne, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursTwo = cv2.findContours(maskTwo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursThree = cv2.findContours(maskThree, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursOne = contoursOne[0] if imutils.is_cv2() else contoursOne[1]
    contoursTwo = contoursTwo[0] if imutils.is_cv2() else contoursTwo[1]
    contoursThree = contoursThree[0] if imutils.is_cv2() else contoursThree[1]

    timeOne = lagDetect('Find Cnt', timeOne, performance)

    # Find and display the squares then increment squares detected if a square was drawn
    squaresDetected += findSquare(contoursOne, image, 'Pink', ratio)
    squaresDetected += findSquare(contoursTwo, image, 'Blue', ratio)
    squaresDetected += findSquare(contoursThree, image, 'Yellow', ratio)

    timeOne = lagDetect('Find Squares', timeOne, performance)

    # Display the image on the computer
    cv2.imshow("image", image)

    timeOne = lagDetect('Show Img', timeOne, performance)

    # If there are three squares detected, save the image in the
    # working directory
    if squaresDetected == 3:
        cv2.imwrite("image " + str(imagesSaved) + ".png", image)
        imagesSaved += 1

    # If the escape key is hit, then the while loop is exited and the program ends
    if cv2.waitKey(1) & 0xFF is 27:
        break

    timeOne = lagDetect('Wait For Key', timeOne, performance)

# Destroy the display
cv2.destroyAllWindows()
imageStream.stopThread()
showLag(frame, performance)
