#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File name: "ShapeDetect.py"
Date created: 9/29/2017
Date last modified: 10/15/2017
Python Version: 3.6

A simple script for detecting and tracking squares of a single color and feeds back the
color of each square.
"""

__author__ = "Hicham Belhseine"
__email__ = "hichambelhseine@gmail.com"

import cv2
import imutils
import numpy as np


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

    # the length of the tuple contourApproximation above gives the number
    # of vertices present in the contour, allowing for shape identification
    vertices = len(contourApproximation)

    # If there are four vertices, then the shape is a square or a rectangle
    # but we must do a little more testing to filter out rectangles
    if vertices != 4:
        return 'Ignore'
    else:
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
        if (var >= .9 and var <= 1.1) and (areaVar >= .3) and (w < 300):
            return 'Square'
        else:
            return 'Ignore'


def findSquare(contourList, image, color):
    # Finds all the detected contours and analyzes them for squares
    for contour in contourList:
        shape = detectSquare(contour)
        # If the shape is not square then loop to the next contour and check if
        # it's a square. Else, find its center and map it on the image.
        if shape != 'Square':
            continue
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

    return 0


# Initialize filters for the colors so the thresholded images only show
# the three tarps

listAvg = []

# Pink Ranges
rangeMinOne = np.array([122, 129, 215], dtype=np.uint8)
rangeMaxOne = np.array([255, 236, 255], dtype=np.uint8)

# Blue Ranges
rangeMinTwo = np.array([255, 255, 255], dtype=np.uint8)
rangeMaxTwo = np.array([255, 255, 255], dtype=np.uint8)

# Yellow Ranges
rangeMinThree = np.array([255, 255, 255], dtype=np.uint8)
rangeMaxThree = np.array([255, 255, 255], dtype=np.uint8)

# Images saved so that all frames with 3 tarps are saved
imagesSaved = 0

# A video capture is created that uses the default camera. For other
# cameras, try incrementing the argument to 1 or 2
videoCapture = cv2.VideoCapture(0)

while True:
    e1 = cv2.getTickCount()

    # Get the image from the camera and resize it so that parsing is easier and faster,
    # It is noticed that at smaller resolutions, smaller squares seem to be harder to
    # detect. A ratio is kept so the image can be rescaled after parsing
    ret, image = videoCapture.read()
    resizedImage = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resizedImage.shape[0])

    # New image, so there should be zero squares detected so far.
    squaresDetected = 0

    # Image converted to HSV for easy color filtering.
    imageHSV = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2HSV)

    # Creating thresholded images based in the the three ranges initialized above
    thresholdOne = cv2.inRange(imageHSV, rangeMinOne, rangeMaxOne)
    thresholdTwo = cv2.inRange(imageHSV, rangeMinTwo, rangeMaxTwo)
    thresholdThree = cv2.inRange(imageHSV, rangeMinThree, rangeMaxThree)

    # Create a list of contours represented on each thresholded image. Ideally, the ranges should
    # be tight enough so at most, 1 contour should be present per image
    contoursOne = cv2.findContours(thresholdOne, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursTwo = cv2.findContours(thresholdTwo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursThree = cv2.findContours(thresholdThree, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursOne = contoursOne[0] if imutils.is_cv2() else contoursOne[1]
    contoursTwo = contoursTwo[0] if imutils.is_cv2() else contoursTwo[1]
    contoursThree = contoursThree[0] if imutils.is_cv2() else contoursThree[1]

    # Find and display the squares then increment squares detected if a square was drawn
    squaresDetected += findSquare(contoursOne, image, 'Pink')
    squaresDetected += findSquare(contoursTwo, image, 'Blue')
    squaresDetected += findSquare(contoursThree, image, 'Yellow')

    # Display the image on the computer
    cv2.imshow("image", image)

    # If there are three squares detected, save the image in the
    # working directory
    if squaresDetected == 3:
        cv2.imwrite("image " + str(imagesSaved) + ".png", image)
        imagesSaved += 1

    # If the escape key is hit, then the while loop is exited and the program ends
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    e2 = cv2.getTickCount()

    # Displaying time to process
    listAvg.append(round(1000 * ((e2 - e1) / cv2.getTickFrequency()), 2))

# Destroy the display
videoCapture.release()
cv2.destroyAllWindows()
