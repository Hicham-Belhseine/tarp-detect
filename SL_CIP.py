#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File name: "SL_CIP.py"
Date created: 1/17/2018
Date last modified: 10/4/2018
Python Version: 3.6

A simple script for determining a region of interest based on target color
and shape that switches towards the use of the CAMShift algorithm after a
region of interest is found.
"""

__author__ = "Hicham Belhseine"
__email__ = "hichambelhseine@gmail.com"

import cv2
import numpy as np
import imutils
import math


class ROIDetection:
    """
    A set of functions useful for searching for and isolating a region of interest
    based on color.
    """

    def __init__(self, rangeMinOne, rangeMaxOne, rangeMinTwo, rangeMaxTwo):
        # Initialize color ranges
        self.rangeMinOne = rangeMinOne
        self.rangeMaxOne = rangeMaxOne
        self.rangeMinTwo = rangeMinTwo
        self.rangeMaxTwo = rangeMaxTwo

    def searchForROI(self, image):
        # Converting image to HSV
        imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Both masks are eroded and dilated to remove any noise from the image
        # both inside and outside the specified color ranges

        # First mask for color range one
        maskOne = cv2.inRange(imageHSV, self.rangeMinOne, self.rangeMaxOne)
        maskOne = cv2.erode(maskOne, None, iterations=1)
        maskOne = cv2.dilate(maskOne, None, iterations=1)

        # Second mask for color range two
        maskTwo = cv2.inRange(imageHSV, self.rangeMinTwo, self.rangeMaxTwo)
        maskTwo = cv2.erode(maskTwo, None, iterations=1)
        maskTwo = cv2.dilate(maskTwo, None, iterations=1)

        # Get largest contours in the first mask
        contoursOne = cv2.findContours(maskOne, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contoursOne = contoursOne[0] if imutils.is_cv2() else contoursOne[1]

        # Get largest contours in the second mask
        contoursTwo = cv2.findContours(maskTwo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contoursTwo = contoursTwo[0] if imutils.is_cv2() else contoursTwo[1]

        # Check contour location
        self.centerContourOne, self.contourOne = self.contourLocation(self.contoursOne)
        self.centerContourTwo, self.contourTwo = self.contourLocation(self.contoursTwo)

    def checkCenters(self):
        # Checks to see if the attained contour centers are close enough to be two adacent
        # tarps

        # First, check if the contours are empty
        if not self.centerContourOne or not self.centerContourTwo:
            return False
        else:
            x1 = self.centerContourOne[0]
            y1 = self.centerContourOne[1]
            x2 = self.centerContourTwo[0]
            y2 = self.centerContourTwo[1]

            # Size of region
            regionDimensions = 2 * int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

            if regionDimensions > 800:
                return False
            else:
                # Check to see if the contour areas are similar
                areaCntOne = cv2.contourArea(max(self.contoursOne, key=cv2.contourArea))
                areaCntTwo = cv2.contourArea(max(self.contoursTwo, key=cv2.contourArea))

                # Top and bottom error bounds
                errorTop = 1.6 * areaCntOne
                errorBot = .4 * areaCntOne

                if areaCntTwo < errorBot or areaCntTwo > errorTop:
                    return False
                elif areaCntOne == 0 or areaCntTwo == 0:
                    return False
                else:
                    return True

    def contourLocation(self, contourList):
        # Returns contour location (x and y)

        if len(contourList) is 0:
            return [], []
        else:
            contour = max(contourList, key=cv2.contourArea)

            moment = cv2.moments(contour)

            if moment["m00"] != 0:

                centerX = int((moment["m10"] / moment["m00"]))
                centerY = int((moment["m01"] / moment["m00"]))

                return [centerX, centerY], contour
            else:
                return [], []

    def getROI(self, image):
        # Get the image of the region of interest used to initialize
        # CAMShift

        # Get the region bounds
        x1 = self.centerContourOne[0]
        x2 = self.centerContourTwo[0]

        y1 = self.centerContourOne[1]
        y2 = self.centerContourTwo[1]

        regionDimensions = 4 * int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        midPoint = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        # Create a minimally enclosing rectangle to contain the image
        sideLeft = midPoint[0] - regionDimensions
        sideRight = midPoint[0] + regionDimensions
        sideTop = midPoint[1] - regionDimensions
        sideBottom = midPoint[1] + regionDimensions

        # Ensure the rectangle does go past the bounds of the image
        if sideLeft < 0:
            sideLeft = 0
        if sideRight < 0:
            sideRight = 0
        if sideTop < 0:
            sideTop = 0
        if sideBottom < 0:
            sideBottom = 0

        # Crop original image
        imageROI = image[sideTop:sideBottom, sideLeft:sideRight]

        # region dimension
        trackWindow = (sideLeft, sideTop, regionDimensions, regionDimensions)

        return imageROI, trackWindow


class CAMShift:
    """
    A set of functions useful for initializing a region of interest and continually
    tracking it until certain specifications are met.
    """

    def __init__(self, rangeMinOne, rangeMaxOne, rangeMinTwo, rangeMaxTwo):
        # Termination criteria
        self.termCriteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2, 1)

        # Tarp ranges
        self.rangeMinOne = rangeMinOne
        self.rangeMaxOne = rangeMaxOne
        self.rangeMinTwo = rangeMinTwo
        self.rangeMaxTwo = rangeMaxTwo

    def getROIMask(self, roi):
        # Get HSV from ROI for histogram backprojection and masking
        roiHSV = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Masks for both ranges
        maskOne = cv2.inRange(roiHSV, self.rangeMinOne, self.rangeMaxOne)
        maskTwo = cv2.inRange(roiHSV, self.rangeMinTwo, self.rangeMaxTwo)

        # combine mask into single image
        combinedMask = cv2.bitwise_or(maskOne, maskTwo)

        return combinedMask

    def camShiftTracking(self, roi, roiMask, image, imageStream, trackWindow):
        # Termination criteria
        termCrit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3, 1)

        self.trackWindow = trackWindow

        roiHSV = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Histogram backprojection
        roiHist = cv2.calcHist([roiHSV], [0], roiMask, [16], [0, 180])
        roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)

        # initial error is zero (frames where tarps are not detected)
        error = 0

        while error < 8:
            # Get the next image in the image stream
            ret, image = imageStream.read()

            # Check to see if image is not NoneType
            if ret == True:
                # Get the HSV image
                imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                dst = cv2.calcBackProject([imageHSV], [0], roiHist, [0, 180], 1)

                # Find new tracking window
                ret, self.trackWindow = cv2.CamShift(dst, self.trackWindow, termCrit)

                points = cv2.boxPoints(ret)
                points = np.int0(points)

                if points[0] is [0, 0]:
                    continue

                imageCAMShift = cv2.polylines(image, [points], True, 255, 1)

                # New window of analysis
                windowOfAnalysis = self.getWindow(points)

                # Define new region of interest
                roiNew = image[windowOfAnalysis[2]:windowOfAnalysis[3], windowOfAnalysis[0]:windowOfAnalysis[1]]

                # check if tarps are found
                tarpsFound = self.findTarps(image, roiNew, windowOfAnalysis)

                # Updating error count
                if not tarpsFound:
                    error += 1
                else:
                    cv2.imshow("image", image)
                    if error > 0:
                        error -= 1

            else:
                error += 1

            if error == 4:
                break

            if cv2.waitKey(1) & 0xFF is 27:
                break

    def getWindow(self, points):
        # Returns the window for CAMShift
        xList = []
        yList = []

        for point in points:
            xList.append(point[0])
            yList.append(point[1])

        return min(xList), max(xList), min(yList), max(yList)

    def findTarps(self, image, roi, points):
        # Find tarps in the CAMShift window
        roiHSV = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Both masks are eroded and dilated to remove any noise from the image
        # both inside and outside the specified color ranges

        # First mask for color range one
        maskOne = cv2.inRange(roiHSV, self.rangeMinOne, self.rangeMaxOne)
        maskOne = cv2.erode(maskOne, None, iterations=1)
        maskOne = cv2.dilate(maskOne, None, iterations=1)

        # Second mask for color range Two
        maskTwo = cv2.inRange(roiHSV, self.rangeMinTwo, self.rangeMaxTwo)
        maskTwo = cv2.erode(maskTwo, None, iterations=1)
        maskTwo = cv2.dilate(maskTwo, None, iterations=1)

        # Get both contours
        contoursOne = cv2.findContours(maskOne, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoursOne = contoursOne[0] if imutils.is_cv2() else contoursOne[1]

        contoursTwo = cv2.findContours(maskTwo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoursTwo = contoursTwo[0] if imutils.is_cv2() else contoursTwo[1]

        # Find the centers and the minimally enclosing radious of the contours
        self.centerContourOne, radiusOne = self.contourLocation(contoursOne)
        self.centerContourTwo, radiusTwo = self.contourLocation(contoursTwo)

        # Check if either contour is empty
        if not self.centerContourOne:
            return False
        if not self.centerContourTwo:
            return False

        # Check similarity in contour area
        areaCntOne = cv2.contourArea(max(contoursOne, key=cv2.contourArea))
        areaCntTwo = cv2.contourArea(max(contoursTwo, key=cv2.contourArea))

        # Error top and bottom bounds
        errorTop = 1.5 * areaCntOne
        errorBot = .5 * areaCntOne

        if areaCntTwo < errorBot or areaCntTwo > errorTop:
            return False
        if areaCntOne == 0 or areaCntTwo == 0:
            return False

        # Find center of contours relative to entire image
        self.centerContourOne[0] = points[0] + self.centerContourOne[0]
        self.centerContourOne[1] = points[2] + self.centerContourOne[1]
        self.centerContourTwo[0] = points[0] + self.centerContourTwo[0]
        self.centerContourTwo[1] = points[2] + self.centerContourTwo[1]

        # Outline the tarps in the image
        image = cv2.circle(image, tuple(self.centerContourOne), int(radiusOne), (0, 255, 0), 2)
        image = cv2.circle(image, tuple(self.centerContourTwo), int(radiusTwo), (0, 255, 0), 2)

        return True

    def contourLocation(self, contourList):
        # Finds the contour location

        # If the contour list is empty, return false
        if len(contourList) is 0:
            return [], []
        else:
            # Take the largest contour
            contour = max(contourList, key=cv2.contourArea)

        # Center of contour
        moment = cv2.moments(contour)

        # Bypass div by 0 error
        if moment["m00"] != 0:
            centerX = int((moment["m10"] / moment["m00"]))
            centerY = int((moment["m01"] / moment["m00"]))

            (circleX, circleY), radius = cv2.minEnclosingCircle(contour)

            return [centerX, centerY], radius

        else:
            return [], []


def main():
    # Defining tarp predetermined color values
    # Peach Ranges
    rangeMinOne = np.array([152, 40, 120], dtype=np.uint8)
    rangeMaxOne = np.array([190, 101, 180], dtype=np.uint8)

    # Blue Ranges
    rangeMinTwo = np.array([90, 100, 60], dtype=np.uint8)
    rangeMaxTwo = np.array([134, 240, 170], dtype=np.uint8)

    # Image Stream
    imageStream = cv2.VideoCapture("CIP_test.mp4")

    # Take first image from camera and get shape
    ret, image = imageStream.read()
    height, width, layers = image.shape

    # Prep Region of Interest finder
    roiDetect = ROIDetection(rangeMinOne, rangeMaxOne, rangeMinTwo, rangeMaxTwo)

    # Prep CAMShift finder
    camShift = CAMShift(rangeMinOne, rangeMaxOne, rangeMinTwo, rangeMaxTwo)

    while True:
        # Take in next image from image stream
        ret, image = imageStream.read()

        # Search for a ROI
        roiDetect.searchForROI(image)

        # Check if ROI found
        roiFound = roiDetect.checkCenters()

        if roiFound:
            # If a region of interest is found, get the location of the region of
            # interest
            roi, trackWindow = roiDetect.getROI(image)

            roiMask = camShift.getROIMask(roi)

            # Begin applying CAMShift algorithm based on attained region of interest
            camShift.camShiftTracking(roi, roiMask, image, imageStream, trackWindow)
        else:
            cv2.imshow("image", image)

        # Exit program is 'esc' key is hit
        if cv2.waitKey(1) & 0xFF is 27:
            break

    imageStream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
