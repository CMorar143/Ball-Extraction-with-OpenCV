               ############## OPENING COMMENT ##############
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Program to Detect and Extract a White Ball From an Image.					  #
# Author: Cian Morar 														  #
# Date:  2019																  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# import the necessary packages:
import cv2
import easygui as gui
from matplotlib import pyplot as plt
import numpy as np


def FindGrass(xCircle, yCircle, yStart, yFinish, xStart, xFinish):
	# Extract grass from the right hand side
	Extracted_grass = np.zeros(shape=[h, w, 3], dtype=np.uint8)
	cv2.circle(Extracted_grass, (xCircle, yCircle), r, (255, 255, 255), -1)
	grass_extraction = cv2.subtract(I, Extracted_grass)
	Extracted_grass = cv2.bitwise_not(Extracted_grass)
	grass_extraction2 = cv2.subtract(I, Extracted_grass)

	# Zoom in and get average pixel value of extracted area
	cropped_grass = grass_extraction2[yStart:yFinish, xStart:xFinish]
	cropped_grass = cropped_grass[:,:,1]
	mean = cv2.mean(cropped_grass)

	return mean[0], grass_extraction2

def FillEmptySpot(rx, ry, grass_extraction):
	rows,cols = grass_extraction.shape[:2]
	M = np.float32([[1,0,rx],[0,1,ry]])
	dst = cv2.warpAffine(grass_extraction,M,(cols,rows))
	dst = cv2.bitwise_or(dst, Test_extraction)

	return dst


# F = gui.fileopenbox()
I = cv2.imread("./Images/snooker.jpg")
# I = cv2.imread(F)
h, w, d = I.shape

Extracted_ball = np.zeros(shape=[h, w, 3], dtype=np.uint8)
Extracted_grass = np.zeros(shape=[h, w, 3], dtype=np.uint8)
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
G = cv2.medianBlur(G, 17)

circles = cv2.HoughCircles(G, cv2.HOUGH_GRADIENT, 1, I.shape[0], param1=50, param2=30, minRadius=0, maxRadius=0)

detected = np.uint16(np.around(circles))

# Increase the radius by 10 to go beyond the circumference
for (x, y, r) in detected[0, :]:
	r = r + 10
	cv2.circle(Extracted_ball, (x, y), (r), (255, 255, 255), -1)

Test_extraction = cv2.subtract(I, Extracted_ball)
Extracted_ball = cv2.bitwise_not(Extracted_ball)
Extracted_ball = cv2.subtract(I, Extracted_ball)

# Extract grass from the right hand side
meanRight, grass_extraction2 = FindGrass(x + (r*2), y, y-r, y+r, x+r, x + (r*3))
print("right: ", meanRight)

# Extract grass from the bottom
meanBottom, grass_extraction2B = FindGrass(x, y + (r*2), y+r, y+(r*3), x-r, x+r)
print("bottom: ", meanBottom)

# Extract grass from the left hand side
meanLeft, grass_extraction2L = FindGrass(x - (r*2), y, y-r, y+r, x-(r*3), x-r)
print("left: ", meanLeft)

# Extract grass from the top
meanTop, grass_extraction2T = FindGrass(x, y - (r*2), y-(r*3), y-r, x-r, x+r)
print("top: ", meanTop)

# Find which side has the highest amount of green
means = (meanRight, meanBottom, meanLeft, meanTop)
highestMean = max(means)

if meanRight == highestMean:
	dst = FillEmptySpot(-r*2, 0, grass_extraction2)

if meanBottom == highestMean:
	dst = FillEmptySpot(0, -r*2, grass_extraction2B)

if meanLeft == highestMean:
	dst = FillEmptySpot(r*2, 0, grass_extraction2L)

if meanTop == highestMean:
	dst = FillEmptySpot(0, r*2, grass_extraction2T)

cv2.imshow("Test_extraction", Test_extraction)

# cv2.imshow("grass_extraction", grass_extraction)
# cv2.imshow("grass_extraction2", grass_extraction2)
# cv2.imshow("grass_extraction2B", grass_extraction2B)
# cv2.imshow("grass_extraction2L", grass_extraction2L)
# cv2.imshow("grass_extraction2T", grass_extraction2T)
# cv2.imshow("output2", output2)
cv2.imshow("dst", dst)
# cv2.imshow("cropped_grass", cropped_grass)
# cv2.imshow("cropped_grassB", cropped_grassB)
# cv2.imshow("cropped_grassL", cropped_grassL)
# cv2.imshow("cropped_grassT", cropped_grassT)
cv2.imshow("Extracted_ball", Extracted_ball)
cv2.imshow("Extracted_grass", Extracted_grass)
cv2.waitKey(0)