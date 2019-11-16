               ############## OPENING COMMENT ##############
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Program to Detect and Extract a White Ball From an Image.					  #
# Author: Cian Morar 														  #
# Date:  2019																  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# import the necessary packages:
import cv2
import easygui as gui
import numpy as np

# Messages for the Graphical User Interface:
opening_message = "This Application Allows You to Remove a Ball From an Image.\n\n\n"
instructions = 	  "Please Choose The Image That You Would Like to Use."
closing_message = "\tSee if You Can Find Where The Ball Was\n\tWould You Like to Select Another Picture?"
final_message =   "\tHave a great day!"
choices = 		  ["Yes", "No"]


def FindGrass(xCircle, yCircle, yStart, yFinish, xStart, xFinish):
	# Extract grass from the right hand side
	Extracted_grass = np.zeros(shape=[h, w, d], dtype=np.uint8)
	Extracted_grass.fill(255)
	cv2.circle(Extracted_grass, (xCircle, yCircle), r, (0, 0, 0), -1)
	grass_extraction = cv2.subtract(I, Extracted_grass)

	# Zoom in and get average pixel value of extracted area
	cropped_grass = grass_extraction[yStart:yFinish, xStart:xFinish]
	cropped_grass = cropped_grass[:,:,1]
	mean = cv2.mean(cropped_grass)

	return mean[0], grass_extraction

def FillEmptySpot(rx, ry, grass_extraction):
	rows,cols = grass_extraction.shape[:2]
	M = np.float32([[1, 0, rx], [0, 1, ry]])
	dst = cv2.warpAffine(grass_extraction, M, (cols,rows))
	dst = cv2.bitwise_or(dst, Removed_ball)

	return dst

# Repeat until the user chooses to exit:
while (1):
	gui.msgbox(opening_message + instructions, "Hello!")
	# F = gui.fileopenbox()
	I = cv2.imread("./Images/spottheball.jpg")
	# I = cv2.imread(F)
	h, w, d = I.shape

	Extracted_ball = np.zeros(shape=[h, w, d], dtype=np.uint8)

	G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
	G = cv2.medianBlur(G, 17)

	circles = cv2.HoughCircles(G, cv2.HOUGH_GRADIENT, 1, I.shape[0], param1=50, param2=30, minRadius=0, maxRadius=0)

	detected = np.uint16(np.around(circles))

	x, y, r = detected[0, 0]
	r = r + 10
	cv2.circle(Extracted_ball, (x, y), (r), (255, 255, 255), -1)

	Removed_ball = cv2.subtract(I, Extracted_ball)
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
		final_image = FillEmptySpot(-r*2, 0, grass_extraction2)

	if meanBottom == highestMean:
		final_image = FillEmptySpot(0, -r*2, grass_extraction2B)

	if meanLeft == highestMean:
		final_image = FillEmptySpot(r*2, 0, grass_extraction2L)

	if meanTop == highestMean:
		final_image = FillEmptySpot(0, r*2, grass_extraction2T)

	cv2.imwrite("./final_image.png", final_image)

	reply = gui.buttonbox(closing_message, image = "./final_image.png", choices = choices)

	# Close the application if they click no or if they try to close the window
	if reply == 'No' or reply != 'Yes':
		gui.msgbox(final_message, title = "Thanks!")
		exit(0)