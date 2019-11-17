               ############## OPENING COMMENT ##############
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Program to Detect and Remove a White Ball From an Image					  #
# Author: Cian Morar 														  #
# Date: November 2019														  #
#																			  #
# The algorithm works as follows:											  #
#																			  #
# The user selects an image.												  #
#																			  #
# The circle hough transform method is used.								  #
# This works by first blurring a greyscale of the image and then using the 	  #
# HoughCircles method to detect the circle in the image.					  #
# 																			  #
# The circle hough transform will return the x and y co-ordinates for the 	  #
# the centre of the circle and the radius of the cirlce.					  #
# 																			  #
# These values are converted to integers and the radius is increased by 10.	  #
# This is done so that the region of interest will extend beyond the 		  #
# detected circle (i.e. the ball).											  #
#																			  #
# Using the centre co-ordinates and radius obtained from the circle hough,	  #
# the program draws a white circle in the same position on a black background.#
#																			  #
# This undergoes a bit-wise-and operation with the original image in order    #
# to remove the ball.														  #
#																			  #
# The script then extracts the grass from the right, left, bottom, and top of #
# the ball.																	  #
#																			  #
# It checks which of these four is the "most green" and then translates	this  #
# extracted grass into the position of where the ball used to be.			  #
#  												  		  					  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# import the necessary packages:
import cv2
import easygui as gui
import numpy as np

# Messages for the Graphical User Interface:
opening_message = "This Application Allows You to Remove a Ball From an Image.\n\n\n"
instructions = 	  "Please Choose The Image That You Would Like to Use."
closing_message = "\tSee if You Can Find Where The Ball Was!\n\n\tWould You Like to Select Another Picture?"
final_message =   "\tHave a great day!"
choices = 		  ["Yes", "No"]


def FindGrass(xCircle, yCircle, yStart, yFinish, xStart, xFinish):
	Extracted_grass = np.zeros(shape=[h, w, d], dtype=np.uint8)
	Extracted_grass.fill(255)
	cv2.circle(Extracted_grass, (xCircle, yCircle), r, (0, 0, 0), -1)
	grass_extraction = cv2.subtract(I, Extracted_grass)

	# Zoom in and get average pixel value of the green channel
	cropped_grass = grass_extraction[yStart:yFinish, xStart:xFinish]
	cropped_grass = cropped_grass[:, :, 1]
	mean = cv2.mean(cropped_grass)

	return mean[0], grass_extraction

def FillEmptySpot(rx, ry, grass_extraction):
	rows, cols = grass_extraction.shape[:2]
	M = np.float32([[1, 0, rx], [0, 1, ry]])
	dst = cv2.warpAffine(grass_extraction, M, (cols, rows))
	dst = cv2.bitwise_or(dst, Removed_ball)

	return dst

# Repeat until the user chooses to exit:
while (1):
	gui.msgbox(opening_message + instructions, "Hello!")
	F = gui.fileopenbox()
	I = cv2.imread(F)
	h, w, d = I.shape

	Extracted_ball = np.zeros(shape=[h, w, d], dtype=np.uint8)

	G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
	G = cv2.medianBlur(G, 17)

	# The minDist parameter is set to the width of the image to ensure that only one circle is returned
	circles = cv2.HoughCircles(G, cv2.HOUGH_GRADIENT, 1, minDist=I.shape[1], param1=50, param2=30, minRadius=0, maxRadius=0)
	detected = np.uint16(np.around(circles))

	x, y, r = detected[0, 0]
	r = r + 10
	cv2.circle(Extracted_ball, (x, y), (r), (255, 255, 255), -1)

	Removed_ball = cv2.subtract(I, Extracted_ball)
	Extracted_ball = cv2.bitwise_not(Extracted_ball)
	Extracted_ball = cv2.subtract(I, Extracted_ball)

	meanRight, grass_extractionR = FindGrass(x + (r*2), y, y-r, y+r, x+r, x + (r*3))
	meanBottom, grass_extractionB = FindGrass(x, y + (r*2), y+r, y+(r*3), x-r, x+r)
	meanLeft, grass_extractionL = FindGrass(x - (r*2), y, y-r, y+r, x-(r*3), x-r)
	meanTop, grass_extractionT = FindGrass(x, y - (r*2), y-(r*3), y-r, x-r, x+r)

	means = (meanRight, meanBottom, meanLeft, meanTop)
	highestMean = max(means)

	if meanRight == highestMean:
		final_image = FillEmptySpot(-r*2, 0, grass_extractionR)

	if meanBottom == highestMean:
		final_image = FillEmptySpot(0, -r*2, grass_extractionB)

	if meanLeft == highestMean:
		final_image = FillEmptySpot(r*2, 0, grass_extractionL)

	if meanTop == highestMean:
		final_image = FillEmptySpot(0, r*2, grass_extractionT)

	cv2.imwrite("./final_image.png", final_image)

	reply = gui.buttonbox(closing_message, image = "./final_image.png", choices = choices)

	# Close the application if they click no or if they try to close the window
	if reply == 'No' or reply != 'Yes':
		gui.msgbox(final_message, title = "Thanks!")
		exit(0)

	
				############## CLOSING COMMENT ##############
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Program to Detect and Remove a Ball From an Image							  #
# Author: Cian Morar 														  #
# Date: November 2019														  #
# 																			  #
# This program works well for all images due to the lack of hard-coded values #
# 																			  #
# It was difficult however to make this work for the snooker image due to 	  #
# the shadow underneath the ball.							  				  #
# 																			  #
# Initially I tried using blurring effects however this made it really 		  #
# obvious on the golf test image.							  				  #
# 																			  #
# For this reason I decided the opt for the translation method of checking	  #
# all the regions around the ball to see which is the most green and to 	  #
# ensure that there were no objects like a player's foot being moved into	  #
# the spot.			  														  #
# 																			  #
# This translation method works better than blurring for both the 		  	  #
# football and golf images.													  #
# 																			  #
# Additionaly, I have tested this script on other similar images and they  	  #
# provide equally satisfying results.										  #
# 																			  #
# I will say however, that it works best on images most like the golf image	  #
# i.e. when the grass is slightly more detailed								  #
# 																			  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #