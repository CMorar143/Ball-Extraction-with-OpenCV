               ############## OPENING COMMENT ##############
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Program to Detect and Extract a White Ball From an Image.					  #
# Author: Cian Morar 														  #
# Date:  2019																  #
#																			  #
# The algorithm works as follows:											  #
#																			  #
# The user selects an image.												  #
#																			  #
# The circle hough transform method is used.								  #
# This works by first blurring the image and then using the HoughCircles	  #
# method to detect the circle in the image.									  #
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
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #