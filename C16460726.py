               ############## OPENING COMMENT ##############
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Program to Detect and Extract a White Ball From an Image.					  #
# Author: Cian Morar 														  #
# Date:  2019																  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# import the necessary packages:
import cv2
import easygui as gui

# Function to apply a threshold to an image
# It is used to remove the background:
def ApplyMask(threshold):
	# Apply the mask to each color channel in the image:
	extracted_sharkB = cv2.bitwise_or(threshold, Enchanced_B)
	extracted_sharkG = cv2.bitwise_or(threshold, Enchanced_G)
	extracted_sharkR = cv2.bitwise_or(threshold, Enchanced_R)

	merged = cv2.merge((extracted_sharkB, extracted_sharkG, extracted_sharkR))

	return merged

# Repeat until the user chooses to exit:

F = gui.fileopenbox()
I = cv2.imread("./spottheball.jpg")

YUV = cv2.cvtColor(I, cv2.COLOR_BGR2YUV)
Y, U, V = cv2.split(YUV)

HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
H, S, HV = cv2.split(HSV)



























# # Using the Contrast Limited Adaptive Histogram Equalization class to enhance the contrast
# # Create the CLAHE object and set the clip limit and tile grid size:
# CLAHE = cv2.createCLAHE(clipLimit = 4.5, tileGridSize = (3,3))

# # This improves definition in the image:
# Enhanced_Y = CLAHE.apply(Y)

# # This is used to remove the background:
# Enhanced_U = CLAHE.apply(U)

# # Create initial mask to remove the background:
# _, Threshold = cv2.threshold(Enhanced_U, 176, 255, cv2.THRESH_TRUNC)
# Adaptive_Threshold = cv2.adaptiveThreshold(Threshold, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 275, 2)

# Enchanced_YUV = cv2.merge((Enhanced_Y,U,V))
# Enchanced_BGR = cv2.cvtColor(Enchanced_YUV, cv2.COLOR_YUV2BGR)
# Enchanced_B, Enchanced_G, Enchanced_R = cv2.split(Enchanced_BGR)

# Removed_Background = ApplyMask(Adaptive_Threshold)

# # Create the final mask to clean more of the noise:
# Enhanced_YUV2 = cv2.cvtColor(Removed_Background, cv2.COLOR_BGR2YUV)
# u = Enhanced_YUV2[:,:,1]

# _, maxVal, _, _ = cv2.minMaxLoc(u)
# masked_range = cv2.inRange(u, 0, maxVal-3)
# final_mask = cv2.bitwise_xor(masked_range, Adaptive_Threshold)
# final_mask = cv2.bitwise_not(final_mask)

# final_image = ApplyMask(final_mask)

# cv2.imwrite("./final_image.png", final_image)