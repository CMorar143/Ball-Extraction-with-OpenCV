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

# F = gui.fileopenbox()
I = cv2.imread("./Images/spottheball.jpg")
# I = cv2.imread(F)
output = I.copy()
h, w, d = I.shape

Extracted_ball = np.zeros(shape=[h, w, 3], dtype=np.uint8)
Extracted_grass = np.zeros(shape=[h, w, 3], dtype=np.uint8)
blank = np.zeros(shape=[h, w, 3], dtype=np.uint8)

YUV = cv2.cvtColor(I, cv2.COLOR_BGR2YUV)
Y, U, V = cv2.split(YUV)

HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
H, S, HV = cv2.split(HSV)

# Using the Contrast Limited Adaptive Histogram Equalization class to enhance the contrast
# Create the CLAHE object and set the clip limit and tile grid size:
CLAHE = cv2.createCLAHE(clipLimit = 4.5, tileGridSize = (3,3))

# This improves definition in the image:
Enhanced_Y = CLAHE.apply(Y)
E = cv2.equalizeHist(Y)

_, Threshold = cv2.threshold(E, 248, 255, cv2.THRESH_BINARY)
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
G = cv2.medianBlur(G, 17)
# G = cv2.medianBlur(G, 19)

circles = cv2.HoughCircles(G, cv2.HOUGH_GRADIENT, 1, I.shape[0], param1=50, param2=30, minRadius=0, maxRadius=0)

detected = np.uint16(np.around(circles))

# Increase the radius by 3 to go beyond the circumference
for (x, y, r) in detected[0, :]:
	r = r + 3
	# cv2.circle(output, (x, y), 2, (0, 0, 255), 1)
	# cv2.circle(output, (x, y), r, (255, 255, 255), 1)
	cv2.circle(Extracted_ball, (x, y), (r), (255, 255, 255), -1)

Test_extraction = cv2.subtract(output, Extracted_ball)
Extracted_ball = cv2.bitwise_not(Extracted_ball)
Test_extraction2 = cv2.subtract(output, Extracted_ball)

cv2.circle(Extracted_grass, (x + (r*2), y), (r), (255, 255, 255), -1)

grass_extraction = cv2.subtract(output, Extracted_grass)
Extracted_grass = cv2.bitwise_not(Extracted_grass)
grass_extraction2 = cv2.subtract(output, Extracted_grass)

rows,cols = grass_extraction2.shape[:2]
M = np.float32([[1,0,-r*2],[0,1,0]])
dst = cv2.warpAffine(grass_extraction2,M,(cols,rows))

# cv2.imshow("G", G)
# cv2.imshow("Threshold", Threshold)
cv2.imshow("Test_extraction", Test_extraction)
cv2.imshow("grass_extraction", grass_extraction)
cv2.imshow("grass_extraction2", grass_extraction2)
# cv2.imshow("Test_extraction2", Test_extraction2)
cv2.imshow("output", output)
cv2.imshow("dst", dst)
# cv2.imshow("blank", Extracted_ball)
cv2.waitKey(0)




################# Plot Diagram #################

fig = plt.figure()
gs = fig.add_gridspec(3, 2)

ax1 = fig.add_subplot(gs[0,0])
ax1.imshow(Y, cmap='gray')

ax2 = fig.add_subplot(gs[0,1])
ax2.hist(Y.ravel(), bins=256, range=[0,256])

ax3 = fig.add_subplot(gs[1,0])
ax3.imshow(E, cmap='gray')

ax4 = fig.add_subplot(gs[1,1])
ax4.hist(E.ravel(), bins=256, range=[0,256])

ax5 = fig.add_subplot(gs[2,0])
ax5.imshow(Enhanced_Y, cmap='gray')

ax6 = fig.add_subplot(gs[2,1])
ax6.hist(Enhanced_Y.ravel(), bins=256, range=[0,256])

# ax7 = fig.add_subplot(gs[3,0])
# ax7.imshow(gray, cmap='gray')

# ax8 = fig.add_subplot(gs[3,1])
# ax8.hist(gray.ravel(), bins=256, range=[0,256])

# plt.show()
# cv2.waitKey(0)
# plt.close()


















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