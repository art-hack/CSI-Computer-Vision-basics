import cv2
import numpy as np 

from collections import deque

pts = deque()
cap = cv2.VideoCapture(0)

# Info : ZOOM in the frame to get the RGB value of the corresponding pixel

# The range of color for the object to be detected in HSV
hsv_supremum = np.array([172, 221, 255])
hsv_infinum = np.array([150, 40, 130])


def operate(frame):
	frame_copy = frame.copy()
	frame_ret = frame.copy()

	# Converting the frame from BGR format to HSV format, Hue Saturation Value GOOGle for more
	frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Thresholding the frame to get our object(color) tracked
	mask = cv2.inRange(frame_copy, hsv_infinum, hsv_supremum)
	mask = cv2.medianBlur(mask, 5)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# cv2.imshow('mask', mask)

	# Obtaining the binary Image for the corresponding mask
	res = cv2.bitwise_and(frame_ret,frame_ret, mask= mask)

	#cv2.imshow('res', res)
	# The result can be improved by smoothening the frame

	mask_copy = mask.copy()
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	if len(cnts) > 0:
		c = max(cnts, key = cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)

		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		if radius > 10:
			cv2.circle(frame_ret, (int(x), int(y)), int(radius), (0, 150, 255), 2)
			cv2.circle(frame_ret, center, 5, (0, 0, 255), -1)

	pts.appendleft(center)
	#print(pts[0])

	for i in range(1 , len(pts)):
		if pts[i-1] is None or pts[i] is None:
			continue
		thickness = 6
		cv2.line(frame_ret, pts[i-1], pts[i], (0, 0, 255), thickness)


	return frame_ret


while True:
	# Let's capture/read the frames for the video
	_, frame = cap.read() 
	key = cv2.waitKey(1) & 0xFF

	if key == ord("s"):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		initBB = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)
		#returns rect(x, y, width, height)
		crop_img = frame[initBB[1]:initBB[3]+initBB[1], initBB[0]:initBB[0]+initBB[2]]
		crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
		start = np.mean(crop_img, axis=(0, 1))
		hsv_supremum = np.array([start[0]*1.3, start[1]*1.3, start[2]*1.3])
		hsv_infinum = np.array([start[0]*0.7, start[1]*0.7, start[2]*0.7])
		# cv2.imshow("cropped", crop_img)
		# print(start)


	# This frame captured is flipped, Lets's get the mirror effect
	frame = cv2.flip(frame, 1)

	# Any operations on the frame captured will be processed in the operate(frame) method
	final_frame = operate(frame)

	# Showing the captured frame
	cv2.imshow('garrix', frame)
	cv2.imshow('martin', final_frame)

	if cv2.waitKey(1) & 0xff == 27:
		break


cap.release()
cv2.destroyAllWindows()