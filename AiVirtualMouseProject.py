import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

############################
wCam, hCam = 680, 450
frameR = 100 #Frame Reduction
smoothinging = 7
#############################

pTime = 0
plockX, plockY = 0,0
clocX, clocY = 0,0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScreen, hScreen = autopy.screen.size()
#print(wScreen, hScreen)

while True: 
	#1. Find hand Landmarks
	ret, frame = cap.read()
	frame = detector.findHands(frame)
	lmList, bbox = detector.findPosition(frame)

	
	# 2. Get the Tip of the index and middle fingers
	if len(lmList) != 0:
		x1, y1 = lmList[8][1:]
		x2, y2 = lmList[12][1:]


	#3. Check which fingers are up
	fingers = detector.fingersUp()
	cv2.rectangle(frame, (frameR, frameR), (wCam -frameR, hCam - frameR), (255,0,255), 2)

	#4. Only Index Finger: Moving Mode
	if fingers[1] == 1 and fingers[2]==0:
		#5. Convert Coordinations 
		x3 = np.interp(x1, (frameR, wCam- frameR),(0,wScreen))
		y3 = np.interp(y1, (frameR, hCam- frameR),(0,hScreen))


		#6. Smoothen Values
		clocX = plockX + (x3 -plockX) / smoothinging
		clocY = plockY + (y3 -plockY) / smoothinging
		

		#7. Move Mouse
		autopy.mouse.move(wScreen-clocX, clocX)
		cv2.circle(frame, (x1, y1), 15, (255,0,255), -1)
		plockX, plockY = clocX, clocY


	#8.Both Index and middel fingers are up: clicking mode
	if fingers[1] == 1 and fingers[2]==1:
		#9. Finding the distance between fingers
		length, frame, info_line = detector.findDistance(8, 12, frame)
		print(length)

		#10.Click mouse if distance short
		if length < 55:
			cv2.circle(frame, (info_line[4], info_line[5]), 15, (0,255,0), -1)
		autopy.mouse.click()
	


	#11. Frame rate
	cTime = time.time()
	fps= 1/(cTime - pTime)
	pTime = cTime
	cv2.putText(frame, str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3, (255,0,0), 3)


	#12. Display
	cv2.imshow('Image', frame)
	if cv2.waitKey(10) & 0xFF == ord('q') or 0xFF == ord('Q'):
        	break	

