import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pandas as pd
from sign_model import SignNN
import torch
from torchvision import datasets, transforms
from PIL import Image
from scipy import stats

class handDetector():
	def __init__(self, mode=False, maxHands=2, detectionCon = 0.6, trackCon = 0.6):
		self.mode = mode
		self.maxHands = maxHands
		self.detectionCon = detectionCon
		self.trackCon = trackCon

		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.detectionCon, self.trackCon)

		self.mpDraw = mp.solutions.drawing_utils
		self.tipIds = [4, 8, 12, 16, 20]


	def findHands(self, img, draw= False):
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.hands.process(imgRGB)

		if self.results.multi_hand_landmarks:
			for handLms in self.results.multi_hand_landmarks:
				if draw:
					self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

		return img

	def findPosition(self, img, handNo=0, draw = False):
		xList=[]
		yList=[]
		bbox=[]
		hands_region = img.copy()
		self.lmList=[]
		if self.results.multi_hand_landmarks:
			myHand = self.results.multi_hand_landmarks[handNo]
			for id, lm in enumerate(myHand.landmark):
				h, w, c = img.shape
				cx, cy = int(lm.x * w), int(lm.y * h)
				xList.append(cx)
				yList.append(cy)

				self.lmList.append([id, cx, cy])
				if draw:
					cv2.circle(img, (cx, cy),5,(255,0,255), -1)

			xmin, xmax= min(xList), max(xList)
			ymin, ymax = min(yList), max(yList)

			hands_region = img[ymin-20: ymax+20,xmin-20:xmax+20]

			if draw:
				cv2.rectangle(img, (xmin -20, ymin-20),(xmax +20, ymax+20),(0,255,255), 2)


		return self.lmList, bbox, hands_region

################################################################################
### For finding and processing the hand
	def process_hand_img(self,img_arr):
		pil_img = Image.fromarray(img_arr)
		test_transforms = transforms.Compose([
									transforms.Grayscale(),
									transforms.Resize((28,28)),
                                    transforms.ToTensor(),
									transforms.Normalize((0.5,), (0.5,)),
                                      ])
		image_tensor = test_transforms(pil_img).float()
		image_tensor = torch.flatten(image_tensor).reshape((1,1,28,28))
		return image_tensor


	def main():
		pTime = 0
		cTime = 0
		cap = cv2.VideoCapture(0)
		detector = handDetector()
		model = torch.load("nn2_adam_lr001_ep10.pth")
		current_pred = None
		avg_pred_tracker = []

		while True:
			ret, frame = cap.read()
################################################################################
### For finding and processing the hand
			hands_frame = detector.findHands(frame)
			lmList, bbox, hands_only = detector.findPosition(hands_frame)
			hands_processed = detector.process_hand_img(hands_only)
			#print(type(hands_processed))

################################################################################
### For predicting the image of the hand
			model.eval()
			with torch.no_grad():
				max_vals, max_indices = model(hands_processed).max(1)
				print(max_indices[0])
				avg_pred_tracker.append(max_indices[0].item())
				if len(avg_pred_tracker)==30:
					prediction = stats.mode(avg_pred_tracker)[0]
					current_pred =  prediction
					print(avg_pred_tracker)
					avg_pred_tracker = []
			model.train()

################################################################################
### For outputting to the screen
			cv2.putText(frame, str(current_pred), (10,70),cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
			cv2.imshow('hands_frame', frame)
			cv2.imshow('hands_only', hands_only)
			#cv2.imshow('processed_hands', hands_processed.numpy())

################################################################################
### For breaking out of video loop use 'q' or 'Q'
			if cv2.waitKey(1) & 0xFF == ord('q') or 0xFF == ord('Q'):
            			break



if __name__ == '__main__':
	handDetector.main()
