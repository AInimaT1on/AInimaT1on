import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pandas as pd
from sign_model import SignNN, SignNN2
import torch
from torchvision import datasets, transforms
from PIL import Image
from scipy import stats
from helper_functions import number_to_letter
from collections import OrderedDict

cv2.__version__

class SignHangMan():
	def __init__(self, tlx=50, tly=100, brx=250, bry=300):
		# HAND REGION OF INTEREST
		self.TL_x = tlx
		self.TL_y = tly
		self.BR_x = brx
		self.BR_y = bry
		self.label_encoder = {i:l for i,l in enumerate("ABCDEFGHIGJKLMNOPQRSTUVWXYZ")}
		self.letter_bank = {l:False for l in "ABCDEFGHIGJKLMNOPQRSTUVWXYZ"}
		self.phrase_bank = [
							"why yolo",
							"lol why",
							"ooo lol",
							"hoy hoy",
							"woo woo",
							]
		self.hangman_display = [
						("floor",),
						("pillar",),
						("hang_bar",),
						("lower_strut",),
						("upper_strut",),
						("rope",),
						("hangman",),
						]
		self.current_phrase = None
		self.current_lives = len(self.hangman_display)
		self.current_pred = None
		self.current_example = "A"
		self.current_thr1 = 100
		self.current_thr2 = 200
		self.current_cntQnt = 1
		self.wt1 = "wt1"
		# self.wt2 = "wt2"
		# self.wt3 = "wt3"
		# self.wt4 = "wt4"
		#
		# cv2.namedWindow(self.wt1, cv2.WINDOW_AUTOSIZE)
		# cv2.namedWindow(self.wt2, cv2.WINDOW_AUTOSIZE)
		# cv2.namedWindow(self.wt3, cv2.WINDOW_AUTOSIZE)
		# cv2.namedWindow(self.wt4, cv2.WINDOW_AUTOSIZE)



	def get_random_from_phrase_bank(self):
		rand_idx = np.random.randint(low=0, high=len(self.phrase_bank))
		return self.phrase_bank[rand_idx]

	def get_roi(self, frame):
		roi = frame[self.TL_y:self.BR_y, self.TL_x:self.BR_x]
		return roi

	def get_contour_areas(contours):
	    all_areas = []
	    for contour in contours:
	        contour_area = cv2.contourArea(contour)
	        all_areas.append(contour_area)
	    return all_areas

	def get_hand_image(self, frame):
		test_img_list = []
		roi = self.get_roi(frame)
		pil_img = Image.fromarray(roi)

		# img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		# img_gray_sub = cv2.subtract(255,img_gray)
		# blur = cv2.GaussianBlur(img_gray,(5,5),1)
		# cnv = np.zeros(img_gray.shape, dtype='uint8')
		# canny_img = cv2.Canny(blur, self.current_thr1,self.current_thr2)
		# contours, h = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		# contours_by_size = sorted(contours, key=cv2.contourArea, reverse=True)
		# contour_img = cv2.drawContours(cnv, contours_by_size[0:5], -1, (255,255,255),thickness=3)
		# kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		# kernel2 = np.ones((3,3),np.uint8)
		# erosion = cv2.erode(contour_img,kernel2,iterations = 1)
		# dilation = cv2.dilate(contour_img,kernel1,iterations = 7)
		#
		#
		#
		# test_img_list.append(canny_img)
		# test_img_list.append(contour_img)
		# test_img_list.append(erosion)
		# test_img_list.append(dilation)
		# for img in test_img_list:
		# 	print(img.shape)

		#test_img_list.append(approx)

		# ret, thresh = cv2.threshold(img_gray,self.current_thr1,255,cv2.THRESH_TOZERO)

		test_transforms = transforms.Compose(
									[
									transforms.Grayscale(),
									transforms.Resize((28,28)),
                                    transforms.ToTensor(),
									transforms.Normalize((0.5,), (0.5,))
									#mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
									])
		image_tensor = test_transforms(pil_img).float()
		image_tensor = torch.flatten(image_tensor).reshape((1,1,28,28))

		return image_tensor


################################################################################
### For drawing the letter interface
	def make_hangman_letter_gui(self):
		gui = np.zeros((120,640,3),dtype='uint8')
		letter_yloc = 40
		letter_xspace = 640 // 26

		if self.current_phrase == None:
			self.current_phrase = self.get_random_from_phrase_bank()

		phrase_yloc = 80

		for i, (key, value) in enumerate(self.letter_bank.items(),1):
			color = (0,0,0)
			if value == False:
				color = (255,255,255)
			cv2.putText(gui, key, (i*letter_xspace-10, letter_yloc),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

		for i, l in enumerate(self.current_phrase,1):
			color = (255,255,255)
			print_letter = "_"
			if l.isalpha():
				if self.letter_bank[l.upper()] == True:
					print_letter = l
				else:
					print_letter = "_"
			else:
				print_letter = " "
			cv2.putText(gui, print_letter, (i*20, phrase_yloc),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

		return gui

################################################################################
### For submitting a letter
	def submit_letter(self, letter_code):
		letter = self.label_encoder[letter_code]
		print(f"current letter submitted :{letter}")
		is_over = False
		if self.letter_bank[letter] == False:
			self.letter_bank[letter] = True
			if letter in self.current_phrase:
				print("Success you found a letter")
			else:
				self.current_lives -=1
				is_over = self.game_over_check()
				print(f" IS_OVER:::::::::: {is_over}")
		else:
			print("You've already selected that letter")

		if is_over:
			self.reset_game()

################################################################################
### For checking if a game is over
	def game_over_check(self):
		letter_checks = []

		print(f"Current Lives: {self.current_lives}")
		print(f"Current Lives: {self.current_phrase}")
		if self.current_lives <= 0:
			print("Ended because of lives ran out")
			return True

		else:
			for l in self.current_phrase:
				if l.isalpha():
					if self.letter_bank[l.upper()] == True:
						letter_checks.append(True)
					else:
						letter_checks.append(False)
		print(letter_checks)
		if False in letter_checks:
			return False
		else:
			print("Victory")
			return True

################################################################################
### For checking if a game is over
	def reset_game(self):
		self.letter_bank = {l:False for l in "HLOWY"}
		self.current_phrase = None
		self.current_lives = len(self.hangman_display)


################################################################################
### For drawing the image interface
	def make_hangman_display_gui(self, frame):
		gui = np.ones((600,200,3),dtype='uint8') *255
		roi = self.get_roi(frame)
		roi_resize = cv2.resize(roi, (200,200))
		example = cv2.imread(f"data/data_collector/examples/{self.current_example}.jpg")
		gui[0:200,0:200,:] = roi_resize
		gui[200:400,:200,:] = example
		if self.current_pred != None:
			letter = self.label_encoder[self.current_pred]
			cv2.putText(gui, letter, (100,500),cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
		return gui

	def change_threshold1_value(self, value):
		self.current_thr1 = value

	def change_threshold2_value(self, value):
		self.current_thr2 = value

	def change_conotour_qnt_value(self, value):
		self.current_cntQnt = value
################################################################################
### Main loop for capturing video
	def main_loop(self):
		# cv2.createTrackbar("Change threshold 1 value", self.wt1, self.current_thr1, 255, self.change_threshold1_value)
		# cv2.createTrackbar("Change threshold2 value", self.wt1, self.current_thr2, 255, self.change_threshold2_value)
		# cv2.createTrackbar("Change contour quantity", self.wt1, self.current_cntQnt , 20, self.change_conotour_qnt_value)
		# cv2.createTrackbar("Change contour quantity", self.wt1, self.current_cntQnt , 20, self.change_conotour_qnt_value)
		# cv2.createTrackbar("Change contour quantity", self.wt1, self.current_cntQnt , 20, self.change_conotour_qnt_value)
		# cv2.createTrackbar("Change contour quantity", self.wt1, self.current_cntQnt , 20, self.change_conotour_qnt_value)
		cap = cv2.VideoCapture(0)
		model = torch.load("oldmodels/double_train.pth")
		avg_pred_tracker = []
		while True:
			print(self.current_pred)
			ret, frame = cap.read()
			og_frame = frame.copy()
################################################################################
### For finding hand and concatenating the interface
			hands_processed  = self.get_hand_image(frame)
			#print(f"hands_procsessed: {hands_processed.shape}")
			gui_letters = self.make_hangman_letter_gui()
			gui_image = self.make_hangman_display_gui(frame)
			og_plus = cv2.vconcat((frame, gui_letters))
			og_plus_plus = cv2.hconcat((og_plus, gui_image))

################################################################################
### For predicting the image of the hand
			model.eval()
			with torch.no_grad():
				max_vals, max_indices = model(hands_processed).max(1)
				avg_pred_tracker.append(max_indices[0].item())
				if len(avg_pred_tracker)==30:
					prediction = stats.mode(avg_pred_tracker)[0]
					self.current_pred = prediction.item()
					print(prediction.item())
					avg_pred_tracker = []
			model.train()

################################################################################
### For outputting to the screen
			#test_vconcat = cv2.vconcat(test_img_list)
			#cv2.imshow(self.wt2, test_img_list[0])
			#cv2.imshow(self.wt1, test_vconcat)
			cv2.imshow('concat', og_plus_plus)

################################################################################
### For breaking out of video loop use 'q' or 'Q'
			key_input = cv2.waitKey(1) & 0xFF
			if key_input == 27:
				break
			elif key_input == ord(" "):
				self.submit_letter(self.current_pred)
			elif chr(key_input) in "ABCDEFGHIKLMNOPQRSTUVWXYabcdefghiklmnopqrstuvwxy":
				self.current_example = chr(key_input).upper()
			else:
				pass
		cap.release()
		cv2.destroyAllWindows()



if __name__ == '__main__':
	x = SignHangMan(tlx=5, tly=100, brx=205, bry=300)
	x.main_loop()
