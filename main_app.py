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

cv2.__version__

class SignHangMan():
	def __init__(self, tlx=50, tly=100, brx=250, bry=300):
		# HAND REGION OF INTEREST
		self.TL_x = tlx
		self.TL_y = tly
		self.BR_x = brx
		self.BR_y = bry
		self.label_encoder = {i:l for i,l in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
		self.letter_bank = {l:False for l in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
		self.phrase_bank = [
						"Go fast alone or far together",
						"Eat sleep rave repeat",
						"Don't eat yellow snow",
							]
		self.current_phrase = None


	def get_random_from_phrase_bank(self):
		rand_idx = np.random.randint(low=0, high=len(self.phrase_bank))
		return self.phrase_bank[rand_idx]

	def get_roi(self, frame):
		print(f"func get_roi {type(frame)}")
		roi = frame[self.TL_y:self.BR_y, self.TL_x:self.BR_x]
		return roi

	def get_hand_image(self, frame):
		roi = self.get_roi(frame)
		print(f"THIS ROI {type(roi)}")
		pil_img = Image.fromarray(roi)
		test_transforms = transforms.Compose(
									[
									transforms.Grayscale(),
									transforms.Resize((28,28)),
                                    transforms.ToTensor(),
									transforms.Normalize((0.5,), (0.5,)),
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
		phrase_xspace = 640 // len(self.current_phrase)

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
### For drawing the image interface
	def make_hangman_display_gui(self, frame):
		gui = np.ones((600,200,3),dtype='uint8') *255
		roi = self.get_roi(frame)
		roi_resize = cv2.resize(roi, (200,200))
		print(f"hangman_display_gui roi_resize {type(roi)}")
		gui[0:200,0:200,:] = roi_resize
		gui[-400:-200,-200:,:] = roi_resize
		return gui

################################################################################
### Main loop for capturing video
	def main_loop(self):
		cap = cv2.VideoCapture(0)
		model = torch.load("oldmodels/nn2_adam_lr001_ep10.pth")
		current_pred = None
		avg_pred_tracker = []
		while True:
			ret, frame = cap.read()
			og_frame = frame.copy()
################################################################################
### For finding hand and concatenating the interface
			hands_processed = self.get_hand_image(frame)
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
					current_pred = prediction
					avg_pred_tracker = []
					print(current_pred)
			model.train()

################################################################################
### For outputting to the screen
			cv2.imshow('concat', og_plus_plus)

################################################################################
### For breaking out of video loop use 'q' or 'Q'
			key_input = cv2.waitKey(1) & 0xFF
			if key_input == 27:
				break
		cap.release()
		cv2.destroyAllWindows()



if __name__ == '__main__':
	x = SignHangMan(tlx=5, tly=100, brx=205, bry=300)
	x.main_loop()
