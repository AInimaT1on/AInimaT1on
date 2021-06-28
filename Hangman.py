from kivymd.app import MDApp 
from kivy.lang import Builder
from kivy.core.window import  Window
from kivy.uix.image import Image
from kivymd.uix.boxlayout import MDBoxLayout
import os
import cv2





class HangmanApp(MDApp):
	def build(self):
		self.theme_cls.theme_style='Light'
		self.theme_cls.primary_palette = 'DeepPurple'
		self.capture = cv2.VideoCapture(0)
		return Builder.load_file('hangman.kv')

	def load_video(self, *args):
		while self.capture.isOpened():
			ret, frame = self.capture.read()
			self.image_frame = frame
			#Frame initulize
			cv2.imshow('Playground', frame)
			if cv2.waitKey(30) & 0xFF == ord('q')or 0xFF == ord('Q'):
        			break

		self.capture.release()
		cv2.destroyAllWindows()
	
	def load_file(self):
		os.system('python main_app.py')
		
		


if __name__ == '__main__':
	HangmanApp().run()