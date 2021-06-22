from kivymd.app import MDApp 
from kivy.lang import Builder
from kivy.core.window import  Window
import cv2





class HangmanApp(MDApp):
	def build(self):
		self.theme_cls.theme_style='Light'
		self.theme_cls.primary_palette = 'DeepPurple'
		self.capture = cv2.VideoCapture(0)
		return Builder.load_file('hangman.kv')

	def load_video(self, *args):
		ret, frame = self.capture.read()
		#Frame initulize
		self.image_frame = frame


if __name__ == '__main__':
	HangmanApp().run()