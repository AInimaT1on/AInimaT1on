import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pandas as pd
from sign_model import SignNN
from helper_functions import save_sign_img
import torch
from torchvision import datasets, transforms
from PIL import Image
from scipy import stats
import os
from datetime import datetime as dt

CURRENT_LETTER = 'A'
RECORDING = 0
TL_x = 50
TL_y = 100
BR_x = 250
BR_y  = 300
cap = cv2.VideoCapture(0)
window_name = 'sign data capture'
counter = 1

TIME_START = dt.now().time()


while True:
    REC_TIME = dt.now().time() - TIME_START
    ret, frame = cap.read()
    og = frame.copy()
    cv2.rectangle(frame, (TL_x, TL_y), (BR_x, BR_y), (0,255,0), 2)
    ROI = frame[TL_y:BR_y, TL_x:BR_x]
    if RECORDING:
        cv2.putText(frame, f"Recording : {CURRENT_LETTER}, T:{str(REC_TIME)}", (10,70),cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
        # INSERT WRITE FUNCITON HERE
    else:
        cv2.putText(frame, f"Not Recording : {CURRENT_LETTER}", (10,70),cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
    cv2.imshow(window_name, frame)
    cv2.imshow("Hand Box", ROI)


    key = cv2.waitKey(1) & 0xFF

    if key == ord('a') or key == ord('A'):
        CURRENT_LETTER = 'A'
    elif key == ord('b') or key == ord('B'):
        CURRENT_LETTER = 'B'
    elif key == ord('c') or key == ord('C'):
        CURRENT_LETTER = 'C'
    elif key == ord('d') or key == ord('D'):
        CURRENT_LETTER = 'D'
    elif key == ord('e') or key == ord('E'):
        CURRENT_LETTER = 'E'
    elif key == ord('f') or key == ord('F'):
        CURRENT_LETTER = 'F'
    elif key == ord('g') or key == ord('G'):
        CURRENT_LETTER = 'G'
    elif key == ord('h') or key == ord('H'):
        CURRENT_LETTER = 'H'
    elif key == ord('i') or key == ord('I'):
        CURRENT_LETTER = 'I'
    elif key == ord('j') or key == ord('J'):
        CURRENT_LETTER = 'J'
    elif key == ord('k') or key == ord('K'):
        CURRENT_LETTER = 'K'
    elif key == ord('l') or key == ord('L'):
        CURRENT_LETTER = 'L'
    elif key == ord('m') or key == ord('M'):
        CURRENT_LETTER = 'M'
    elif key == ord('n') or key == ord('N'):
        CURRENT_LETTER = 'N'
    elif key == ord('o') or key == ord('O'):
        CURRENT_LETTER = 'O'
    elif key == ord('p') or key == ord('P'):
        CURRENT_LETTER = 'P'
    elif key == ord('q') or key == ord('Q'):
        CURRENT_LETTER = 'Q'
    elif key == ord('r') or key == ord('R'):
        CURRENT_LETTER = 'R'
    elif key == ord('s') or key == ord('S'):
        CURRENT_LETTER = 'S'
    elif key == ord('t') or key == ord('T'):
        CURRENT_LETTER = 'T'
    elif key == ord('u') or key == ord('U'):
        CURRENT_LETTER = 'U'
    elif key == ord('v') or key == ord('V'):
        CURRENT_LETTER = 'V'
    elif key == ord('w') or key == ord('W'):
        CURRENT_LETTER = 'W'
    elif key == ord('x') or key == ord('X'):
        CURRENT_LETTER = 'X'
    elif key == ord('y') or key == ord('Y'):
        CURRENT_LETTER = 'Y'
    elif key == ord('z') or key == ord('Z'):
        CURRENT_LETTER = 'Z'
    elif key == ord(' '):
        RECORDING = np.abs(RECORDING-1)
        TIME_START = dt.now().time()
        print(RECORDING)
    elif cv2.waitKey(1) & key == 27:
        break
    else:
        continue
cap.release()
cv2.destroyAllWindows()
