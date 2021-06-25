import random
import os
from PIL import Image

PATH = '../sign_datav2/Test'
PATH_OUT = '../sign_datav2/random_test_100'
os.mkdir(PATH_OUT)
folders = os.listdir(PATH)

for folder in folders:
    i=0
    os.mkdir(PATH_OUT+'/'+folder)
    img_names = os.listdir(PATH+'/'+folder)
    print(folder)
    for file in img_names:
        while i<100:
            rand = random.randint(1,1000)
            img = Image.open(PATH+'/'+folder+'/'+img_names[i])
            img.save(PATH_OUT+'/'+folder+'/'+img_names[i])
            i+=1
