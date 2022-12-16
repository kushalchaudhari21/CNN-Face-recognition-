import numpy as np
import cv2
import os
import glob
import dlib
from tqdm import tqdm


#This file does some preprocessing on the dataset using opencv 
#Essentially, in every celebrity folder we create a resizedgray folder that contains only about 100 images of that celebrity resized and grayscaled.

#path to folder
path = "./10classpins/"

d = 1

#labels = folders of celebrities
labels = sorted(glob.glob(path+"*/", recursive = True))
for label in labels:
    images = sorted(glob.glob(label+'*.jpg'))
    os.mkdir(label+'resizedgray')

    for i in tqdm(range(len(images))):
        img = cv2.imread(images[i])
        resized = cv2.resize(img, (35,35), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        filename = label + 'resizedgray/' +  f'{d:03d}.jpg'
        cv2.imwrite(filename,gray)
        d+=1
        if d>100:
            break
    d=0

cv2.destroyAllWindows()

