
import fitz # PyMuPDF
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2  
import os
#0_0_0.66_10.106_1_9_3.101

path = "Paperall"
files = os.listdir(path)

for f in files:
    if "10.10" in f and "3.101"in f :
        print(f)




