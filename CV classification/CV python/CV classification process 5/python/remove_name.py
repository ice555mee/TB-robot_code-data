import fitz # PyMuPDF
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2  
import os
import tensorflow as tf
import PIL
import time


path = "C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV\CV\% confident_Pseudocapacitor/50"
files = os.listdir(path)
for f in files:
    f2 = f[2:]
    print(f2)
    
    test_path = f"{path}\{f}"
    imgresize = cv2.imread(test_path)
    cv2.imwrite(f"C:\work\Konthee\VISTEC\ML job\Sudo battery\MLClasiifiy\CV\CV\All_Pseudocapacitor/{f2}",imgresize)
    os.remove(f"{path}\{f}") 
    
