
import fitz # PyMuPDF
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2  
import os


path = "D:\work\Konthee\VESTEC\ML job\Sudo battery\TestMLProgram\PaperwithoutCopy"
files = os.listdir(path)
#N 806
j,N=101,150
for j in range(j,N):
    z= (files[j])
    print(z)
    local_O  = f"Paper/{files[j]}"
    pdf_file = fitz.open(local_O) 
    # iterate over PDF pages
    for page_index in range(len(pdf_file)):
        # get the page itself
        page = pdf_file[page_index]
        image_list = page.getImageList() 
        for image_index, img in enumerate(page.getImageList(), start=1):
            # get the XREF of the image
            xref = img[0]
            # extract the image bytes
            base_image = pdf_file.extractImage(xref)
            image_bytes = base_image["image"]
            # get the image extension
            image_ext = base_image["ext"]
            # load it to PIL
            image = Image.open(io.BytesIO(image_bytes))
            # save it to local disk
            local  = f"PictureAll/0_0_{z[:5]}{page_index+1}_{image_index}_{z[-9:-4]}.{image_ext}"
            image.save(open(local, "wb"))
            imgresize = cv2.imread(local)
            imgresize2 = cv2.imread(local)
            # Grayscale
            gray = cv2.cvtColor(imgresize, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray,(15,15),0)
            thresh=cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,1)
            kernel=np.ones((3,3),np.uint8)
            closing=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=9)
            result_img=closing.copy()
            contours,hierachy=cv2.findContours(result_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            i=0
            for cnt in contours:
                i=i+1
                area = cv2.contourArea(cnt)
            
                if area > 20000: 
                
                    (x,y,w,h)= cv2.boundingRect(cnt)
                    cv2.rectangle(imgresize,(x,y),(x+w,y+h),(0,0,255),2)
                    result = imgresize2[y:y+h,x:x+w]
                    cv2.imwrite(f"PictureExt/0_0_{z[:5]}{page_index+1}_{image_index}_{(i)}_{z[-9:-4]}.{image_ext}",result)
                    
                    #cv2.imshow("show",result) 
                    #cv2.waitKey(0)
                

print(j+1)