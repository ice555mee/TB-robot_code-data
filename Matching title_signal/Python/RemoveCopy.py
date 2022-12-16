import fitz # PyMuPDF
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2  
import os


path = "D:\work\Konthee\VESTEC\ML job\Sudo battery\TestMLProgram\Paper"
files = os.listdir(path)

print(files[5])
i,j=0,0
for f in files:
    x = files[i]
    
    if x[-9:] == "Copie.pdf" :
        print(i)
    else :
        print(f)
        j=j+1
         # save it to local disk
        local_O  = f"Paper/{f}"
        local_S  = f"PaperwithoutCopy/{f}"
        pdf_file = fitz.open(local_O) 
        if len(pdf_file) ==0:
            print(i)
        else:
            pdf_file.save(local_S)



    i=i+1

print(j)


'''
# file path you want to extract images from
#"Battery/Battery_ZnLFP.pdf"
#"Pseudocapacitor/Pseudo_Ionic LiquidsIonic.pdf"
file = "Pseudocapacitor/Pseudo_Ionic LiquidsIonic.pdf"
# open the file
pdf_file = fitz.open(file)
print(file[:4])
# iterate over PDF pages
for page_index in range(len(pdf_file)):
    # get the page itself
    page = pdf_file[page_index]
    image_list = page.getImageList()
    #print(image_list)

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
        #print(image)
        # save it to local disk
        
        local  = f"Picture/{file[:4]}{page_index+1}_{image_index}_{file[-9:-4]}.{image_ext}"
        image.save(open(local, "wb"))
        imgresize = cv2.imread(local)
        imgresize2 = cv2.imread(local)
        # Grayscale
        gray = cv2.cvtColor(imgresize, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray,(15,15),0)
        thresh=cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,1)
        kernel=np.ones((3,3),np.uint8)
        closing=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=12)
        result_img=closing.copy()
        contours,hierachy=cv2.findContours(result_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        i=0
        for cnt in contours:
            i=i+1
            area = cv2.contourArea(cnt)
            
            if area > 50000: 
                
                (x,y,w,h)= cv2.boundingRect(cnt)
                cv2.rectangle(imgresize,(x,y),(x+w,y+h),(0,0,255),2)
                result = imgresize2[y:y+h,x:x+w]
                cv2.imwrite(f"Picture2/{file[:4]}{page_index+1}_{image_index}_{i}_{file[-9:-4]}.{image_ext}",result)
                #cv2.imshow("show",result) 
                #cv2.waitKey(0)
                


        #cv2.imshow("show",imgresize) 
        #cv2.waitKey(0)

  
  
'''
'''
cv2.imshow('Contours', gray)
cv2.imshow('test', gray_blur)
cv2.imshow('thresh', thresh)
cv2.imshow('closing', closing)
'''
'''
cv2.waitKey(0)
cv2.destroyAllWindows()


'''

