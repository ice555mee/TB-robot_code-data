from operator import mod
import os
from tkinter import Y
from unittest import result
from pdfrw import PdfReader
import shutil
import fitz # PyMuPDF
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2  
import os

A= []

path = r'montree_bibliographie_21 fevrier 2021'
files = os.listdir(path)
Result=[]
i=0
for j in range(0,len(files)) :
    z= (files[j])
    if ( not z[-4:] != '.pdf'):
        continue
    i+=1
    newfolder = os.path.join(path, z)
    #print(i," ",z)
    #list file in folder
    Total,S_p=0,0
    files2 = os.listdir(newfolder)
    for k in range(0,len(files2)) :
        y=  (files2[k])  
        if y[-4:] == '.pdf' :
            print((y[:-4]).replace('-', '/',1)) 
            A.append((y[:-4]).replace('-', '/',1))
            newName = PdfReader(f"{newfolder}/{y}").Info.Title

        if ( not y[:7] != 'picture') or ( not y[-4:] != '.pdf'):
            continue 

        Total+=1 
        if y[0] == "1" :
            S_p+=1 
        elif y[0] == "0" :
            S_p+=0
    
    
    if "bat" in newName or "Bat" in newName:
        if Total != 0 :
            if S_p/Total <0.5 :
                Result.append(0)
            elif S_p/Total > 0.5 : 
                Result.append(1)
            elif S_p/Total == 0.5 :
                Result.append(0.5)
            pass
    elif "cap" in newName:
        if Total != 0 :
            if S_p/Total <0.5 :
                Result.append(1)
            elif S_p/Total > 0.5 : 
                Result.append(0)
            elif S_p/Total == 0.5 :
                Result.append(0.5)
            pass

print(Result)
print("Number of papar =",len(Result))
print("Number of Pseudocapacitor = "  ,Result.count(1) )
print("Number of confused  = "  ,Result.count(0.5) )
print("Number of Battery = "  ,Result.count(0))
print("Accuracy = "  ,Result.count(1)/len(Result))   



labels = 'Non-correlated', 'confused', 'Correlated'
sizes = [Result.count(1)/len(Result),Result.count(0.5)/len(Result),Result.count(0)/len(Result)]
explode = (0.1, 0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')


fig1, ax1 = plt.subplots()


ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',colors=("red","blue","green"),
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax1.set_title(f"Prediction of Word in Title of Montree Paper ({len(Result)})", color= "blue",alpha= 0.6)
 
plt.show()


'''
import csv
with open('TEST2.CSV', 'w',newline='') as f:
    writer = csv.writer(f)
    # write the header
    for word in A:
        writer.writerow([word])
'''