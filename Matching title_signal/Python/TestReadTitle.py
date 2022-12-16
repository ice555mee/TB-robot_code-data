
from operator import mod
import os
from pdfrw import PdfReader
import shutil
import fitz # PyMuPDF

path = r'C:\work\Konthee\VISTEC\ML job\Sudo battery\TestMLProgram\Paperall'
path_s = r'C:\work\Konthee\VISTEC\ML job\Sudo battery\TestMLProgram\pdf(battery)'


i=0

for fileName in os.listdir(path):
    # Rename only pdf files
    i+=1
    fullName = os.path.join(path, fileName)

    newfounder = os.path.join(path_s, fileName)

    if (not os.path.isfile(fullName) or fileName[-4:] != '.pdf'):
        continue
    
    fullName = os.path.join(path, fileName)
    # Extract pdf title from pdf file
    try :
        newName = PdfReader(fullName).Info.Title

        if "battery" in newName:
            newName = newName.strip('()')+'.pdf'
            print(i," is ",newName)
            #newName = newName.replace('a', '')
            newFullName = os.path.join(newfounder, fileName)
            shutil.copyfile(fullName, newfounder)

    except  :
        continue

    


    # Remove surrounding brackets that some pdf titles have
    
    