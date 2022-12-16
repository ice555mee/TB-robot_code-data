import cv2
from cv2 import log
from cv2 import sqrt 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from matplotlib.widgets import Slider, Button
import matplotlib.ticker
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
from matplotlib.pyplot import cm, hsv
from matplotlib.colors import hsv_to_rgb
import matplotlib as mpl
from cycler import cycler
import os
import math


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap



#parameter 
R = 8.314
F = 96500
T = 273 + 25
n = 1
E = 0.5
tl = 1
x,y=[],[]
k=1800

#color
cmap = plt.get_cmap('hsv')
new_cmap = truncate_colormap(cmap, 0.1, 0.9)
cmap2 = plt.get_cmap('hsv')
new_cmap2 = truncate_colormap(cmap, 0.1, 0.5)

parameterToColorBy = np.linspace(0.5, 1, 1800, dtype=float)
colors = [new_cmap(i)
          for i in np.linspace(0, 1, parameterToColorBy.shape[0])]
colors2 = [new_cmap2(i)
          for i in np.linspace(0, 1, parameterToColorBy.shape[0])]



#size
fig, (ax) = plt.subplots(nrows=3, ncols=1, sharex=True,
                                    figsize=(18,9 ))

#plot a
ax1 = plt.subplot(1,  2, 1)
ax1.set_xlabel("Time",fontsize=16,weight='bold',fontname='Times New Roman')
ax1.set_ylabel("Potential",fontsize=16,weight='bold',fontname='Times New Roman')
ax1.text(-0.13,0.78,"a",fontsize=18,weight='bold',fontname='Times New Roman')
plt.axis([0, 1, 0, 0.8])
for t in ax1.get_yticklabels():
    t.set_fontsize(14)
    t.set_fontname('Arial')
for t in ax1.get_xticklabels():
    t.set_fontsize(14)
    t.set_fontname('Arial')

m=1.6
for M,c in zip(range(k),colors) :
    x,y=[],[] 
    m+=0.01
    for t in range(1,100,1):
        x.append((-t/100)+1)
        p=m*R*T/(n*F)*math.log((math.sqrt(tl)-math.sqrt(t/100))/(math.sqrt(t/100)))+E
        y.append(p)
    if m> 6.3 :
        ax1.plot(x,y,color=colors[300],linewidth=3)
    elif m==6.3 :
        ax1.plot(x,y,color=c,linewidth=3)
    else :
        ax1.plot(x,y,color=c,linewidth=3)

#plot b
ax2 = plt.subplot(1,  2, 2) 
ax2.set_xlabel("Time",fontsize=16,weight='bold',fontname='Times New Roman')
ax2.set_ylabel("Potential",fontsize=16,weight='bold',fontname='Times New Roman')
ax2.text(-0.13,0.78,"(b)",fontsize=18,weight='bold',fontname='Times New Roman')
plt.axis([0, 1, 0, 0.8])  
for t in ax2.get_yticklabels():
    t.set_fontsize(14)
    t.set_fontname('Arial')
for t in ax2.get_xticklabels():
    t.set_fontsize(14)
    t.set_fontname('Arial')




m=1.6
for M,c in zip(range(k),colors2) :
    x,y=[],[] 
    m+=0.01
    
    for t in range(1,100,1):
        x.append((t/100))
        p=m*R*T/(n*F)*math.log((math.sqrt(tl)-math.sqrt(t/100))/(math.sqrt(t/100)))+E
        y.append(p)
    if m> 9.1 :
        ax2.plot(x,y,color=c,linewidth=3)
    elif m==9.1 :
        ax2.plot(x,y,c,linewidth=3)
    else :
        ax2.plot(x,y,color=c,linewidth=3)


norm = mpl.colors.Normalize(vmin=0.5, vmax=1)

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=new_cmap),
             ax=ax1, orientation='horizontal', label='Some Units')
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=new_cmap2),
             ax=ax2, orientation='horizontal', label='Some Units')


plt.show()
           