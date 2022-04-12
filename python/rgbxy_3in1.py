#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:21:28 2022

@author: Kim Miikki 2022
"""
import argparse
import csv
import cv2        
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

'''
Name template:
img_name='05.png'
rgb_iratios='05-rgbratios.csv'

Generate:
out_file="05-rgbratios-img.png"
'''

# Start parser for arguments
parser = argparse.ArgumentParser()

# Positional arguments
parser.add_argument('img_name', type=str, help='RAT window image filename')
parser.add_argument('csv_name', type=str, help='RAT window RGB internal ratios CSV filename')
parser.add_argument('-f',type=int,help='font size (default=8)',const=8,required=False,nargs='?')
args = parser.parse_args()

img_name=args.img_name
rgb_iratios=args.csv_name

if args.f != None:
    fsize=args.f
else:
    fsize=8

out_file=Path(img_name).stem+'-img.png'

# x-axis label
xaxis="X coordinate"

# y-axis label
yaxis="Y coordinate"

# secondary y-axis label
yaxis_label_r_g="RGB, R/G, R/B"

print("RGBXY 3 in 1, (c) Kim Miikki 2022")

rgb_data=[]
hlen=0
try:
    with open(rgb_iratios,"r") as reader_obj:
        csv_reader=csv.reader(reader_obj)
        header_rgb=next(csv_reader)
        hlen=len(header_rgb)
                
        for row in csv_reader:
            rgb_data.append(list(map(float,row)))

except OSError:
    curdir=os.getcwd()
    print("Unable to open "+rgb_iratios+" in following directory:\n"+curdir)
    print("Program is terminated.")
    print("")
    sys.exit(1)


# Template
# ['Column (X) coordinate','R','G','B','BW','RGmean','RBmean','GBmean',
# 'R/G', 'R/B', 'G/R', 'G/B', 'B/R', 'B/G', 'R/GB', 'G/RB', 'B/RG']

np.set_printoptions(suppress=True)
datalen=len(rgb_data)
arr1=np.array(rgb_data)
arr1=arr1.T

xs=arr1[0]

r=arr1[1]
g=arr1[2]
b=arr1[3]

maxv=max([arr1[8].max(),arr1[9].max()])
minv=min([arr1[8].min(),arr1[9].min()])
dv=maxv-minv

# Normalize
# rg_norm = (rg-minv)*255/(maxv-minv)
# rg_norm = (rg-minv)*255/dv
rg=(arr1[8]-minv)*255/dv
rb=(arr1[9]-minv)*255/dv

img=cv2.imread(img_name)
h,w,ch=img.shape

width=10
height=width*h/w

# Set figure size
fig, ax1 = plt.subplots(figsize = (width,height))

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

#ax1 = plt.gca()
ax2 = ax1.twinx()

# Set axes range
ax1.set_xlim(0,w)
ax1.set_ylim(h,0)
ax2.set_ylim(0,260)

ax1.set_xlabel(xaxis)
ax1.set_ylabel(yaxis)
ax2.set_ylabel(yaxis_label_r_g)

# Plot RGB data on the second axis
ax2.plot(xs,r,color="r",label='red mean')
ax2.plot(xs,g,color="g",label='green mean')
ax2.plot(xs,b,color="b",label='blue mean')

# Plot RGB ratios on the second axis
ax2.plot(xs,rg,color="g",linestyle='dashed',label='R/G normalized')
ax2.plot(xs,rb,color="b",linestyle='dashed',label='R/B normalized')

# Legend 
legend_outside = plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
plt.legend(frameon=False, framealpha=0.5, shadow=True, borderpad=1)
# Set the legend font size
plt.rc('legend', fontsize=fsize)

plt.show()

fig.savefig(out_file,dpi=300, bbox_inches='tight')