#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 10:11:33 2022

@author: Kim Miikki
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
import numpy as np
import cv2

np.set_printoptions(suppress=True)

xmargin=0
ymargin=0
distance=100
prominence=0.5
width=2
xtrim_max=400
xtrim_th=0.5
discardSideZones=True

extensions=['.png','.jpg','.bmp','.tif','.tiff','.dib','.jpeg','.jpe',',pbm','.pgm','.ppm','.sr','.ras']
exclude=['roi.ini']

stem=''
filename=''
minThreshold=1e-9
maxThreshold=255
outdir1='crop'
defaultExt=".png"
xm=0
ym=0
threshold=0
decimals=4

isInput=False
isROI=False
preserveExt=False 
isX=False
isY=False
isThreshold=False
isBatch=True
isFixDim=False

def roix(value):
    return round(value/img_x1,decimals)

def roiy(value):
    return round(value/img_y1,decimals)

parser=argparse.ArgumentParser()
parser.add_argument('-i',type=str,help='input file name',required=False)
parser.add_argument('-o',action='store_true',help='preserve original file format')
parser.add_argument('-x',type=float,help='maximum horizontal marginal as integer',required=False)
parser.add_argument('-y',type=float,help='maximum vertical marginal as integer',required=False)
parser.add_argument('-t',type=float,help='threshold value as float, '+str(minThreshold)+' < t <= '+str(maxThreshold))
parser.add_argument('-f',action='store_true',help='fix odd dimensions by adjusting crop marginals')
args = parser.parse_args()

if args.i != None:
    isInput=True
    isROI=True
    isBatch=False
    filename=args.i
    stem=Path(filename).stem

if args.o:
    preserveExt=True

if args.x != None:
    isX=True    
    tmp=int(args.x)
    if tmp<0:
        tmp=0
    xm=tmp   

if args.y != None:
    isY=True    
    tmp=int(args.y)
    if tmp<0:
        tmp=0
    ym=tmp

if args.t != None:
    isThreshold=True    
    tmp=float(args.t)
    if tmp<minThreshold:
        tmp=minThreshold
    elif tmp>maxThreshold:
        tmp=maxThreshold
    threshold=tmp

if args.f:
    isFixDim=True

print('Autocrop 1.0, (c) Kim Miikki 2022')
print('')
print('Current directory:')
curdir=os.getcwd()
print(curdir)
print('')
path=Path(curdir)
adir=str(path)+'/'+outdir1+'/'

try:
  if not os.path.exists(outdir1):
    os.mkdir(outdir1)
except OSError:
  print("Unable to create a directory under following path:\n"+curdir)
  print("Program is terminated.")
  print("")
  sys.exit(1)

# Generate a list of input files
files=[]
if isInput:
    files.append(filename)
else:
    for p in sorted(path.iterdir()):
        suffix=p.suffix.lower()
        if p.is_file() and suffix in extensions:
            fname=p.name
            if not fname in exclude:
                files.append(fname)

# Crop all images in the list
num=0
t1=datetime.now()                
if len(files)>0:
    print("Processing:")
for name in files:
    try:
        img=cv2.imread(name)
        h,w,ch=img.shape
    except:
        print(' - Unable to open: '+name)
        continue
    print(name+', size '+str(w)+'x'+str(h))
    
    x0=0
    x1=w-1
    y0=0
    y1=h-1
    
    # Analyze the image in y direction
    distance=250
    prominence=0.5
    width=1 
    
    # Find the line(s) upper and lower y values
    yrgb=img.mean(axis=1)
    yrgb=np.array(yrgb).T   
    yr=yrgb[2]
    yg=yrgb[1]
    yb=yrgb[0]
    yrg=yr-yg
    yrb=yr-yb
    ybw=img.mean(axis=1).mean(axis=1)

    # Find ybw_diff peaks
    ybw_diff=np.diff(ybw)
    ypeaks_diff,yproperties_diff=find_peaks(ybw_diff,distance=distance,prominence=prominence,width=width)
    
    # Calculate y zones r-g and r-b means
    yzones=len(ypeaks_diff)+1
    i=0
    last=0
    r_gb=(yrg+yrb)/2
    zones=len(ypeaks_diff)+1
    result=[]
    for element in ypeaks_diff:
        current=ypeaks_diff[i]
        if discardSideZones and i in [0,zones-1]:
            include=False
        else:
            include=True
        #print(last,current,r_gb[last:current].mean())
        if include:
            # result.append([r_gb[last:current].mean(),last,current])
            result.append([np.median(r_gb[last:current]),last,current])
        last=current
        i+=1
    
    # Crop the image to max(r_gb) zone
    if len(max(result))==3:
        yn0=max(result)[1]
        yn1=max(result)[2]+1
        # Crop the image in y direction
        img=img[yn0:yn1,:]
    
    # Analyze the image in x direction
    xrgb=img.mean(axis=0)
    xrgb=np.array(xrgb).T   
    xr=xrgb[2]
    xg=xrgb[1]
    xb=xrgb[0]
    xrg=xr-xg
    xrb=xr-xb
    xbw=img.mean(axis=0).mean(axis=1)
    
    # Find the red peak or peaks
    pr=5 # 5
    peaks_rg,rg_properties=find_peaks(xrb,distance=distance,prominence=pr,width=width)
    peaks_rb,rb_properties=find_peaks(xrg,distance=distance,prominence=pr,width=width)
    
    # Find maximum peak values and positions
    
    # R-G
    rg_max_pr=0
    rg_pos=-1
    i=0
    last=0
    while i<len(rg_properties['prominences']):
        p=round(rg_properties['prominences'][i],10)
        if p>last:
            last=p
            rg_max_pr=p
            rg_pos=i
        i+=1
    if rg_pos>=0:
        rg_width=rg_properties['widths'][rg_pos]
        
    # R-B
    rb_max_pr=0
    rb_pos=-1
    i=0
    last=0
    while i<len(rb_properties['prominences']):
        p=round(rb_properties['prominences'][i],10)
        if p>last:
            last=p
            rb_max_pr=p
            rb_pos=i
        i+=1
    if rb_pos>=0:
        rb_width=rb_properties['widths'][rb_pos]

    h,w,ch=img.shape

    # Calculate sums of R-G and R-B
    pos_mean=int((peaks_rg[rg_pos]+peaks_rb[rb_pos])/2)
    half_width=int((rg_width+rb_width)/4)
    sum_r_gb=xrg+xrb
    pr=0.5
    peaks,pps=find_peaks(xrg+xrb,distance=h,prominence=pr,width=width) # h is the minimum length to the next line or wall

    # Find minimum and maximum positions for rg and rb 
    len_rg=len(peaks_rg)
    len_rb=len(peaks_rb)
    a1=peaks_rg.min()
    a2=peaks_rb.min()
    b1=peaks_rg.max()
    b2=peaks_rb.max()


    xn0=-1
    xn1=-1    
    if len_rg>=2 and len_rb>=2:
        xn0=min([a1,a2])
        xn1=min([b1,b2])
    elif len_rg==1 and len_rb>=2:
        xn0=a1
    elif len_rg>=2 and len_rb==1:
        xn0=a2
    elif len_rg==1 and len_rb==1:
        xn0=a1

    if xn1==-1:
        xn1=xn0
    
    a=xn0
    b=xn1
    foundT=False
    if a!=b:
        if b>a:
            foundT=True
        else:
            foundT=False
    
    # Find xbw_grad peaks
    xbw_grad=abs(np.gradient(xbw))
    xpeaks_grad,xproperties_diff=find_peaks(xbw_grad,distance=distance,prominence=1,width=1)
    
    # Find the red peak or peaks
    pr=0.6 # 0.5
    width=1
    distance=half_width*2
    peaks_rg,rg_properties=find_peaks(xrb,distance=distance,prominence=pr,width=width)
    peaks_rb,rb_properties=find_peaks(xrg,distance=distance,prominence=pr,width=width)

    # Search left edge
    a=xn0-half_width
    width=200
    for p in sorted(peaks_rg,reverse=True):
        if p>=a:
            continue
        pos=p-width
        if pos<0:
            pos=0
        l=np.median(xr[pos:p])
        pos=p+width
        if pos>=w:
            pos=w-1
        r=np.median(xr[p:pos])
        if abs(l-r)>=2:
            xn0=p
            break

    # Search right edge
    b=xn1+half_width      
    for p in sorted(peaks_rg):
        if p<=b:
            continue
        pos=p-width
        if pos<0:
            pos=0
        l=np.median(xr[p-pos:p])
        if pos>=w:
            pos=w-1
        r=np.median(xr[p:p+pos])
        if abs(l-r)>=2:
            xn1=p
            break
    
    if xn0<0:
        xn0=0
    if xn1<0:
        xn1=w-1
    if xn0>=xn1:
        print('Invalid crop coordinates: ',xn0,xn1,yn0,yn1)
    else:
        pass

    # Crop and save the image
    img=img[:,xn0:xn1]
    if preserveExt:
        cv2.imwrite(adir+name,img)
    else:
        tmp=adir
        stem=Path(name).stem
        tmp+=stem+".png"
        cv2.imwrite(tmp,img)
    num+=1


if isROI:    
    print("")
    print("Saving roi.ini")
    # Build and save a ROI file
    img_x0=0
    img_y0=0
    img_x1=w
    img_y1=h
    crop_x0=xn0
    crop_x1=xn1
    crop_y0=yn0
    crop_y1=yn1
    roi_x0=roix(crop_x0)
    roi_w=roix(crop_x1-crop_x0)
    roi_y0=roiy(crop_y0)
    roi_h=roiy(crop_y1-crop_y0)
    
    roilist=[]

    roilist.append(["scale","coordinate name","value"])
    roilist.append(["original","img_x0",img_x0])
    roilist.append(["original","img_x1",img_x1])
    roilist.append(["original","img_y0",img_y0])
    roilist.append(["original","img_y1",img_y1])
    roilist.append(["original","crop_x0",crop_x0])
    roilist.append(["original","crop_x1",crop_x1])
    roilist.append(["original","crop_y0",crop_y0])
    roilist.append(["original","crop_y1",crop_y1])
    roilist.append(["normalized","roi_x0",roi_x0])
    roilist.append(["normalized","roi_y0",roi_y0])
    roilist.append(["normalized","roi_w",roi_w])
    roilist.append(["normalized","roi_h",roi_h])
    
    with open("roi.ini","w",newline="") as csvfile:
        csvwriter=csv.writer(csvfile,delimiter=";")
        for s in roilist:
            csvwriter.writerow(s)
    
    # Save ROI images
    cv2.imwrite("roi.jpg",img)

t2=datetime.now()

print("")
print("Files processed: "+str(num))
print("Time elapsed: "+str(t2-t1))

