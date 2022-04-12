#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sun Mar 13 18:56:36 2022

@author: Kim Miikki
Examples:
-s 1 -u "Time (min)" -i 0.08333333333333333
-s 57 -i 0.1
-s 55 -i 0.1 -p 1

'''

import argparse
import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, peak_widths
from sklearn.metrics import auc

start=0
interval=0.04 # 1/25 s
defaultXlabel='Time (s)'
xlabel=defaultXlabel
distance=1000
prominence=1
width=10
rel_height=1
direction='' # x or y
showGraphs=False
scale_y=False
scale_x=False
linew=1
grid=True

parser=argparse.ArgumentParser()
parser.add_argument('-s',type=float,help='start time for first frame as float (default = '+str(start)+')',required=False)
parser.add_argument('-i',type=float,help='interval between two frames as float (default = '+str(interval)+')',required=False)
parser.add_argument('-u', nargs='*', type=str,help='x label (default = '+defaultXlabel+')',required=False)
parser.add_argument('-d',type=int,help='minimal horizontal distance as integer',required=False)
parser.add_argument('-p',type=float,help='required prominence as float',required=False)
parser.add_argument('-w',type=int,help='minimal peak width as integer',required=False)
parser.add_argument('-r',type=float,help='auc: relative height as float',required=False)
parser.add_argument("-x", action="store_true", help="auto X scale")
parser.add_argument("-y", action="store_true", help="auto Y scale")
args = parser.parse_args()

errors=[]

if args.u != None:
    if len(args.u)>0:
        xlabel=args.u[0]

if args.s != None:
    tmp=float(args.s)
    if tmp>0:
        start=tmp
    else:
        errors.append('invalid s value: s>0')

if args.i != None:
    tmp=float(args.i)
    if tmp>0:
        interval=tmp
    else:
        errors.append('invalid s value: i>0')

if args.d != None:
    tmp=int(args.d)
    if tmp>0:
        distance=tmp
    else:
        errors.append('invalid d value: d>0')

if args.p != None:
    tmp=float(args.p)
    if tmp>0:
        prominence=tmp
    else:
        errors.append('invalid p value: p>0')
        
if args.w != None:
    tmp=float(args.w)
    if tmp>0:
        width=tmp
    else:
        errors.append('invalid w value: w>0')

if args.r != None:
    tmp=float(args.r)
    if tmp>0:
        rel_height=tmp
    else:
        errors.append('invalid r value: r>0')

if args.x:
    scale_x=True

if args.y:
    scale_y=True

print('RGBXY Peak Analyze Utility 1.1, (c) Kim Miikki 2022')
print('')
print('Current directory:')
curdir=os.getcwd()
print(curdir)
print('')
path=Path(curdir)
adir=str(path)+'/peaks/'

# Generate a list of input files (CSV)
files=[]
for p in sorted(path.iterdir()):
    suffix=p.suffix.lower()
    if p.is_file() and suffix=='.csv':
        files.append(p.name)            
if len(files)==0:
    errors.append('csv files not found in current directory')

# Exit the program if any errors
if len(errors)>0:
    print('Error(s):')
    for text in errors:
        print(text)
    print('')
    print('Program is terminated.')
    print('')
    sys.exit(1)

try:
  if not os.path.exists(adir):
    os.mkdir(adir)
except OSError:
  print('Unable to create a directory under following path:\n'+curdir)
  print('Program is terminated.')
  print('')
  sys.exit(1)

# Process all csv files
first=True
count=0

# peaks
pred=[]
pgreen=[]
pblue=[]

# valleys
vred=[]
vgreen=[]
vblue=[]

for file in files:
    # Read the data from a csv file
    #
    # Input template:
    # x,R,G,B,BW
    rgb_data=[]
    pos=[]
    hlen=0
    try:
        with open(file,'r') as reader_obj:
            csv_reader=csv.reader(reader_obj)
            header_rgb=next(csv_reader)
            if first:
                direction=header_rgb[0].upper()
                first=False
            hlen=len(header_rgb)
                   
            for row in csv_reader:
                if hlen==5:
                    rgb_data.append(list(map(float,row[1:-1])))
                elif hlen==4:
                    rgb_data.append(list(map(float,row[1:])))
                pos.append(int(row[0]))
    except OSError:
        print('Unable to open '+file+' in following directory:\n'+curdir)
        print('Program is terminated.')
        print('')
        sys.exit(1)
    count+=1
    if len(rgb_data)==0:
        continue
    
    time=round(start+(count-1)*interval,3)
    
    # Extract RGB data
    data=[]

    datalen=len(rgb_data)
    arr1=np.array(rgb_data)
    arr1=arr1.T
    
    xs=pos
    r=arr1[0]
    g=arr1[1]
    b=arr1[2]
    
    # Search red peaks
    rpeaks,rproperties=find_peaks(r,distance=distance,prominence=prominence,width=width)
    rcontour_heights=r[rpeaks]-rproperties['prominences']*rel_height
    rfull=peak_widths(r,rpeaks,rel_height=rel_height)
    ph=r[rpeaks]-rcontour_heights
    if showGraphs:
        fig=plt.figure()
        plt.xlabel(header_rgb[0])
        plt.ylabel(header_rgb[1])
        plt.xlim(xs[0],xs[-1])
        plt.plot(r,color='r')
        plt.plot(rpeaks,r[rpeaks],'x',color='k')
        plt.vlines(x=rpeaks,ymin=rcontour_heights,ymax=r[rpeaks],color='k')
        plt.hlines(*rfull[1:],color='k')
        plt.grid()
        plt.show()
        plt.close()
    
    i=0
    while i<len(rpeaks):
        color='R'
        peak=1 # 1 = Peak, -1 = Valley
        pos=rpeaks[i]
        rheight=rel_height
        # Calculate area under curve
        x1=int(round(rfull[2][i]))
        x2=int(round(rfull[3][i]))
        area=int(round(auc(xs[x1:x2],r[x1:x2])))
        peak_height=round(ph[i],6)
        ybase=rcontour_heights[i]
        pred.append([time,color,peak,pos,rheight,area,peak_height,ybase,x1,x2])   
        i+=1
        # Skip next peaks
        break
    
    # Search red valleys
    r=-r
    rpeaks,rproperties=find_peaks(r,distance=distance,prominence=prominence,width=width)
    rcontour_heights=r[rpeaks]-rproperties['prominences']*rel_height
    rfull=peak_widths(r,rpeaks,rel_height=rel_height)
    ph=r[rpeaks]-rcontour_heights
    if showGraphs:
        fig=plt.figure()
        plt.xlabel(header_rgb[0])
        plt.ylabel(header_rgb[1])
        plt.xlim(xs[0],xs[-1])        
        plt.plot(-r,color="r")
        plt.plot(rpeaks,-r[rpeaks],"x",color="k")
        plt.vlines(x=rpeaks,ymin=-rcontour_heights,ymax=-r[rpeaks],color="k")
        plt.hlines(y=-rfull[1],xmin=rfull[2],xmax=rfull[3],color="k")    
        plt.grid()
        plt.show()
        plt.close()
    
    i=0
    while i<len(rpeaks):
        color='R'
        peak=-1 # 1 = Peak, -1 = Valley
        pos=rpeaks[i]
        rheight=rel_height
        # Calculate area under curve
        x1=int(round(rfull[2][i]))
        x2=int(round(rfull[3][i]))
        area=abs(int(round(auc(xs[x1:x2],r[x1:x2]))))
        peak_height=round(ph[i],6)
        ybase=rcontour_heights[i]
        vred.append([time,color,peak,pos,rheight,area,peak_height,ybase,x1,x2])   
        i+=1
        # Skip next valleys
        break
    
    # Search green peaks
    gpeaks,gproperties=find_peaks(g,distance=distance,prominence=prominence,width=width)
    gcontour_heights=g[gpeaks]-gproperties['prominences']*rel_height
    gfull=peak_widths(g,gpeaks,rel_height=rel_height)
    ph=g[gpeaks]-gcontour_heights
    if showGraphs:
        fig=plt.figure()
        plt.xlabel(header_rgb[0])
        plt.ylabel(header_rgb[2])
        plt.xlim(xs[0],xs[-1])
        plt.plot(g,color='g')
        plt.plot(gpeaks,g[gpeaks],'x',color='k')
        plt.vlines(x=gpeaks,ymin=gcontour_heights,ymax=g[gpeaks],color='k')
        plt.hlines(*gfull[1:],color='k')
        plt.grid()
        plt.show()
        plt.close()
    
    i=0
    while i<len(gpeaks):
        color='G'
        peak=1 # 1 = Peak, -1 = Valley
        pos=gpeaks[i]
        rheight=rel_height
        # Calculate area under curve
        x1=int(round(gfull[2][i]))
        x2=int(round(gfull[3][i]))
        area=int(round(auc(xs[x1:x2],g[x1:x2])))
        peak_height=round(ph[i],6)
        ybase=gcontour_heights[i]
        pgreen.append([time,color,peak,pos,rheight,area,peak_height,ybase,x1,x2])   
        i+=1
        # Skip next peaks
        break
    
    # Search green valleys
    g=-g
    gpeaks,gproperties=find_peaks(g,distance=distance,prominence=prominence,width=width)
    gcontour_heights=g[gpeaks]-gproperties['prominences']*rel_height
    gfull=peak_widths(g,gpeaks,rel_height=rel_height)
    ph=g[gpeaks]-gcontour_heights
    if showGraphs:
        fig=plt.figure()
        plt.xlabel(header_rgb[0])
        plt.ylabel(header_rgb[2])
        plt.xlim(xs[0],xs[-1])
        plt.plot(-g,color="g")
        plt.plot(gpeaks,-g[gpeaks],"x",color="k")
        plt.vlines(x=gpeaks,ymin=-gcontour_heights,ymax=-g[gpeaks],color="k")
        plt.hlines(y=-gfull[1],xmin=gfull[2],xmax=gfull[3],color="k")
        plt.grid()
        plt.show()
        plt.close()
    
    i=0
    while i<len(gpeaks):
        color='G'
        peak=-1 # 1 = Peak, -1 = Valley
        pos=gpeaks[i]
        rheight=rel_height
        # Calculate area under curve
        x1=int(round(gfull[2][i]))
        x2=int(round(gfull[3][i]))
        area=abs(int(round(auc(xs[x1:x2],g[x1:x2]))))
        peak_height=round(ph[i],6)
        ybase=gcontour_heights[i]
        vgreen.append([time,color,peak,pos,rheight,area,peak_height,ybase,x1,x2])   
        i+=1
        # Skip next valleys
        break

    # Search blue peaks
    bpeaks,bproperties=find_peaks(b,distance=distance,prominence=prominence,width=width)
    bcontour_heights=b[bpeaks]-bproperties['prominences']*rel_height
    bfull=peak_widths(b,bpeaks,rel_height=rel_height)
    ph=b[bpeaks]-bcontour_heights
    if showGraphs:
        fig=plt.figure()
        plt.xlabel(header_rgb[0])
        plt.ylabel(header_rgb[3])
        plt.xlim(xs[0],xs[-1])
        plt.plot(b,color='b')
        plt.plot(bpeaks,b[bpeaks],'x',color='k')
        plt.vlines(x=bpeaks,ymin=bcontour_heights,ymax=b[bpeaks],color='k')
        plt.hlines(*bfull[1:],color='k')
        plt.grid()
        plt.show()
        plt.close()
    
    i=0
    while i<len(bpeaks):
        color='B'
        peak=1 # 1 = Peak, -1 = Valley
        pos=bpeaks[i]
        rheight=rel_height
        # Calculate area under curve
        x1=int(round(bfull[2][i]))
        x2=int(round(bfull[3][i]))
        area=int(round(auc(xs[x1:x2],b[x1:x2])))
        peak_height=round(ph[i],6)
        ybase=gcontour_heights[i]
        pblue.append([time,color,peak,pos,rheight,area,peak_height,ybase,x1,x2])   
        i+=1
        # Skip next peaks
        break
    
    # Search blue valleys
    b=-b
    bpeaks,bproperties=find_peaks(b,distance=distance,prominence=prominence,width=width)
    bcontour_heights=b[bpeaks]-bproperties['prominences']*rel_height
    bfull=peak_widths(b,bpeaks,rel_height=rel_height)
    ph=b[bpeaks]-bcontour_heights
    if showGraphs:
        fig=plt.figure()
        plt.xlabel(header_rgb[0])
        plt.ylabel(header_rgb[3])
        plt.xlim(xs[0],xs[-1])        
        plt.plot(-b,color="b")
        plt.plot(bpeaks,-b[bpeaks],"x",color="k")
        plt.vlines(x=bpeaks,ymin=-bcontour_heights,ymax=-b[bpeaks],color="k")
        plt.hlines(y=-bfull[1],xmin=bfull[2],xmax=bfull[3],color="k")        
        plt.grid()
        plt.show()
        plt.close()
    
    i=0
    while i<len(bpeaks):
        color='B'
        peak=-1 # 1 = Peak, -1 = Valley
        pos=bpeaks[i]
        rheight=rel_height
        # Calculate area under curve
        x1=int(round(bfull[2][i]))
        x2=int(round(bfull[3][i]))
        area=abs(int(round(auc(xs[x1:x2],b[x1:x2]))))
        peak_height=round(ph[i],6)
        ybase=gcontour_heights[i]
        vblue.append([time,color,peak,pos,rheight,area,peak_height,ybase,x1,x2])   
        i+=1
        # Skip next valleys
        break

# Create numpy arrays
if len(pred)>len(vred):
    red=np.array(pred)
else:
    red=np.array(vred)

if len(pgreen)>len(vgreen):
    green=np.array(pgreen)
else:
    green=np.array(vgreen)

if len(pblue)>len(vblue):
    blue=np.array(pblue)
else:
    blue=np.array(vblue)

red=red.T
green=green.T
blue=blue.T

xr=red[0].astype(float)
xg=green[0].astype(float)
xb=blue[0].astype(float)

ra=red[5].astype(float)
ga=green[5].astype(float)
ba=blue[5].astype(float)

rh=red[6].astype(float)
gh=green[6].astype(float)
bh=blue[6].astype(float)

# Calculate Rh/Gh and Rh/Bh ratios
isRG=True
isRB=True
try:
    rg=rh/gh
except:
    isRG=False
try:
    rb=rh/bh
except:
    isRB=False


w=8
h=2.4
width=10
height=width*h/w

# Plot RGB peaks heights = f(time)
#fig=plt.figure(figsize = (width,height))
fig=plt.figure()
ylabel="RGB extrema heights"
ymins=[rh.min(),gh.min(),bh.min()]
ymaxs=[rh.max(),gh.max(),bh.max()]
ymin=min(ymins)
ymax=max(ymaxs)
if not scale_y:
    ymin=0
    ymax=255
xmins=[xr.min(),xg.min(),xb.min()]
xmaxs=[xr.max(),xg.max(),xb.max()]
xmin=min(xmins)
xmax=max(xmaxs)
if not scale_x:
    xmin=0
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.plot(xr,rh,color="r",linewidth=linew)
plt.plot(xg,gh,color="g",linewidth=linew)
plt.plot(xb,bh,color="b",linewidth=linew)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
if grid:
    plt.grid()
plt.savefig(adir+"RGB-peaks-heights.png",dpi=300,bbox_inches='tight')
plt.show()
plt.close(fig)

# Plot RGB peaks areas = f(time)
fig=plt.figure()
ylabel="Peak/valley area (a.u.)"
ymins=[ra.min(),ga.min(),ba.min()]
ymaxs=[ra.max(),ga.max(),ba.max()]
ymin=min(ymins)
ymax=max(ymaxs)
if not scale_y:
    ymin=0
xmins=[xr.min(),xg.min(),xb.min()]
xmaxs=[xr.max(),xg.max(),xb.max()]
xmin=min(xmins)
xmax=max(xmaxs)
if not scale_x:
    xmin=0
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.plot(xr,ra,color="r",linewidth=linew)
plt.plot(xg,ga,color="g",linewidth=linew)
plt.plot(xb,ba,color="b",linewidth=linew)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
if grid:
    plt.grid()
plt.savefig(adir+"RGB-peaks-areas.png",dpi=300,bbox_inches='tight')
plt.show()
plt.close(fig)

if isRB or isRG:
    # Plot RGB peaks heights = f(time)
    fig=plt.figure()
    ylabel="Peak R/G and R/B ratios"
    ymins=[]
    ymaxs=[]
    xmins=[]
    xmaxs=[]
    if isRG:
        ymins.append(rg.min())
        ymaxs.append(rg.max())
    if isRB:
        ymins.append(rb.min())
        ymaxs.append(rb.max())
    ymin=min(ymins)
    ymax=max(ymaxs)
    if not scale_y:
        ymin=0
    xmins=[xr.min(),xg.min(),xb.min()]
    xmaxs=[xr.max(),xg.max(),xb.max()]
    xmin=min(xmins)
    xmax=max(xmaxs)
    if not scale_x:
        xmin=0
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    if isRG:
        plt.plot(xr,rg,color="g",linewidth=linew)
    if isRB:
        plt.plot(xr,rb,color="b",linewidth=linew)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if grid:
        plt.grid()
    plt.savefig(adir+"RGB-peaks-heights-ratios.png",dpi=300,bbox_inches='tight')
    plt.show()
    plt.close(fig)

