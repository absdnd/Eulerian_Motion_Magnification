## Butterworth bandpassing. 
import cv2
import sys
import scipy.signal as signal        
import scipy.fftpack as fftpack      
import skimage as ski               
from matplotlib import pyplot as plt
from skimage import img_as_float    
from skimage import img_as_ubyte     
import time
import numpy as np
import pyrtools as pt
import os
from skimage import color
import pdb

def reconstruct(pyr,pind,levels = 5):
	
	reshaped_pyr = []
	for k in range(levels-1,-1,-1):
		startIndex = sum(np.prod(j) for j in pind[0:k])
		endIndex = ind - sum(np.prod(j) for j in pind[k+1:])
		reshaped_pyr.append((pyr[startIndex:endIndex]).reshape(pind[k]))                
	reshaped_pyr = reshaped_pyr[::-1]                                                  
	return reconPyr(reshaped_pyr)

# Linearize the pyramid. 
def linearize(pyramid,pind):
	nLevels = len(pyramid)
	se = sum([np.prod(i) for i in pind])
	pyr = np.zeros(se)
	pyrl = []

	# for j in range(nLevels)
	# We are finding the linear pyramid. 
	for j in range(nLevels):
		shape = np.shape(pyramid[(j,0)])
		pyrl = pyrl +  (pyramid[(j,0)].reshape(shape[0]*shape[1])).tolist()  
	   

	return np.asarray(pyrl)

# Reconstruct the pyramid. 
def reconPyr(pyr):
	# Reconstruct the pyramid now.                                                
    filt2 = 'binom5'                #The binomial filter for image reconstruction 
    edges = 'reflect1';             #The edges is reflect1. I have used this here. 
    maxLev = len(pyr)
    levs = range(0,maxLev)                 # The levels is range(0,maxLev)
    filt2 = pt.binomial_filter(5)  #The named Filter filt2 . This has been finalized here. 
    res = []
    lastLev = -1

    # pdb.set_trace()

    for lev in range(maxLev-1, -1, -1):
        if lev in levs and len(res) == 0:
            res = pyr[lev]
        elif len(res) != 0:
            res_sz = res.shape
            new_sz = pyr[lev].shape
            filt2_sz = filt2.shape
            if res_sz[0] == 1:
                hi2 = pt.upConv(image = res, filt = filt2,
                                        step = (2,1), 
                                        stop = (new_sz[1], new_sz[0])).T
            elif res_sz[1] == 1:
                hi2 = pt.upConv(image = res, filt = filt2.T,
                                        step = (1,2), 
                                        stop = (new_sz[1], new_sz[0])).T
            else:
                hi = pt.upConv(image = res, filt = filt2, 
                                       step = (2,1), 
                                       stop = (new_sz[0], res_sz[1]))
                hi2 = pt.upConv(image = hi, filt = filt2.T, 
                                       step = (1,2),
                                       stop = (new_sz[0], new_sz[1]))
            if lev in levs:
                bandIm =  pyr[lev]
                bandIm_sz = bandIm.shape
                res = hi2 + bandIm
            else:
                res = hi2
    return res                             
				
   
# Loading the video from the source path. 
video_name = 'baby.mp4'
base_path = './source'
video_path = os.path.join(base_path, video_name)
vid  = cv2.VideoCapture(video_path)
alpha = 100          
lambda_c = 10          
chromAttenuation = 1.
fl = 2.3333
fh = 2.6666         
samplingRate = 30.    


## Low_a, Low_B is being used. 
[low_a,low_b] = signal.butter(1,fl/samplingRate,'low')   
[high_a,high_b] = signal.butter(1,fh/samplingRate,'low')


width, height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
nChannels = 3 
fps =  int(vid.get(cv2.CAP_PROP_FPS)) 
frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) 


# Start index = 1 to frame_count-10. 
startIndex  = 1                                                
end =  frame_count-10    

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')   
os.makedirs('results', exist_ok=True)            
writer = cv2.VideoWriter('results/butter'+str(video_name)+'fl_'+str(fl)+'fh_'+str(fh)+'.avi', fourcc, 30, (width, height), 1) 

# Setting video for writing. 
vid.set(1,1) 
_, bgrframe = vid.read() 
rgbframe = bgrframe[:,:,::-1]
rgbframe = img_as_float(rgbframe)
YIQ = color.rgb2yiq(rgbframe)
Y,I,Q  = YIQ[:,:,0], YIQ[:,:,1], YIQ[:,:,2]

# PyramidY is being constructed here. 
pyYa = pt.pyramids.LaplacianPyramid(Y)
pyYa._build_pyr()
pyY = pyYa.pyr_coeffs

# PyramidI is being constructed. 
pyIa = pt.pyramids.LaplacianPyramid(I)
pyIa._build_pyr()
pyI = pyIa.pyr_coeffs

# PyramidQ is being contructed here. 
pyQa = pt.pyramids.LaplacianPyramid(Q)
pyQa._build_pyr()
pyQ = pyQa.pyr_coeffs

# number of levels is mentioned here. 
nLevels  = len(pyY)
pind = ([np.shape(pyY[(i,0)]) for i in range(nLevels)])  

# Linearize the pyramid using pind. 
pyY = linearize(pyY,pind)
pyI = linearize(pyI,pind)
pyQ = linearize(pyQ,pind)
   
# Creating an array using pyY, pyI, pyQ. 
pyr = np.asarray([pyY,pyI,pyQ])

# Lowpass1, Lowpass2, pyr_prev. 
lowpass1 = pyr
lowpass2 = pyr
pyr_prev = pyr

output = rgbframe
writer.write(np.uint8(rgbframe*255))

# Printing the frames from startIndex+1 to end. 
for i in range(startIndex+1,end):                           
	vid.set(1,i)
	print("frame", i,"of",end)
	_, bgrframe = vid.read() 
	rgbframe = bgrframe[:,:,::-1]
	rgbframe = img_as_float(rgbframe)
	YIQ = color.rgb2yiq(rgbframe)
	Y,I,Q  = YIQ[:,:,0], YIQ[:,:,1], YIQ[:,:,2]
	
	pyYa = pt.pyramids.LaplacianPyramid(Y)
	pyYa._build_pyr()
	pyY = pyYa.pyr_coeffs

	# PyramidI is being used. 
	pyIa = pt.pyramids.LaplacianPyramid(I)
	pyIa._build_pyr()
	pyI = pyIa.pyr_coeffs

	# PyramidQ is being contructed here. 
	pyQa = pt.pyramids.LaplacianPyramid(Q)
	pyQa._build_pyr()
	pyQ = pyQa.pyr_coeffs

	# Number of levels.
	nLevels  = len(pyY)
	pind = ([np.shape(pyY[(i,0)]) for i in range(nLevels)])
	
	pyY = linearize(pyY,pind)
	pyI = linearize(pyI,pind)
	pyQ = linearize(pyQ,pind)
	
	pyr = np.asarray([pyY,pyI,pyQ])

	# Filtering the signal.
	lowpass1 = (-high_b[1]*lowpass1 + high_a[0]*pyr + high_a[1]*pyr_prev)/high_b[0]
	lowpass2 = (-low_b[1]*lowpass2 + low_a[0]*pyr + low_a[1]*pyr_prev)/low_b[0]
	filtered =  lowpass1-lowpass2 
	
	pyr_prev = pyr                                               
	ind = len(pyr[0])                                           
   
	delta = lambda_c/8./(1+alpha)                                                           
	exaggeration_factor = 2                                       
	lambd = (width^2+height^2)/3.

	# for all levels we try to obtain the chrom attenutation. 
	for l in range(nLevels-1,-1,-1):
		startIndex = sum(np.prod(j) for j in pind[0:l])
		endIndex = ind - sum(np.prod(j) for j in pind[l+1:])
		currAlpha = lambd/delta/8. - 1
		currAlpha = currAlpha*exaggeration_factor;
		indices = range(startIndex,endIndex)                   
		
		if(l == nLevels - 1 or l==0):
			filtered[:,indices] = 0.                                                                       
		elif (currAlpha>alpha):
			filtered[:,indices] = alpha*filtered[:,indices]           
		
		else:
			filtered[:,indices] = currAlpha*filtered[:,indices]           

		lambd = lambd/2.
   

	# Reconstrion of the signal. 
	output[:,:,0] = reconstruct(filtered[0,:],pind,nLevels)                      
	output[:,:,1] = reconstruct(filtered[1,:],pind,nLevels)
	output[:,:,2] = reconstruct(filtered[2,:],pind,nLevels)
	output[:,:,1] = chromAttenuation*output[:,:,1]
	output[:,:,2] = chromAttenuation*output[:,:,2]
	
	# Rgb output obtained here. 
	rgb_out = color.yiq2rgb(output)
	
	# Rgbframe is being obtained here. 
	rgbframe = rgbframe + rgb_out
	output_final = rgbframe[:,:,::-1]
	
	output_final[output_final>1] = 1
	output_final[output_final<-1] = -1
	
	output_final  = img_as_ubyte(output_final)
	writer.write(output_final)
	
writer.release()
