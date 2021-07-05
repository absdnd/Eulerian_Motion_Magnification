import cv2   
import scipy.signal as signal        
import scipy.fftpack as fftpack                      
from matplotlib import pyplot as plt 
from skimage import img_as_float     
from skimage import img_as_ubyte              
import numpy as np
import time
import pyrtools as pt
import pdb
import copy

# Reconstruct pyramid is being done here. 
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

    # for lev in range(levels).
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

# Magnify only the grayscale image. 
class Magnify(object):
	
	def  __init__(self, gray1,alpha, lambda_c, fl, fh,samplingRate):

		[low_a,low_b] = signal.butter(1,fl/samplingRate,'low')   
		[high_a,high_b] = signal.butter(1,fh/samplingRate,'low')

		# For py1 in range number of levels. 
		py1 = pt.pyramids.LaplacianPyramid(gray1)
		py1._build_pyr()
		# Building the initial pyramid. 
		pyramid_1 = py1.pyr_coeffs
		# Pyramid_1 has 7 keys. 
		# pdb.set_trace()
		nLevels = len(pyramid_1)
		self.filtered = pyramid_1
		self.alpha  = alpha
		self.fl = fl
		self.fh = fh
		self.samplingRate = samplingRate
		self.low_a = low_a
		self.low_b = low_b
		self.high_a = high_a
		self.high_b = high_b
		self.width = gray1.shape[0]
		self.height = gray1.shape[1]
		self.gray1 = img_as_float(gray1)
		self.lowpass1 = copy.deepcopy(pyramid_1)
		self.lowpass2 = copy.deepcopy(self.lowpass1)
		self.pyr_prev = copy.deepcopy(pyramid_1)
		self.filtered = [None for _ in range(nLevels)]
		self.nLevels = nLevels
		self.lambd = (self.width^2+self.height^2)/3.
		self.lambda_c = lambda_c
		self.delta =  self.lambda_c/8./(1+self.alpha) 
	
	
	def Magnify(self, gray2): 
		u = 0
		l = 0
		gray2 = img_as_float(gray2)
		# Building second pyramid. 
		py2 = pt.pyramids.LaplacianPyramid(gray2)
		py2._build_pyr()
		pyr = py2.pyr_coeffs
		nLevels = self.nLevels
		for u in range(nLevels):
			self.lowpass1[(u,0)] = (-self.high_b[1]*self.lowpass1[(u,0)] + self.high_a[0]*pyr[(u,0)]+ self.high_a[1]*self.pyr_prev[(u,0)])/self.high_b[0]
			self.lowpass2[(u,0)] = (-self.low_b[1]*self.lowpass2[(u,0)]+ self.low_a[0]*pyr[(u,0)] + self.low_a[1]*self.pyr_prev[(u,0)])/self.low_b[0]
			self.filtered[u] = self.lowpass1[(u,0)]-self.lowpass2[(u,0)]
		 
		self.pyr_prev = copy.deepcopy(pyr)
		exaggeration_factor = 2                                       
		lambd = self.lambd
		delta = self.delta
		filtered  = self.filtered
		
		for l in range(nLevels-1,-1,-1):
			
			currAlpha = lambd/delta/8. - 1                            
			currAlpha = currAlpha*exaggeration_factor;
		
			if(l == nLevels - 1 or l==0):
				filtered[l] = np.zeros(np.shape(filtered[l]))

			elif (currAlpha>self.alpha):
				filtered[l] = self.alpha*filtered[l]  
		 
			else:
				filtered[l] = currAlpha*filtered[l]           
		   
			lambd = lambd/2.

		# Reconstruct the pyramid using filtered. 
		output = reconPyr(filtered)          
		output = gray2 + output	    
		output[output<0] =  0
		output[output>1]  = 1 
		output  = img_as_ubyte(output)
		
		return output

	

fps = 8.
alpha = 300
lambda_c = 200
fl = 0.3
fh = 1.
cam = cv2.VideoCapture(0)
_,img1 = cam.read()
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
s = Magnify(gray,alpha,lambda_c,fl,fh,fps)
while True:
    t1 = time.clock()
    _,final_img = cam.read()
    gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    out = s.Magnify(gray)
    cv2.imshow('final_img',final_img)
    cv2.imshow('final', out) 
    k = cv2.waitKey(1)
    t2 = time.clock()
    
    print("set fps",1./(t2-t1))
    
    if(k==27):
        break
    
    if t2 - t1>1./fps:
        print("delayed")
    



cam.release()
cv2.destroyAllWindows()
