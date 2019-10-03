import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Lenna.png')
img =np.asarray(img, np.float64)

h1 = np.array(([1,1,0],[1, 0, -1], [0, -1 , -1]), dtype = 'float32')
fi1 = cv2.filter2D(img, -1, h1)
fi1 =abs(fi1)
fi1 = fi1/np.amax(fi1[:])

h2 = np.array(([1,1,1],[0, 0, 0], [-1, -1 , -1]), dtype = 'float32')
fi2 = cv2.filter2D(img, -1, h2)
fi2 =abs(fi2)
fi2 = fi2/np.amax(fi2[:])

h3 = np.array(([0,1,1],[-1, 0, 1], [-1, -1 , 0]), dtype = 'float32')
fi3 = cv2.filter2D(img, -1, h3)
fi3 =abs(fi3)
fi3 = fi3/np.amax(fi3[:])

h4 = np.array(([1,0,-1],[1, 0, -1], [1, 0 , -1]), dtype = 'float32')
fi4 = cv2.filter2D(img, -1, h4)
fi4 =abs(fi4)
fi4 = fi4/np.amax(fi4[:])

h5 = np.array(([0,-1,-1],[1, 0, -1], [1, 1 , 0]), dtype = 'float32')
fi5 = cv2.filter2D(img, -1, h5)
fi5 =abs(fi5)
fi5 = fi5/np.amax(fi5[:])

h6 = np.array(([-1,-1,-1],[0, 0, 0], [1, 1 , 1]), dtype = 'float32')
fi6 = cv2.filter2D(img, -1, h6)
fi6 =abs(fi6)
fi6 = fi6/np.amax(fi6[:])

h7 = np.array(([-1,-1,0],[-1, 0, 1], [0, 1 , 1]), dtype = 'float32')
fi7 = cv2.filter2D(img, -1, h7)
fi7 =abs(fi7)
fi7 = fi7/np.amax(fi7[:])

h8 = np.array(([-1,0,1],[-1, 0, 1], [-1, 0 , 1]), dtype = 'float32')
fi8 = cv2.filter2D(img, -1, h8)
fi8 =abs(fi8)
fi8 = fi8/np.amax(fi8[:])

edgeSum1=cv2.add(np.power(fi1,2),np.power(fi2,2),np.power(fi3,2), np.power(fi4,2), dtype = cv2.CV_8UC1)
edgeSum = cv2.add(edgeSum1,np.power(fi5,2),np.power(fi6,2),np.power(fi7,2),np.power(fi8,2), dtype = cv2.CV_8UC1)
thresh = cv2.threshold(edgeSum, 0.05, 1, cv2.THRESH_BINARY)

plt.figure(1)
plt.subplot(421), plt.imshow(fi1)
plt.subplot(422), plt.imshow(fi2)
plt.subplot(423), plt.imshow(fi3)
plt.subplot(424), plt.imshow(fi4)
plt.subplot(425), plt.imshow(fi5)
plt.subplot(426), plt.imshow(fi6)
plt.subplot(427), plt.imshow(fi7)
plt.subplot(428), plt.imshow(fi8)

plt.figure(2), plt.imshow(thresh), plt.title("EDGES")

plt.show()
