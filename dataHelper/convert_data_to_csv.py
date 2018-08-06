import os
import struct as st
import numpy as np
import pandas as pd

filename = {'images' : 't10k-images.idx3-ubyte' ,'labels' : 't10k-labels.idx1-ubyte'}
test_imagesfile = open(filename['images'],'rb')

test_imagesfile.seek(0)
magic = st.unpack('>4B',test_imagesfile.read(4))

nImg = st.unpack('>I',test_imagesfile.read(4))[0] #num of images
nR = st.unpack('>I',test_imagesfile.read(4))[0] #num of rows
nC = st.unpack('>I',test_imagesfile.read(4))[0] #num of column

images_array = np.zeros((nImg,nR,nC))

nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
images_array = np.asarray(st.unpack('>'+'B'*nBytesTotal,test_imagesfile.read(nBytesTotal))).reshape((nImg,nR*nC))


print(images_array)
test_labelsfile = open(filename['labels'],'rb')

test_labelsfile.seek(0)
magic = st.unpack('>4B', test_labelsfile.read(4))

nLbl = st.unpack('>I', test_labelsfile.read(4))[0] #number of labels
label_array = np.asarray(st.unpack('>'+'B'*nLbl,test_labelsfile.read(nLbl)))


total = np.c_[label_array,images_array]
print(total)
import csv
from tqdm import tqdm
with open('test.csv', 'w', newline='') as csvfile:
    testwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for i in tqdm(range(nLbl)):
        testwriter.writerow(total[i])


