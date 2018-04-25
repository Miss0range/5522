"""Basic idea comes from "Color-Based Segmentation Using K-Means Clusterings"
https://www.mathworks.com/help/images/examples/color-based-segmentation-using-k-means-clustering.html"""

from __future__ import print_function
from __future__ import division
import time
import numpy as np
import scipy.misc as spm
from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn.svm import SVC


def segmentImage():
    img = spm.imread("Kaggle_Image_TrainingData/train/0d1a9caf4350_05.jpg")
    ab= img[:,:,2:3]
    height =ab.shape[0]
    width =ab.shape[1]
    ab = np.reshape(ab,(height*width,1))
    x= np.array(ab)
    clusterN=3
    kmeans= KMeans(n_clusters=clusterN,random_state=0).fit_predict(x)
    svm= SVC()
    result=svm.fit(x,kmeans)
    print(result)
    labels= np.reshape(kmeans.labels_,(height,width))
    pyplot.imshow(result)
    '''pyplot.show(block=True)
    rgb_label = np.reshape(labels,(height,width,1))
    rgb_label = np.tile(rgb_label,(1,1,3))
    print(rgb_label)

    for i in range (0,clusterN):
        print(i)
        color = img.copy()
        color[rgb_label != i] = 0
        pyplot.imshow(color)
        pyplot.show(block= True)
    pyplot.imshow(img)
    pyplot.show(block=True)'''

#main
if __name__ == '__main__':
    start_time = time.clock()
    pyplot.ion()
    segmentImage()
    print(time.clock() - start_time, "seconds")