"""Basic idea comes from "Color-Based Segmentation Using K-Means Clusterings"
https://www.mathworks.com/help/images/examples/color-based-segmentation-using-k-means-clustering.html"""

from __future__ import print_function
from __future__ import division
import time
import os
import numpy as np
import scipy.misc as spm
from matplotlib import pyplot
from sklearn.cluster import KMeans


def segmentImage():
    name ="0d1a9caf4350_06"

    img = spm.imread("Kaggle_Image_TrainingData\Train\\" + name +".jpg" )
    mask =spm.imread("Kaggle_Image_TrainingData\\train_masks\\" +name+ "_mask.gif")
    mask[mask==0] = 1
    mask[mask==255]=0
    mask[mask==1]=255

    ab= img[:,:,2:3]
    height =ab.shape[0]
    width =ab.shape[1]
    ab = np.reshape(ab,(height*width,1))
    x= np.array(ab)
    clusterN=3
    kmeans= KMeans(n_clusters=clusterN,random_state=0).fit(x)
    labels= np.reshape(kmeans.labels_,(height,width))
    pyplot.imshow(labels)
    pyplot.xlabel("K-mean clustering result with k="+str(clusterN))
    pyplot.savefig("Result\\"+name+"Cluster")
    rgb_label = np.reshape(labels,(height,width,1))
    rgb_label = np.tile(rgb_label,(1,1,3))
    errors=[]
    for i in range (0,clusterN):
        nerror=0
        total=0
        print(i)
        color = img.copy()
        diff_color=img.copy()
        color[rgb_label != i] = 0
        diff_color[rgb_label ==i]=0
        diff_color[rgb_label !=i]=255
        pyplot.imshow(diff_color)
        pyplot.xlabel(str(i)+"-mask")
        pyplot.savefig("Result\\"+name+"mask"+ str(i))
        for j,k in zip(mask,diff_color):
            if j.all()==k.all():
                nerror=nerror+1
            total=total+1
        nerror=nerror/total
        print(nerror)
        errors.append(nerror)
        pyplot.imshow(color)
        pyplot.xlabel(str(i) + "-cluster:Error"+str(nerror))
        pyplot.savefig("Result\\"+name+ "result"+str(i))
    print(max(errors))


#main
if __name__ == '__main__':
    start_time = time.clock()
    pyplot.ion()
    if (os.path.isdir("Result")):
        '''do nothing'''
    else: os.mkdir("Result")

    segmentImage()
    print(time.clock() - start_time, "seconds")