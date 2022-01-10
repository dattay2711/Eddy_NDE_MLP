import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.utils import shuffle
from numpy import save

def plot_graph(image):

    plt.imshow(image)
    plt.colorbar()
    plt.show()

    x = range(image.shape[1])
    y = range(image.shape[0])
    X,Y = np.meshgrid(x,y)
    Z = image

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot a 3D surface
    ax.plot_surface(X, Y, Z,cmap = 'jet')
    plt.show()

    """ Ma tran nhieu nen"""
def image_bg(image,top,bottom,left,right,dis):

    line_top = np.zeros((1, image.shape[1] - left-right))
    line_bottom = np.zeros((1, image.shape[1] - left-right))
    image_ground=np.zeros((image.shape[0]-top-bottom,image.shape[1]-left-right))
    for col in np.arange(left,image.shape[1] - right,1):
        for line in np.arange(top,top+dis,1):
            line_top[0][col - left] = line_top[0][col - left]+ image[line][col]
        for line in np.arange(image.shape[0]-dis-bottom,image.shape[0]-bottom,1):
            line_bottom[0][col - left] = line_bottom[0][col - left]+ image[line][col]

    line_top = line_top / dis
    line_bottom = line_bottom / dis
    line_average = 0.5 * (line_top + line_bottom)


    for line in np.arange(0,image.shape[0] - top-bottom,1):
        image_ground[line][:] = line_average
    return image_ground
    
"""Hieu chinh kich thuoc cua anh"""

def image_resize(image,top,bottom,left,right):
    image_corr=np.zeros((image.shape[0]-top-bottom,image.shape[1]-left-right))
    for line in np.arange(0,image.shape[0] - top-bottom,1):
        for col in np.arange(left,image.shape[1] - right,1):
            image_corr[line][col - left] = image[line + top][col]
    return image_corr

"""Bo anh nen"""
def image_remove_bg(image_ground,image_corr):
    image_sub = np.abs(image_ground-image_corr)
    return image_sub

'''Tim gia tri max'''
def image_max(image_sub):
    im_corr_size = image_sub.shape
    delta_max = image_sub[0][0]
    max_lin = 0
    max_col = 0

    for line in np.arange(0,im_corr_size[0]-1,1):
        for col in np.arange(0,im_corr_size[1]-1,1):
            if image_sub[line][col] - delta_max > 0:
                delta_max = image_sub[line][col]
                max_lin = line
                max_col = col
    return delta_max,max_lin,max_col

'''Tinh gia tri goc pha tai diem max'''
def max_angle (image_angle,max_lin,max_col):
    angle_max=image_angle[max_lin][max_col]
    return angle_max

"""Bo nhieu nen"""
'''base la gia tri so hang lay trung binh nhieu nen'''
def remove_noise_vertical(image_sub,base,std):
    noise_mat=np.zeros((base,image_sub.shape[1]))
    for line in np.arange(0,base,1):
        noise_mat[line,:] = image_sub[line,:]
    ground=np.mean(np.mean(noise_mat[:,0:10]))
    limit = std * np.mean(np.std(noise_mat,axis=0,ddof=1))
    length_finding = np.zeros(image_sub.shape)
    for line in np.arange(0,image_sub.shape[0],1):
        for col in np.arange(0,image_sub.shape[1],1):
            if image_sub[line][col] >= limit:
                length_finding[line][ col] = image_sub[line][col]
    return length_finding,limit,ground

def remove_noise_horizontal(image_sub,base,std):
    noise_mat=np.zeros((image_sub.shape[0],base))
    for col in np.arange(0,base,1):
        noise_mat[:,col] = image_sub[:,col]
    ground = np.mean(np.mean(noise_mat [0:10,:]))
    limit = std * np.mean(np.std(noise_mat,axis=0,ddof=1))
    length_finding = np.zeros(image_sub.shape)
    for line in np.arange(0,image_sub.shape[0],1):
        for col in np.arange(0,image_sub.shape[1],1):
            if image_sub[line,col] >= limit:
                length_finding[line, col] = image_sub[line,col]
    return length_finding,limit,ground

'''Tim chieu dai'''
def find_length(length_finding):
    mark = 0
    first_line=0
    last_line=0
    for line in np.arange(0,length_finding.shape[0],1):
        for col in np.arange(0,length_finding.shape[1],1):
            if length_finding[line][ col] > 0 and mark == 0:
                first_line = line
                mark = 1
            elif length_finding[line][ col] > 0 and mark == 1:
                last_line = line

    return  first_line,last_line

'''Tim chieu rong'''
def find_width(length_finding,limit):
    mark = 0
    mark1 = 0
    first_col=0
    last_col=0
    for col in np.arange(0,length_finding.shape[1],1):
        if (np.mean(length_finding[:,col]) > 0.05 * limit) and (mark == 0):
            first_col = col
            mark = 1
    for col in np.arange(0,length_finding.shape[1],1):
        if (np.mean(length_finding[:,col]) < 0.1 * limit) and (mark1 == 0) and (col > first_col):
            last_col = col - 1
            mark1 = 1
    return first_col,last_col

""" Tim 8 diem gia tri cao nhat"""
def find_8max(length_finding,peak):
    max = np.zeros((1, 8))
    max[0,0]=peak
    for num in np.arange(1,8,1):
        for lin in np.arange(0,length_finding.shape[0],1):
            for col in np.arange(0,length_finding.shape[1],1):
                if (length_finding[lin, col] >= max[0,num])and(length_finding[lin, col]<max[0,num-1]):
                    max[0, num]=length_finding[lin, col]
    return max