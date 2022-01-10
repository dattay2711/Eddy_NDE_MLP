import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.utils import shuffle
from numpy import save
import process


def model1(image_comp,top,bottom,left,right,dis,base,std):
    image = np.abs(image_comp)
    img_bg = process.image_bg(image, top, bottom, left, right, dis)
    img_corr = process.image_resize(image, top, bottom, left, right)
    img_sub = process.image_remove_bg(img_bg, img_corr)
    peak,max_lin,max_col = process.image_max(img_sub)
    img_length_finding,limit,ground = process.remove_noise_vertical(img_sub,base,std)
    hole = np.mean(np.mean(img_sub[max_lin-5:max_lin+6,:]))
    hole1 = np.mean(np.std(img_sub[max_lin-5:max_lin+6,:],axis=1,ddof=1))
    return peak,hole,hole1,limit,ground

def model2(image_comp,top,bottom,left,right,dis,base,std):
    image = np.abs(image_comp)
    img_bg = process.image_bg(image, top, bottom, left, right, dis)
    img_corr = process.image_resize(image, top, bottom, left, right)
    img_sub = process.image_remove_bg(img_bg, img_corr)
    peak,max_lin,max_col = process.image_max(img_sub)
    img_length_finding,limit,ground = process.remove_noise_horizontal(img_sub,base,std)
    hole = np.mean(np.mean(img_sub[max_lin-5:max_lin+6,:]))
    hole1 = np.mean(np.std(img_sub[max_lin-5:max_lin+6,:],axis=1,ddof=1))
    return peak,hole,hole1,limit,ground

def load_data():
    name=["2A","2B","2E","2H","44A","45A","45C","45D"]
    path1="Dulieu-dong-xoay/1000"
    data=[]
    result=[]
    filename=[]
    for file in os.listdir(path1):
        filename.append(file)
    for i in range(1):
        image_imag = Image.open(os.path.join(path1,filename[1],name[1]+"_500kHz_"+str(200*2**i)+"u_imagpart.tif"))
        image_real = Image.open(os.path.join(path1,filename[1],name[1]+"_500kHz_"+str(200*2**i)+"u_realpart.tif"))
        img_imag = np.array(image_imag)
        img_real = np.array(image_real)
        img1_comp= img_real + 1j*img_imag
        inp=np.zeros(6)
        inp[0:5]=model1(img1_comp,10,10,3,1,15,20,1)
        inp[5]=0
        data.append(inp)

    for i in [1,2]:
        image_imag = Image.open(os.path.join(path1,filename[1],name[1]+"_500kHz_"+str(200*2**i)+"u_imagpart.tif"))
        image_real = Image.open(os.path.join(path1,filename[1],name[1]+"_500kHz_"+str(200*2**i)+"u_realpart.tif"))
        img_imag = np.array(image_imag)
        img_real = np.array(image_real)
        img2_comp= img_real + 1j*img_imag
        inp=np.zeros(6)
        inp[0:5]=model2(img2_comp,3,7,19,1,12,20,1)
        inp[5]=0
        data.append(inp)        
    for i in [0,4,5,2,3,6,7]:
        for j in range(1):
            image_imag = Image.open(os.path.join(path1,filename[i],name[i]+"_500kHz_"+str(200*2**j)+"u_imagpart.tif"))
            image_real = Image.open(os.path.join(path1,filename[i],name[i]+"_500kHz_"+str(200*2**j)+"u_realpart.tif"))
            img_imag = np.array(image_imag)
            img_real = np.array(image_real)
            img1_comp= img_real + 1j*img_imag
            inp=np.zeros(6)
            inp[0:5]=model1(img1_comp,10,10,3,1,15,20,1)
            inp[5]=0
            data.append(inp)
        for j in [1,2]:
            image_imag = Image.open(os.path.join(path1,filename[i],name[i]+"_500kHz_"+str(200*2**j)+"u_imagpart.tif"))
            image_real = Image.open(os.path.join(path1,filename[i],name[i]+"_500kHz_"+str(200*2**j)+"u_realpart.tif"))
            img_imag = np.array(image_imag)
            img_real = np.array(image_real)
            img2_comp= img_real + 1j*img_imag
            inp=np.zeros(6)
            inp[0:5]=model2(img2_comp,3,3,19,1,12,20,1)
            inp[5]=0
            data.append(inp)


    name=["2A","2B","2E","2H","44A","45A","45C","45D"]
    path2="Dulieu-dong-xoay/1003"
    filename=[]
    for file in os.listdir(path2):
        filename.append(file)
    for i in range(8):
        for j in range(1):
            image_imag = Image.open(os.path.join(path2,filename[i],name[i]+"_400kHz_"+str(200*2**j)+"u_imagpart.tif"))
            image_real = Image.open(os.path.join(path2,filename[i],name[i]+"_400kHz_"+str(200*2**j)+"u_realpart.tif"))
            img_imag = np.array(image_imag)
            img_real = np.array(image_real)
            img1_comp= img_real + 1j*img_imag
            inp=np.zeros(6)
            inp[0:5]=model1(img1_comp,10,10,3,1,15,20,1)
            inp[5]=1
            data.append(inp)
        for j in [1,2]:
            image_imag = Image.open(os.path.join(path2,filename[i],name[i]+"_400kHz_"+str(200*2**j)+"u_imagpart.tif"))
            image_real = Image.open(os.path.join(path2,filename[i],name[i]+"_400kHz_"+str(200*2**j)+"u_realpart.tif"))
            img_imag = np.array(image_imag)
            img_real = np.array(image_real)
            img2_comp= img_real + 1j*img_imag
            inp=np.zeros(6)
            inp[0:5]=model2(img2_comp,3,3,19,1,12,20,1)
            inp[5]=1
            data.append(inp)

    data = np.array(data)

    result = [[150],[350],[700],[200],[400],[750],[200],[350],[700],[150],[300],[700],[200],[400],[950],[200],[400],[900],[200],[400],[1050],[200],[400],[950],[200],[400],[750],[150],[350],[700],[200],[350],[700],[150],[300],[700],[200],[400],[950],[200],[400],[900],[200],[400],[1050],[200],[400],[950]]
    result = np.array(result )/1000
    data,result = shuffle(data,result, random_state=2)
    return data,result
