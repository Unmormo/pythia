#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pylab as plt
import os
from zuobiao import*




        
def NPconvert_to_vtk(data,path,x1,y1,x4,y4,h1,h2):#转npy为vtk格式,x1,y1,x4,y4,h1,h2为顶点的绝对坐标

        # - preparatory steps 

        
        N = data.shape[0]*data.shape[1]*data.shape[2]   

        # - open file and write header

        fid = open(path, 'w')



        fid.write('# vtk DataFile Version 3.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET UNSTRUCTURED_GRID\n')

        # - write grid points

        fid.write('POINTS '+str(N)+' float\n')
        for i in np.arange(data.shape[0]):
            for j in np.arange(data.shape[1]):
                for k in np.arange(data.shape[2]):
                    
                    #x = self.m[n].lat[i]
                    #y = self.m[n].lon[j]
                    #z = (self.m[n].r[nz[n] - 1 - k] - self.m[n].r[0]) / 100
                    z,x,y=xiangdui2zhenshi(data,i,j,k,x1,y1,x4,y4,h1,h2)

                    fid.write(str(x)+' '+str(y)+' '+str(z)+'\n')

        
        # - write connectivity

        n_cells = 0

        n_cells = n_cells+(data.shape[0]-1)*(data.shape[1]-1)*(data.shape[2]-1)

        fid.write('\n')
        fid.write('CELLS '+str(n_cells)+' '+str(9*n_cells)+'\n')


        for i in np.arange(1, data.shape[0]):
            for j in np.arange(1, data.shape[1]):
                for k in np.arange(1, data.shape[2]):
                    # i j k
                    a = k+(j-1)*data.shape[2]+(i-1) *                         data.shape[1]*data.shape[2]-1     	# 0 0 0
                    b = k+(j-1)*data.shape[2]+(i-1) *                         data.shape[1]*data.shape[2]       	# 0 0 1
                    c = k+(j)*data.shape[2]+(i-1)*data.shape[1] *                         data.shape[2]-1       	# 0 1 0
                    d = k+(j)*data.shape[2]+(i-1)*data.shape[1] *                         data.shape[2]        	# 0 1 1
                    e = k+(j-1)*data.shape[2]+(i)*data.shape[1] *                         data.shape[2]-1       	# 1 0 0
                    f = k+(j-1)*data.shape[2]+(i)*data.shape[1] *                         data.shape[2]         	# 1 0 1
                    g = k+(j)*data.shape[2]+(i)*data.shape[1] *                         data.shape[2]-1         	# 1 1 0
                    h = k+(j)*data.shape[2]+(i)*data.shape[1] *                         data.shape[2]           	# 1 1 1

                    fid.write('8 '+str(a)+' '+str(b)+' '+str(c)+' '+str(d) +
                                  ' '+str(e)+' '+str(f)+' '+str(g)+' '+str(h)+'\n')


        # - write cell types

        fid.write('\n')
        fid.write('CELL_TYPES '+str(n_cells)+'\n')



        for i in np.arange(data.shape[0]-1):
            for j in np.arange(data.shape[1]-1):
                for k in np.arange(data.shape[2]-1):

                    fid.write('11\n')

        # - write data

        fid.write('\n')
        fid.write('POINT_DATA '+str(N)+'\n')
        fid.write('SCALARS scalars int\n')
        fid.write('LOOKUP_TABLE mytable\n')





        idx = np.arange(data.shape[0])
        idx[data.shape[0]-1] = data.shape[0]-1

        idy = np.arange(data.shape[1])
        idy[data.shape[1]-1] = data.shape[1]-1

        idz = np.arange(data.shape[2])
        idz[data.shape[2]-1] = data.shape[2]-1
        print(idx,idy,idz)
        for i in idx:
            for j in idy:
                for k in idz:

                    fid.write(str(data[i,j,k])+'\n')

        # - clean up

        fid.close()
        
        
    
data=np.load('./output/Ti.npy')
#data=simgridex(data,0.1)
data = data.transpose(0, 2, 1) 
data=data[:,:,:]

path='./output/jiangtailuTI.vtk'
#y1=24227.759
#y4=24315.447

#x4=39083.771
#x1=38933.433
y1=0
y4=88

x4=150
x1=0
h1=16
h2=-56
NPconvert_to_vtk(data,path,x1,y1,x4,y4,h1,h2)


data=np.load('./output/reconstruction2.npy')
data = data.transpose(0, 2, 1) 
data=data[:,:,:]
path='./output/outputfinalzuobiao.vtk'

y1=0
y4=88

x4=150
x1=0
#y1=24227.759
#y4=24315.447

#x4=39083.771
#x1=38933.433
h1=16
h2=-56

h1=16
h2=-56
NPconvert_to_vtk(data,path,x1,y1,x4,y4,h1,h2)
print('done')

