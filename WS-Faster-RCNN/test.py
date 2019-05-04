from pyhdf.SD import SD,SDC
import pprint
import matplotlib.pyplot as plt
import numpy as np
HDF_DIR='/media/winshare/98CA9EE0CA9EB9C8/Hackthon/Data/MOD44B.A2000065.h28v09.006.2017081110435.hdf'
HDF_DIR2='/media/winshare/98CA9EE0CA9EB9C8/Hackthon/Data/MOD44B.A2016065.h28v09.006.2017081145329.hdf'
file=SD(HDF_DIR)
file2=SD(HDF_DIR2)
print(file.info())

datasets=file.datasets()
datasets2=file.datasets()

for idx ,sds in enumerate(datasets.keys()):
    print(idx,sds)

percentage_obj=file.select('Percent_Tree_Cover')
data=percentage_obj.get()
percentage_obj2=file2.select('Percent_Tree_Cover')
data2=percentage_obj2.get()

print(percentage_obj.get())
print(type(data))
data[data==200]=0
data2[data2==200]=0
data3=np.zeros(data.shape,dtype=np.float)
data3[:,:]=abs(data2[:,:]/100)-(data[:,:]/100)
# data3[data3<0]=0

fig,ax=plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True)
ax[0].imshow(data)
ax[1].imshow(data2)
ax[2].imshow(data3)
plt.show()