import glob
localpath="/nfs/tq-data3/Datasets/sparc/sparcs_data_L8/*.tif"
files=sorted(glob.glob(localpath))
for i in range(0,len(files),2):
    print(files[i])
    print(files[i+1])