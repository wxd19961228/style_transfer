import os
import shutil
dir=r"D:\wxd\dataset\lfw"
dstdir = r"D:\wxd\dataset\human_face"
f=open("dir.txt","a")
for root,dirs,files in os.walk(dir):
    for file in files:
        f.writelines(os.path.join(root,file)+"\n")
        print(os.path.join(root,file))
        srcname = os.path.join(root,file)
        shutil.copy(srcname, dstdir)

