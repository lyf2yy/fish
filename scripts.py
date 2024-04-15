import os
import shutil

path = r'F:\pr_tmp\ultralytics_fish\data\images\test'

files = os.listdir(path)
dst = r'F:\pr_tmp\ultralytics_fish\data\labels'
for it in files:
    shutil.copyfile(os.path.join(dst, it.replace('.JPG', '.txt')), os.path.join(dst, 'test', it.replace('.JPG', '.txt')))


