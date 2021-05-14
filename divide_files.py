import os
import shutil
from tqdm import tqdm

j = 0
os.mkdir('data_' + str(j))
for i, file_name in tqdm(enumerate(os.listdir('data_updated'))):
    if (i+1) % 200 == 0:
        j += 1
        os.mkdir('data_' + str(j))
    shutil.move('data_updated'+'/'+file_name, 'data_'+str(j)+'/'+file_name)
