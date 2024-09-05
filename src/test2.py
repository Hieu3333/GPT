import torch
import os

files = ['Dataset/Haruki Murakami/1Q84.txt','Dataset/Haruki Murakami/Kafka on the shore.txt','Dataset/Haruki Murakami/Norwegian Wood.txt','Dataset/Haruki Murakami/The wind-up bird chronicle.txt']
output = 'Dataset/Haruki Murakami/HM.txt'

with open(output,'w',encoding='utf-8') as out:
    for file in files:
        if os.path.exists(file):
            with open(file,'r', encoding='utf-8') as input:
                out.write(input.read())
                print('concatenated'+file)
                
