import torch
import pandas as pd
import numpy as np
import os,glob,csv,random
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
from torch import nn

class NumbersDataset(Dataset):
    def __init__(self,root,resize,mode):
        super(NumbersDataset,self).__init__()
        self.root=root#image save root
        self.resize=resize#image resize
        self.name2label={}#real label
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root,name)):
                continue
            self.name2label[name]=len(self.name2label.keys())
        # print(self.name2label)
        self.images,self.labels=self.load_csv('images.csv')
        if mode=='train':
            self.images=self.images[:int(0.6*len(self.images))]
            self.labels=self.labels[:int(0.6*len(self.labels))]
        elif mode=='val':
            self.images = self.images[int(0.6 * len(self.images)):int(0.8*len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8*len(self.labels))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]
    def load_csv(self,filename):
        if not os.path.exists(os.path.join(self.root,filename)):
            images=[]
            for name in self.name2label.keys():
                images+=glob.glob(os.path.join(self.root,name,'*.png'))
                images+=glob.glob(os.path.join(self.root,name,'*.jpg'))
                images+=glob.glob(os.path.join(self.root,name,'*.jpeg'))
            print(len(images),images)
            random.shuffle(images)
            #save datas
            with open(os.path.join(self.root,filename),mode='w',newline='') as f:
                write=csv.writer(f)
                for img in images:
                    name=img.split(os.sep)[-2]
                    label=self.name2label[name]
                    write.writerow([img,label])
                print('writen into csv file:',filename)
        #read datas
        labels,images=[],[]
        with open(os.path.join(self.root,filename)) as f:
            reader=csv.reader(f)
            for row in reader:
                img,label=row
                label=int(label)
                images.append(img)
                labels.append(label)
            assert len(images)==len(labels)
        # print(len(images), images)
        return images,labels

    def __len__(self):
        return len(self.images)
    def denormalize(self,x):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # x=(x-mean)/std
        # x^=x_hat*std+mean
        mean=torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std=torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x=x*std+mean
        return x

    def __getitem__(self, item):
        #item==0~len(images)
        img,label=self.images[item],self.labels[item]
        #img is root,label is int
        tf=transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),
            transforms.Resize((int(self.resize*1.2),int(self.resize*1.2))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
            ])
        img=tf(img)
        label=torch.tensor(label)
        return img,label
def main():
    import visdom
    import time
    viz=visdom.Visdom()
    pk=NumbersDataset('pokeman',64,'train')
    x,y=next(iter(pk))
    # print('Sample',x.shape,y.shape,y)
    viz.images(pk.denormalize(x),win='samplt_x',opts=dict(title='sample_x'))
    loader=DataLoader(pk,batch_size=32,shuffle=True)
    for x,y in loader:
        viz.images(pk.denormalize(x),nrow=8,win='batch',opts=dict(title='batch'))
        viz.text(str(y.numpy()),win='label',opts=dict(title='batch_y'))
        time.sleep(3)
if __name__ == '__main__':
    main()
