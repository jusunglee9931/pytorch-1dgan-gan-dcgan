import os
import torch
from PIL import Image
import torchvision.transforms as transforms
 
mypath = '/home/jusunglee/vision_db/celebA/img_align_celeba/'
#print(mypath)
#onlyfiles = [f for f in os.listdir(mypath)]
#print(onlyfiles)

class celebaldr(object):
    def __init__(self,path,batch_size):
         self.path = path
         self.bn   = batch_size
         self.cnt  = 0
         self.getlist()
    def getlist(self):
         self.list = [f for f in os.listdir(self.path)]
         self.lens = len(self.list)
         print("len : %s" %self.lens)
    def getbn(self):
         if self.cnt+self.bn >= self.lens:
          self.cnt = 0
          return 0 , 0
         else :
          
          batch_img = torch.FloatTensor()
          trans = transforms.ToTensor()
          for i in range(self.bn):
           file = self.path + self.list[i+self.cnt]
           im = Image.open(file)
           im=im.crop((60,80,60+64,80+80))
           im=im.resize((64,64))
           #im.show()
           batch_img = torch.cat((trans(im).unsqueeze(0),batch_img))
         self.cnt += self.bn
#         print(batch_img)
         return batch_img,1


           

#cel = celebaldr(mypath,25)
#cel.getbn()
#cel.getbn()
     
