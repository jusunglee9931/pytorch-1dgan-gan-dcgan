import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import math


train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.Compose([
                                                          transforms.Resize(64)
                                                          ]),  
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False,
                           transform=transforms.ToTensor())

class DataDistribution(object):
    def __init__(self,mu = 0,sigma = 1):
        self.mu =mu 
        self.sigma =sigma
    def sample(self, z_size,bs):
        samples = np.random.uniform(-1. , 1., (bs,z_size,1,1)).astype(np.float32)
       # samples.sort()
        return samples




seed = 2
#seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

'''
variables
'''
x_size =64
y_size =64
in_ch = 1
z_size =100
g_gd = DataDistribution()
g_gs = Variable(torch.from_numpy(g_gd.sample(100,25)).cuda())


print(torch.cuda.get_device_name(0))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=100,
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=25,
                        help='the batch size')
    parser.add_argument('--learning-rate', type=float, default = 0.0002,
                        help='learning rate')
    parser.add_argument('--eval', action='store_true',
                        help='eval mode')
    parser.add_argument('--save', action='store_true',
                        help='save on')
    return parser.parse_args()



'''
m  =model

'''
def drawlossplot( m,loss_g,loss_d,e):
    print(loss_g)
    g_x = np.linspace(0, len(loss_g), len(loss_g))
    f, ax = plt.subplots(1)
   
    plt.plot(g_x, loss_g, label='loss_g')
    plt.plot(g_x, loss_d, label='loss_d')
    ax.set_xlim(0, m.epoch)

    plt.title('Vanilla Generative Adversarial Network Loss Graph')
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.legend()
    plt.savefig("mnist_dcgan_loss_epoch%d" %e)
    plt.close()

def convblocklayer(in_ch,out_ch):
    return nn.Sequential(nn.Conv2d(in_ch,out_ch,kernel_size =4, stride = 2,padding = 1),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU()
                         )
def deconvblocklayer(in_ch,out_ch,pad):
    return nn.Sequential(nn.ConvTranspose2d(in_ch,out_ch,kernel_size = 4, stride = 2,padding = pad),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU()
                         )

class generator(nn.Module):
      def __init__(self,z_channel,o_channel):
          super(generator, self).__init__()
          self.layer1= deconvblocklayer(z_channel,1024,0)
          self.layer2= deconvblocklayer(1024,512,1)
          self.layer3= deconvblocklayer(512,256,1)
          self.layer4= deconvblocklayer(256,128,1)
          self.conv5 = nn.ConvTranspose2d(128,o_channel, kernel_size =4,stride = 2, padding =1)
          self.sg    = nn.Sigmoid()
      def forward(self, x):
          out = self.layer1(x)
          out = self.layer2(out)
          out = self.layer3(out)
          out = self.layer4(out)
          out = self.conv5(out)
          out = self.sg(out)
          return out

class discriminator(nn.Module):
      def __init__(self, channel):
          super(discriminator,self).__init__()
          self.layer1 =convblocklayer(channel,128)
          self.layer2 =convblocklayer(128,256)
          self.layer3 =convblocklayer(256,512)
          self.layer4 =convblocklayer(512,1024)
          self.conv5 = nn.Conv2d(1024,1, kernel_size =4)
          self.sg = nn.Sigmoid()
      def forward(self,x):
          out = self.layer1(x)
          out = self.layer2(out)
          out = self.layer3(out)
          out = self.layer4(out)
          out = self.conv5(out)
          out = self.sg(out)
          return out


class GAN(object):
      def __init__(self,params,in_ch,o_ch):
          self.g = generator(z_size,o_ch)
          self.d = discriminator(in_ch)
          self.g.cuda(0)
          self.d.cuda(0)
          self.batch_size = params.batch_size
          self.lr = params.learning_rate
          self.ct = nn.BCELoss()
          self.g_opt = torch.optim.Adam(self.g.parameters(),lr = self.lr)
          self.d_opt = torch.optim.Adam(self.d.parameters(),lr = self.lr)
          self.epoch = params.num_steps
      def save(self):
          torch.save(self.g,"g.pt")
          torch.save(self.d,"d.pt")
      def load(self):
          self.g = torch.load("g.pt")
          self.d = torch.load("d.pt")

def train(model,trl,gd):
    
    ones = Variable(torch.ones(model.batch_size,1,1,1).cuda())
    zeros = Variable(torch.zeros(model.batch_size,1,1,1).cuda())
    batch_img = torch.FloatTensor()
    trans = transforms.ToTensor()
    a_loss_g = []
    a_loss_d = []
    for i in range(model.epoch):
     e_loss_g = 0
     e_loss_d = 0
     
     print("epoch :%s" %i)
     for m,(images,label) in enumerate(trl):
       batch_img = torch.cat((trans(images).unsqueeze(0),batch_img))
       #torch.stack((batch_img,trans(images)),out=batch_img)
        
       if (m+1) % model.batch_size == 0 :
        #print(batch_img.size()) 
        ds = Variable(batch_img.cuda())
        gs = Variable(torch.from_numpy(gd.sample(z_size,model.batch_size)).cuda())

        model.d_opt.zero_grad()

        d1 = model.d(ds)
        g = model.g(gs)
        d2 = model.d(g)
      
        loss_d1 = model.ct(d1,ones)
        loss_d2 = model.ct(d2,zeros)

        loss = loss_d1 + loss_d2
        loss.backward(retain_graph=True)
        model.d_opt.step()


        model.g_opt.zero_grad()
        
        loss_g = model.ct(d2,ones)
        loss_g.backward()
        model.g_opt.step()
        #print("loss_d :%s loss_g:%s" %(loss,loss_g))
        batch_img = torch.FloatTensor()
        e_loss_g += loss_g.data[0]
        e_loss_d += loss.data[0]
     a_loss_g.append(e_loss_g/trl.__len__())
     a_loss_d.append(e_loss_d/trl.__len__())
     drawlossplot(model,a_loss_g,a_loss_d,i)
     generateimage(model,gd,i,0)  
  

def generateimage(m,gd,e,new=False):
       if new :
        gs = Variable(torch.from_numpy(gd.sample(z_size,m.batch_size)).cuda())
        g = m.g(gs)
       else :
        g = m.g(g_gs)
       g = g.data.cpu().numpy()
       g = g.reshape(m.batch_size,y_size,x_size)
       #print(g)
       fig = plt.figure(figsize=(y_size,x_size),tight_layout=True)
       grid = math.sqrt(m.batch_size)
       for i in range(m.batch_size):
        ax = fig.add_subplot(grid,grid,i+1)
        ax.set_axis_off()
        plt.imshow(g[i],shape=(64,64),cmap='Greys_r')
       plt.savefig("dc_gan_figure_epoch%s" %e)
       plt.close()

def drawimage(bn,trl):
    batch = torch.FloatTensor(bn,64,64)
    print(batch)
    for i,(image,label) in enumerate( trl):
      trans = transforms.ToTensor()
      batch +=trans(image)
      if i > 1: 
        image.show()
        print(batch)
        break
     #plt.imshow(image)
     #plt.show()
    # break

def main(args):
   
    model = GAN(args,1,1)
    dd = DataDistribution(0,1)
    #model.load()
    if args.eval:
     for i in range(10):
      generateimage(model,dd,i)
    else: 
     train(model, train_dataset, dd)    
   
    
    if args.save:
     model.save()

if __name__ == '__main__':
    main(parse_args())










