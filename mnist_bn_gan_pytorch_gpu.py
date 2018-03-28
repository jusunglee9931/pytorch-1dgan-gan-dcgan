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
                            transform=transforms.ToTensor(),  
                            download=True)




seed = 11
#seed = 42
np.random.seed(seed)
torch.manual_seed(seed)



class DataDistribution(object):
    def __init__(self,mu = 0,sigma = 1):
        self.mu = mu
        self.sigma = sigma

    def sample(self, in_s,bs):
        samples = np.random.uniform(-1. , 1.,(bs,in_s)).astype(np.float32)
       # samples.sort()
        return samples


class GeneratorDistribution(object):
    def __init__(self, range = 8):
        self.range = range

    def sample(self, N):
        return (np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=100,
                        help='the number of training steps to take')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='MLP hidden size')
    parser.add_argument('--batch-size', type=int, default=25,
                        help='the batch size')
    parser.add_argument('--learning-rate', type=float, default = 0.0002,
                        help='learning rate')
    parser.add_argument('--z-size', type=int, default=100,
                        help='the z size')
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
    plt.savefig("mnist_gan_loss_epoch%d" %e)
    plt.close()



class generator(nn.Module):
      def __init__(self,input_size, hidden_size, output_size):
          super(generator, self).__init__()
          self.fc1 = nn.Linear(input_size, hidden_size)
          self.bn1= nn.BatchNorm1d(hidden_size)
          self.stp = nn.ReLU()
          self.fc2  = nn.Linear(hidden_size,2* hidden_size)
          self.bn2= nn.BatchNorm1d(2* hidden_size)
          self.fc3 = nn.Linear(2*hidden_size,output_size)
          self.bn3= nn.BatchNorm1d(output_size)
          self.sg  = nn.Sigmoid()
  
      def forward(self, x):
          out = self.fc1(x)
          out = self.bn1(out)
          out = self.stp(out)
          out = self.fc2(out)
          out = self.bn2(out)
          out = self.stp(out)
          out = self.fc3(out)
          out = self.bn3(out)
          out = self.sg(out)
          return out

class discriminator(nn.Module):
      def __init__(self, input_size, hidden_size):
          super(discriminator,self).__init__()
          self.fc1 = nn.Linear(input_size, hidden_size)
          self.relu= nn.ReLU()
          self.bn1= nn.BatchNorm1d(hidden_size)
          self.fc2 = nn.Linear(hidden_size, int(hidden_size/2))
          self.bn2= nn.BatchNorm1d( int(hidden_size/2))
          self.fc3 = nn.Linear( int(hidden_size/2),1)
          self.sg  = nn.Sigmoid()
      def forward(self,x):
          out = self.fc1(x)
          out = self.bn1(out)
          out = self.relu(out)
          out = self.fc2(out)
          out = self.bn2(out)
          out = self.relu(out)
          out = self.fc3(out)
          out = self.sg(out)
          return out

class GAN(object):
      def __init__(self,params,in_s,in_g):
          self.in_g = in_g
          self.g = generator(in_g,params.hidden_size,in_s)
          self.d = discriminator(in_s,params.hidden_size)
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
    
    ones = Variable(torch.ones(model.batch_size,1).cuda())
    zeros = Variable(torch.zeros(model.batch_size,1).cuda())
    a_loss_g = []
    a_loss_d = []
    for i in range(model.epoch):
     e_loss_g = 0
     e_loss_d = 0
     print("epoch :%s" %i)
     
     for m,(image,label) in enumerate(trl):
       ds = Variable(image.view(-1,28*28).cuda())
       gs = Variable(torch.from_numpy(gd.sample(model.in_g,model.batch_size)).cuda())
       #print(ds[0])
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

       e_loss_g += loss_g.data[0]
       e_loss_d += loss.data[0]
       
     a_loss_g.append(e_loss_g/trl.__len__())
     a_loss_d.append(e_loss_d/trl.__len__())
     drawlossplot(model,a_loss_g,a_loss_d,i)
     generateimage(model,gd,i,0)

def generateimage(m,gd,e,new):
       if new :
        gs = Variable(torch.from_numpy(gd.sample(m.in_g,m.batch_size)).cuda())
        g = m.g(gs)
       else :
        g = m.g(g_gs)
       g = g.data.cpu().numpy()
       g = g.reshape(m.batch_size,28,28)
       fig = plt.figure(figsize=(28,28), tight_layout = True)
       grid = math.sqrt(m.batch_size)
       for i in range(m.batch_size):
        ax = fig.add_subplot(grid,grid,i+1)
        ax.set_axis_off()
        plt.imshow(g[i],shape=(28,28),cmap='Greys_r')
       title = 'Epoch {0}'.format(e+1)
       fig.text(0.5, 0.04, title, ha='center' )
       plt.savefig("mnist_gan_epoch%s" %e)
       plt.close()




'''
global variable for draw....
'''
g_arg = parse_args()
g_gd = DataDistribution(0,1)
g_gs = Variable(torch.from_numpy(g_gd.sample(g_arg.z_size,g_arg.batch_size)).cuda())



def main(args):
    trl = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=args.batch_size, 
                                           shuffle=True)



    model = GAN(args,28*28,args.z_size)
    dd = DataDistribution(0,1)
    
    #model.load()
    if args.eval:
     for i in range(10):
      generateimage(model,dd,i)
    else: 
     train(model, trl, dd)    
   
    
    if args.save:
     model.save()

if __name__ == '__main__':
    main(parse_args())















