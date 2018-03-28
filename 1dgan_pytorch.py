import torch 
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


#seed = 7
#seed = 42
#np.random.seed(seed)
#torch.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=1000,
                        help='the number of training steps to take')
    parser.add_argument('--hidden-size', type=int, default=4,
                        help='MLP hidden size')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='the batch size')
    parser.add_argument('--learning-rate', type=float, default = 0.01,
                        help='learning rate')
    parser.add_argument('--save', action='store_true',
                        help='save on')
    return parser.parse_args()


class DataDistribution(object):
    def __init__(self,mu = 0,sigma = 1):
        self.mu = mu
        self.sigma = sigma

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N).astype(np.float32)
        samples.sort()
        return samples


class GeneratorDistribution(object):
    def __init__(self, range = 8):
        self.range = range

    def sample(self, N):
        return (np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01).astype(np.float32)



class generator(nn.Module):
      def __init__(self,input_size, hidden_size, output_size):
          super(generator, self).__init__()
          self.fc1 = nn.Linear(input_size, hidden_size)
          self.stp = nn.ReLU()
          self.fc2  = nn.Linear(hidden_size, output_size)
      def forward(self, x):
          out = self.fc1(x)
          out = self.stp(out)
          out = self.fc2(out)
          return out

class discriminator(nn.Module):
      def __init__(self, input_size, hidden_size):
          super(discriminator,self).__init__()
          self.fc1 = nn.Linear(input_size, hidden_size)
          self.relu= nn.ReLU()
          self.fc2 = nn.Linear(hidden_size, hidden_size)
          self.fc3 = nn.Linear(hidden_size,1)
          self.sg  = nn.Sigmoid()
      def forward(self,x):
          out = self.fc1(x)
          out = self.relu(out)
          out = self.fc2(out)
          out = self.relu(out)
          out = self.fc3(out)
          out = self.sg(out)
          return out

class GAN(object):
      def __init__(self,params):
          self.g = generator(1,params.hidden_size,1)
          self.d = discriminator(1,params.hidden_size)
          self.batch_size = params.batch_size
          self.lr = params.learning_rate
          self.ct = nn.BCELoss()
          self.g_opt = torch.optim.SGD(self.g.parameters(),lr = self.lr)
          self.d_opt = torch.optim.SGD(self.d.parameters(),lr = self.lr)
          self.epoch = params.num_steps

def train(model,dd,gd,save):
    
    ones = Variable(torch.ones(model.batch_size,1))
    zeros = Variable(torch.zeros(model.batch_size,1))
    for i in range(model.epoch):
       ds = Variable(torch.from_numpy(dd.sample(model.batch_size)).view(-1,1))
       gs = Variable(torch.from_numpy(gd.sample(model.batch_size)).view(-1,1))

       model.d_opt.zero_grad()
       model.g_opt.zero_grad()
       d1 = model.d(ds)
       g = model.g(gs)
       d2 = model.d(g)
      
       loss_d1 = model.ct(d1,ones)
       loss_d2 = model.ct(d2,zeros)

       loss = loss_d1 + loss_d2
       loss.backward(retain_graph=True)
       model.d_opt.step()

       model.d_opt.zero_grad()
       model.g_opt.zero_grad()
        
       loss_g = model.ct(d2,ones)
       loss_g.backward()
       model.g_opt.step()
       if save :
        drawplot(model,dd,model.batch_size,gd,i) 
'''
m  =model
dd =data distribution
bs =batch size
zd =z distribution
'''
def drawplot( m, dd,bs,zd,i,sr = 10,nps=10000,nb=100):
    xs = np.linspace(-sr, sr, nps).astype(np.float32)
    bins = np.linspace(-sr, sr, nb).astype(np.float32)
  
    d = dd.sample(nps)
    pd, _ = np.histogram(d, bins=bins, density=True)

    p_x = np.linspace(-sr, sr, len(pd))
    f, ax = plt.subplots(1)
    v = Variable(torch.from_numpy(xs).view(-1,1))
    db = m.d(v)
    db = db.data.numpy()

    z = zd.sample(nps)
    zv = Variable(torch.from_numpy(z).view(-1,1))
    gd = m.g(zv)
    gd = gd.data.numpy()

    pgh, _ = np.histogram(gd,bins=bins,density=True) 


    ax.plot(xs, db, label='decision boundary')
    ax.set_ylim(0, 1)

    plt.plot(p_x, pd, label='real data')
    plt.plot(p_x, pgh, label='generated data')

    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    plt.savefig("1dgan_figure:%s" %i)
    plt.close()




def main(args):
    model = GAN(args)
    dd = DataDistribution(0,0.5)
    gd = GeneratorDistribution()
    train(model, dd, gd,args.save)
    drawplot(model,dd,model.batch_size,gd,'result') 


if __name__ == '__main__':
    main(parse_args())
