import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

from torchdiffeq import odeint_adjoint as odeint
from .wavelet_convolution import WaveConvSim2d
from .unet2d import U_net



class Encoder(nn.Module):
    def __init__(self, width):
        super(Encoder, self).__init__()
        self.width = width
        self.fc0 = nn.Linear(5, self.width)

    def forward(self, x):
        x = F.pad(x, (1, 1, 0, 0), "replicate")
        x = x.permute(0, 3, 2, 1) 
        x = self.fc0(x)  
        x = x.permute(0, 3, 1, 2) 
        
        return x
    
    
class Decoder(nn.Module):

    def __init__(self, width, level, size, wavelet, modes1, modes2):
        super(Decoder, self).__init__()

        self.level = level
        self.width = width
        self.size = size
        self.wavelet = wavelet
        self.modes1 = modes1
        self.modes2 = modes2
        
        self.conv2 = WaveConvSim2d(self.width, self.width, self.level, self.size, self.wavelet)
        self.conv3 = WaveConvSim2d(self.width, self.width, self.level, [72,512], self.wavelet)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.unet2 = U_net(self.width, self.width, 3, 0)
        self.unet3 = U_net(self.width, self.width, 3, 0)
        self.linear2 = nn.Linear(1000, 512)
        self.linear3 = nn.Linear(512, 70)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x3 = self.unet2(x)
        x = x1 + x2 + x3
        x = self.linear2(x)
        x = F.mish(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, 512)
        x3 = self.unet3(x)
        x = x1 + x2 + x3
        x = self.linear3(x)
        x = F.mish(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.mish(x)
        x = self.fc2(x)

        x = x.view(batchsize, 1, size_x, 70)[:, :, 1:-1, :]

        return x
    

class HDerivative(nn.Module):
    def __init__(self, num_param, width, level, size, wavelet, modes1, modes2):
        super().__init__()

        self.num_param = num_param
        self.level = level
        self.width = width
        self.size = size
        self.wavelet = wavelet
        self.modes1 = modes1
        self.modes2 = modes2

        
        self.conv0 = WaveConvSim2d(self.width, self.width, self.level, self.size, self.wavelet)
        self.conv1 = WaveConvSim2d(self.width, self.width, self.level, self.size, self.wavelet)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.unet = U_net(self.width, self.width, 3, 0)
        self.linear = nn.Linear(1000, 1000)

        self.A = nn.Parameter(torch.empty(self.num_param, self.width)) 
        self.B = nn.Parameter(torch.empty(self.num_param, self.width)) 
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.B, a=math.sqrt(5))


    def forward(self, x, param):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x3 = torch.einsum('bi,io->bo', param, self.A)
        x3 = x3.unsqueeze(2).unsqueeze(2)
        x = x1 + x2 + x3
        x = F.mish(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x3 = self.unet(x)
        x4 = torch.einsum('bi,io->bo', param, self.B)
        x4 = x4.unsqueeze(2).unsqueeze(2)
        x = x1 + x2 + x3 + x4
        x = self.linear(x)
        x = F.mish(x)

        return x



class HIWaveletDeepONet(nn.Module):
    def __init__(
            self,
            num_parameter,
            level,
            size,
            wavelet,
            width=64,
            modes1=20,
            modes2=20
    ):
        super().__init__()
        self.num_parameter = num_parameter     
        self.level = level
        self.width = width
        self.size = size
        self.wavelet = wavelet
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.encoder = Encoder(self.width)
        self.func = Derivative( self.num_parameter, self.width, self.level, self.size, self.wavelet, self.modes1, self.modes2)
        self.decoder = Decoder(self.width, self.level, self.size, self.wavelet, self.modes1, self.modes2)
        self.b = nn.Parameter(torch.tensor(0.0))

        self.t = torch.linspace(0, 0.1, 7).float().cuda()

    def forward(self, inputs):
        x = self.encoder(inputs[0])
        x = x + self.b
        
        self.func.param = inputs[1]
        u = odeint(self.func, y0=x, t=self.t, method='rk4')[-1,...]

        x = self.decoder(u)
        return x
    