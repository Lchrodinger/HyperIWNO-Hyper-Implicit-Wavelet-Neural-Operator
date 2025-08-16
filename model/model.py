import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .wavelet_convolution import WaveConv2d
from .unet2d import U_net


""" WNO """
class WNO2d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, omega, padding=0):
        super(WNO2d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 3-channel tensor, Initial input and location (a(x,y), x,y)
              : shape: (batchsize * x=width * x=height * c=3)
        Output: Solution of a later timestep (u(x,y))
              : shape: (batchsize * x=width * x=height * c=1)
        
        Input parameters:
        -----------------
        width     : scalar, lifting dimension of input
        level     : scalar, number of wavelet decomposition
        layers    : scalar, number of wavelet kernel integral blocks
        size      : list with 2 elements (for 2D), image size
        wavelet   : string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: list with 2 elements (for 2D), right supports of 2D domain
        omega     : scalar, controls the number of Fourier modes used in the spectral convolution
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet = wavelet
        self.omega = omega
        self.in_channel = in_channel
        self.grid_range = grid_range 
        self.padding = padding
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 3: (a(x, y), x, y)
        for i in range( self.layers ):
            self.conv.append( WaveConv2d(self.width, self.width, self.level, size=self.size, 
                                         wavelet=self.wavelet, omega=self.omega) )
            self.w.append( nn.Conv2d(self.width, self.width, 1) )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)    
        x = self.fc0(x)                      # Shape: Batch * x * y * Channel
        x = x.permute(0, 3, 1, 2)            # Shape: Batch * Channel * x * y
        if self.padding != 0:
            x = F.pad(x, [0,self.padding, 0,self.padding]) 
        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.mish(x)                # Shape: Batch * Channel * x * y
                
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]     
        x = x.permute(0, 2, 3, 1)            # Shape: Batch * x * y * Channel
        x = F.mish( self.fc1(x) )            # Shape: Batch * x * y * Channel
        x = self.fc2(x)                      # Shape: Batch * x * y * Channel
        return x
    
    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    



class Decoder(nn.Module):
    def __init__(self, width, level, size, wavelet, modes1, modes2):
        super(Decoder, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 5-channel tensor, Initial input and location
              : shape: (batchsize * x=width * x=height * c=5)
        Output: Solution of a later timestep (u(x,y))
              : shape: (batchsize * x=width * x=height * c=1)
        
        Input parameters:
        -----------------
        width     : scalar, lifting dimension of input
        level     : scalar, number of wavelet decomposition
        layers    : scalar, number of wavelet kernel integral blocks
        size      : list with 2 elements (for 2D), image size
        wavelet   : string, wavelet filter
        """

        self.level = level
        self.width = width
        self.size = size
        self.wavelet = wavelet
        self.modes1 = modes1
        self.modes2 = modes2
        
        self.conv0 = WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet, modes1=self.modes1, modes2=self.modes2)
        self.conv1 = WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet, modes1=self.modes1, modes2=self.modes2)
        self.conv2 = WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet, modes1=self.modes1, modes2=self.modes2)
        self.conv3 = WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet, modes1=self.modes1, modes2=self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.unet1 = U_net(self.width, self.width, 3, 0)
        self.unet2 = U_net(self.width, self.width, 3, 0)
        self.unet3 = U_net(self.width, self.width, 3, 0)
        self.linear1 = nn.Linear(1000, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 70)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.mish(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x3 = self.unet1(x)
        x = x1 + x2 + x3
        x = self.linear1(x)
        x = F.mish(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, 512)
        x3 = self.unet2(x)
        x = x1 + x2 + x3
        x = self.linear2(x)
        x = F.mish(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, 256)
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
   



    

