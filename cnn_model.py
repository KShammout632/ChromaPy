import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(64,128,kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(128,256,kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(256,512,kernel_size=5),
            nn.ReLU(True)
        )  
              
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(512,256,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,2,kernel_size=5),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    