import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
import copy
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,128,kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,256,kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,512,kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512,512,kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
        )  
              
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(512,512,kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512,256,kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,256,kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256,128,kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,128,kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128,64,kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,64,kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,2,kernel_size=5, padding=1),
        )
        
    def fit(self, dataloaders, num_epochs):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        # Reduces learning rate when less progess is made
        scheduler = optim.lr_scheduler.StepLR(optimizer, 4)
        # Loss function
        criterion = nn.MSELoss()
        since = time.time()
        best_model_wts = copy.deepcopy(self.state_dict())
        best_acc = 0.0
        valid_loss_min = np.Inf

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            
            train_loss = 0.0
            valid_loss = 0.0
            
            valid_acc = 0
            
            # Set model to train
            self.train()
            
            for j, (inputs, true) in enumerate(dataloaders['train']):
                    # clear all gradients since gradients get accumulated after every iteration.
                    optimizer.zero_grad()
                    
                    outputs = self.forward(inputs)
                    # Calculates the loss                           
                    loss = criterion(outputs, true)
                    
                    # Backpropagation
                    loss.backward()
                    # Update parameters
                    optimizer.step()
                
                    train_loss += loss.item() * inputs.size(0)
                    
                    size = len(dataloaders['train'])
                    
                    train_loss = train_loss / size
                    
                    print(f'Epoch: {epoch}\t{100 * (j + 1) / size:.2f}% complete.\n', end='\r')
            
            # Do scheduler step after learning rate step
            scheduler.step()

            # Validation phase
            with torch.no_grad():
                # Set to evaluation mode
                self.eval()
                
                for inputs, labels in dataloaders['val']:                                
                    outputs = self.forward(inputs)
                    # Calculates the loss                          
                    loss = criterion(outputs, labels)

                    valid_loss += loss.item() * inputs.size(0)

                    size = len(dataloaders['val'])
                    
                    train_loss = train_loss / size

                # Calculate average losses
                train_loss = train_loss / len(dataloaders['train'])
                valid_loss = valid_loss / len(dataloaders['val'])
                    
                print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}')
                    
                # Copy the model if it has a better validation loss
                # if valid_loss < valid_loss_min:
                #     valid_loss_min = valid_loss
                #     best_acc = valid_acc
                #     best_model_wts = copy.deepcopy(self.state_dict())
                best_model_wts = copy.deepcopy(self.state_dict())
                    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        
        # Load best model parameters 
        self.load_state_dict(best_model_wts)
        torch.save(self.state_dict(), 'C:\\Users\\karee\\Desktop\\ChromaPy\\model1.pth')
        # Return model
        return self

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    