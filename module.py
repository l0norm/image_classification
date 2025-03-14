import torch 
from torch import nn
import torch.nn.functional as F 
from torchvision import transforms, datasets
import matplotlib.pyplot as plt 



transform = transforms.Compose([transforms.ToTensor()])


train_data = datasets.CIFAR10(root='./data',train=True,download=True,
                               transform=transform)
test_data = datasets.CIFAR10(root='./data',train=False,download=True,
                               transform=transform)

# batches 
train_loader = torch.utils.data.DataLoader(train_data,
                                         batch_size=32,
                                         shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,
                                         batch_size=32,
                                         shuffle=True)




class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,16,3,1,1)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,3,1,1)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout2d(0.25)

        self.conv3 = nn.Conv2d(32,64,3,1,1)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,128,3,1,1)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2,2)
        self.dropout2 = nn.Dropout2d(0.25)

        self.conv5 = nn.Conv2d(128,256,3,1,1)
        self.batch_norm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256,512,3,1,1)
        self.batch_norm6 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2,2)
        self.dropout3 = nn.Dropout2d(0.25)


        self.fc1 = nn.Linear(512*4*4,128)
        self.dropout4 = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)




    def forward(self,x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
  

        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.batch_norm5(self.conv5(x)))
        x = F.relu(self.batch_norm6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)
    
