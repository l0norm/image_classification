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
                                         batch_size=128,
                                         shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,
                                         batch_size=128,
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
    


train_losses = []
test_losses = []


def train_loop(train_loader,model,loss_fn,optimizer,train_losses):

    for b,(x_train,y_train) in enumerate(train_loader):
        pred = model(x_train)
        loss  = loss_fn(pred,y_train)

 
        loss.backward() #computing gradients 
        optimizer.step() #updating params
        optimizer.zero_grad() #reset

        if b%100  == 0:
            print(f' loss: {loss.item()}')
    train_losses.append(loss)

def test_loop(test_loader,model, loss_fn, test_losses):
    model.eval()
    with torch.no_grad():
        for b, (x_test,y_test) in enumerate(test_loader):
            pred = model(x_test)
            loss = loss_fn(pred, y_test)
            print(f'test losses : {loss.item()}')
    test_losses.append(loss)






model = Net()

epochs = 10
learning_rate = 0.0001

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



for i in range(epochs):
    print(f'Epoch {i+1}')
    train_loop(train_loader, model, loss_fn, optimizer,train_losses)
    test_loop(test_loader, model , loss_fn,test_losses)
print('done')

PATH = 'model_weights.pth'
torch.save(model.state_dict(),PATH)

train_losses = [t1.item() for t1 in train_losses]
plt.plot(train_losses, label='train losses')
plt.plot(test_losses, label='test losses ')
plt.legend()
plt.show()



