import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

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







class CharRNN(nn.Module):
    def __init__(self):
        super(CharRNN, self).__init__()

        self.input_size = 28
        self.time_steps = 28
        self.hidden_size = 128

        self.lstm1 = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, dropout=0.5)
        
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, dropout=0.5)
        
        self.fc1 = nn.Linear(self.hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc_out = nn.Linear(16, 10) 

    def forward(self, x):
        x,_ = self.lstm1(x)
        x,_ = self.lstm2(out)

        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))

        x = x[:, -1, :]    # take the last time step output
        x = self.fc_out(x)

        return output



def train_loop(train_loader,model,loss_fn,optimizer,trian_losses):
    model.train()
    for b, (x_train, y_train) in enumerate(train_loader):
        #cnn takes (batch_size,channels,height,width)
        # rnn takes (batch_size, time_steps, input_size) )

        # dont forget to reshape the data....................

        pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if b % 100 == 0:
            print(f"Batch {b} loss: {loss.item()}")
        trian_losses.append(loss.item())


def test_loop(test_loader,model,loss_fn,test_losses):
    model.eval()
    with torch.no_grad():
        for b, (x_test, y_test) in enumerate(test_loader):
         
            pred = model(X_test)
            loss = loss_fn(pred, y_test)
            print(f"Batch {b} test loss: {loss.item()}")
        test_losses.append(loss.item())