import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert 3-channel to 1
    transforms.ToTensor()
])

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

input_size = 32  # Image width and height after resizing
sequence_length = 32  # Number of time steps (height of the image)
num_classes = 10  # Number of classes in CIFAR-10
hidden_size = 128  # Number of hidden units in LSTM
num_layers = 2  # Number of LSTM layers
batch_size = 128  # Batch size
learning_rate = 0.001  # Learning rate
epochs = 10  # Number of epochs

# Define the RNN model
class CharRNN(nn.Module):
    def __init__(self):
        super(CharRNN, self).__init__()

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)


        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc_out = nn.Linear(16, 10)

    def forward(self, x):
        # print("input value : ", x.shape)  # [batch_size, time_steps, input_size]
        # CAUSE we reshaped it in train_loop and test_loop
 
        x = x.squeeze(1)
        x, _ = self.lstm1(x)

# after finishing the lstm layer we have to take the last time step output

        x = x[:, -1, :]  # Take the last time step output
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))


        x = self.fc_out(x)
        # print("output value : ", x.shape)  # [batch_size, num_classes]

        return x

# Training loop
def train_loop(train_loader, model, loss_fn, optimizer, train_losses):
    model.train()
    for b, (x_train, y_train) in enumerate(train_loader):
        # Reshape input for RNN_LSTM : (batch_size, time_steps, input_size)
        x_train = x_train.view(x_train.size(0), 32, 32)

        # print("input value for train : ", x_train.shape)  # [batch_size, time_steps, input_size]
        pred = model(x_train)
        loss = loss_fn(pred, y_train)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if b % 100 == 0:
            print(f"Batch {b} loss: {loss.item()}")
        train_losses.append(loss.item())

# Testing loop
def test_loop(test_loader, model, loss_fn, test_losses):
    model.eval()
    with torch.no_grad():
        for b, (x_test, y_test) in enumerate(test_loader):
            # Reshape input for RNN: (batch_size, time_steps, input_size)
            x_test = x_test.view(x_test.size(0), 32, 32)

            pred = model(x_test)
            loss = loss_fn(pred, y_test)
            print(f"Batch {b} test loss: {loss.item()}")
            test_losses.append(loss.item())






# Initialize model, loss function, and optimizer
model = CharRNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



train_losses = []
test_losses = []

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