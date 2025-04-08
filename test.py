import torch
import torch.nn as nn
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
# train loader is 4dim 

# image, label = train_data[0]
# print(image.shape) #one image and its label ---> [3,32,32]
# print(label)

# Check the shape of train_data
print(f"Train data shape: {train_data.data.shape}")  # For CIFAR-10, this will be (50000, 32, 32, 3)

# Check the shape of a batch from train_loader
for images, labels in train_loader:
    print(f"Train loader batch shape: {images.shape}")  # Expected: [batch_size, channels, height, width]
    print(f"Labels batch shape: {labels.shape}")       # Expected: [batch_size]
    break  # Only check the first batch

# images, labels = next(iter(train_data))
# print(images.shape) #batch of 32 images ---> [32,3,32,32]
# print(labels.shape) # 32 labels


# plt.imshow(image.permute(1,2,0))            #image is blury from normalizing the image
# plt.title(f'label: {label}')
# plt.show()

# for i,(x,y) in enumerate(train_data):
#     break

# tensor ---> [batchSize , channels , height , width ]
# print('input data : ' , x.shape)
# x = x.view(1,3,32,32)

# conv1 = nn.Conv2d(3,16,3,1,1)
# batch_norm1 = nn.BatchNorm2d(16)
# conv2 = nn.Conv2d(16,32,3,1,1)
# batch_norm2 = nn.BatchNorm2d(32)
# pool1 = nn.MaxPool2d(2,2)
# dropout1 = nn.Dropout2d(0.25)

# conv3 = nn.Conv2d(32,64,3,1,1)
# batch_norm3 = nn.BatchNorm2d(64)
# conv4 = nn.Conv2d(64,128,3,1,1)
# batch_norm4 = nn.BatchNorm2d(128)
# pool2 = nn.MaxPool2d(2,2)
# dropout2 = nn.Dropout2d(0.25)

# conv5 = nn.Conv2d(128,256,3,1,1)
# batch_norm5 = nn.BatchNorm2d(256)
# conv6 = nn.Conv2d(256,512,3,1,1)
# batch_norm6 = nn.BatchNorm2d(512)
# pool3 = nn.MaxPool2d(2,2)
# dropout3 = nn.Dropout2d(0.25)

# # # flatten
# fc1 = nn.Linear(512*4*4,128)
# dropout4 = nn.Dropout2d(0.25)
# fc2 = nn.Linear(128,64)
# fc3 = nn.Linear(64,10)





# x = F.relu(batch_norm1(conv1(x)))
# print(x.shape)
# x = F.relu(batch_norm2(conv2(x)))
# print(x.shape)
# x = pool1(x) # /2 of size
# x = dropout1(x)
# print(x.shape)

# x = F.relu(batch_norm3(conv3(x)))
# x = F.relu(batch_norm4(conv4(x)))
# x = pool2(x)
# x = dropout2(x)

# x = F.relu(batch_norm5(conv5(x)))
# x = F.relu(batch_norm6(conv6(x)))
# x = pool3(x)
# x = dropout3(x)

# print(x.shape)
# x = torch.flatten(x,1)

# x = F.relu(fc1(x))


# x = F.relu(fc2(x))


# x = fc3(x)


# x = F.log_softmax(x, dim=1)

