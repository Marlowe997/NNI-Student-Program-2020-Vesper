# Task2

## Task 2.1 CIFAR10训练小型神经网络

### 训练&测试集

CIFAR-10

![avatar](https://images2018.cnblogs.com/blog/1196151/201712/1196151-20171225161744462-2083152737.png)

​	该数据集共有60000张彩色图像，这些图像是32*32，分为10个类，每类6000张图。这里面有50000张用于训练，构成了5个训练批，每一批10000张图；另外10000用于测试，单独构成一批。测试批的数据里，取自10类中的每一类，每一类随机取1000张。抽剩下的就随机排列组成了训练批。注意一个训练批中的各类图像并不一定数量相同，总的来看训练批，每一类都有5000张图。

通过`torchvision.datasets、torch.utils.data.DataLoader`这两个包来加载

```python
transform = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform1 = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform1)
testloader = torch.utils.data.DataLoader(testset, batch_size=50,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

### 神经网络定义

网络模型如下：

```python
self.conv1 = nn.Conv2d(3,64,3,padding=1)
self.conv2 = nn.Conv2d(64,64,3,padding=1)
self.pool1 = nn.MaxPool2d(2, 2)
self.bn1 = nn.BatchNorm2d(64)
self.relu1 = nn.LeakyReLU()

self.conv3 = nn.Conv2d(64,128,3,padding=1)
self.pool2 = nn.MaxPool2d(2, 2, padding=1)
self.bn2 = nn.BatchNorm2d(128)
self.relu2 = nn.LeakyReLU()

self.conv4 = nn.Conv2d(128,128, 3,padding=1)
self.conv5 = nn.Conv2d(128,128, 1,padding=1)
self.pool3 = nn.MaxPool2d(2, 2, padding=1)
self.bn3 = nn.BatchNorm2d(128)
self.relu3 = nn.LeakyReLU()

self.conv6 = nn.Conv2d(128, 256, 3,padding=1)
self.conv7 = nn.Conv2d(256, 256, 1, padding=1)
self.pool4 = nn.MaxPool2d(2, 2, padding=1)
self.bn4 = nn.BatchNorm2d(256)
self.relu4 = nn.LeakyReLU()

self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
self.conv9 = nn.Conv2d(512, 512, 1, padding=1)
self.pool5 = nn.MaxPool2d(2, 2, padding=1)
self.bn5 = nn.BatchNorm2d(512)
self.relu5 = nn.LeakyReLU()

self.fc14 = nn.Linear(512*4*4,1024)
self.drop1 = nn.Dropout2d()
self.fc15 = nn.Linear(1024,1024)
self.drop2 = nn.Dropout2d()
self.fc16 = nn.Linear(1024,10)
```

loss函数： `CrossEntropyLoss`

optimizer: `Adam`

网络：`VGG16变体`

`batch_size == 100`

`epoch == 100`

### 训练及测试代码

```python
    def train_sgd(self,device):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        x = np.zeros(100)
        y = np.zeros(100)
        path = 'weights.tar'
        initepoch = 0
        if os.path.exists(path) is not True:
            loss = nn.CrossEntropyLoss()
            # optimizer = optim.SGD(self.parameters(),lr=0.01)
        else:
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            initepoch = checkpoint['epoch']
            loss = checkpoint['loss']

        for epoch in range(initepoch, 100):  # loop over the dataset multiple times
            timestart = time.time()
            running_loss = 0.0
            total = 0
            correct = 0
            for i, data in enumerate(trainloader, 100):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device),labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                l = loss(outputs, labels)
                l.backward()
                optimizer.step()

                # print statistics
                running_loss += l.item()
                # print("i ",i)
                if i % 500 == 499:  # print every 500 mini-batches
                    print('[%d, %5d] loss: %.4f' %
                          (epoch, i, running_loss / 500))
                    y[epoch] = running_loss / 500
                    running_loss = 0.0
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    print('Accuracy of the network on the %d train images: %.3f %%' % (total,
                            100.0 * correct / total))
                    x[epoch] = 100 * correct / total
                    total = 0
                    correct = 0
                    torch.save({'epoch':epoch,
                                'model_state_dict':net.state_dict(),
                                'optimizer_state_dict':optimizer.state_dict(),
                                'loss':loss
                                },path)

            print('epoch %d cost %3f sec' %(epoch,time.time()-timestart))
        print('Finish Training')
        return x,y
    def test(self,device):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %.3f %%' % (
                100.0 * correct / total))
```

### 运行结果

![image-20210103144307951.png](https://i.loli.net/2021/01/03/k6LBv54w8ZGPHcg.png)

最终在测试集上准确率83.9%

![image-20210103145332970.png](https://i.loli.net/2021/01/03/96tsYhciOvHTV3u.png)

### 总结

​	目前图像分类方面效果较好的网络主要有VGG-16和ResNet，其识别准确率普遍达到93%以上。在进行了一定的调研后，笔者对两种网络都进行了基本的测试，并最终决定选用目前的网络。本神经网络基于VGG-16，由于VGG-16在个人PC端运行时存在内存不足等问题，故在一定程度上减少了卷积核的数量，并对其结构进行了一定的改进。

​	为了解决梯度爆炸和梯度消失的问题，激活函数采用了LeakyRelu，在一定程度上避免了以上问题的出现，同时也采取了一定的Dropout操作。优化方面采用了Adam作为optimizer，相对而言计算更加简单且高效，参数的变换不会受到梯度的伸缩变换影响，更新的步长也被限制在大致的范围内并能自然地实现步长退火过程。