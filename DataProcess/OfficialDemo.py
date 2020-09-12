import torch
import torchvision
import torchvision.transforms as transforms
from model import InitNet
import torch.nn as nn
import torch.optim as optim

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=False)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=36, shuffle=True, num_workers=0)

val_set = torchvision.datasets.CIFAR10(root='./root', train=False, transform=transform, download=False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000, shuffle=True, num_workers=0)

val_data_iter = iter(val_loader)
val_image, val_label  = val_data_iter.next()

classes = ('x', 'y', 'z')

# 开始训练数据
net = InitNet()
loss_functoin = nn.CrossEntropyLoss()
optimer = optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(3):
    runningloss = 0
    for step, data in enumerate(train_loader):
        inputs, labels = data
        optimer.zero_grad()

        outputs = net(inputs)
        loss = loss_functoin(outputs, labels)
        loss.backward()

        # print log
        runningloss += loss.item()
        if step%500 == 499:     # every 500 step print
            with torch.no_grad():
                val_outputs = net(val_image)
                predict_y = torch.max(outputs, dim=1)
                accuracy = (predict_y == val_label).sum().item() / val_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print('Finished Training')

save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)

