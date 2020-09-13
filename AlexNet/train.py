import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import os, json
from model import AlexNet
import torch.nn as nn
import torch.optim as optim

device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transfrom = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

data_root = os.path.abspath(os.getcwd()) + "\\flower_data"
print(data_root)

train_set = torchvision.datasets.ImageFolder(root=data_root+"\\train", transform=data_transfrom["train"])
train_num = len(train_set)
val_set = torchvision.datasets.ImageFolder(root=data_root+"\\val", transform=data_transfrom["val"])
val_num = len(val_set)

name_list = train_set.class_to_idx
cla_dict = dict((val, key) for key, val in name_list.items())
print(cla_dict)
# 写入 json 文件
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
net = AlexNet(num_classes=5, init_weights=True)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimer = optim.Adam(net.parameters(), lr=0.0002)
save_path = './AlexNet.pth'
best_acc = 0.0

# 开始训练
for epoch in range(2):
    net.train()
    running_loss = 0
    for step, data in enumerate(train_loader, start=0):
        inputs, labels = data
        optimer.zero_grad()
        outputs = net(inputs.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimer.step()

        # print statistics
        running_loss += loss.item()
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    # validate
    net.eval()
    acc = 0.0
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data
            val_outputs = net(val_images.to(device))
            predict_y = torch.max(val_outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if(val_accurate > best_acc):
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')








