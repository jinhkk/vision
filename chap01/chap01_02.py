# import numpy as np
# import torch
# import torch.nn as nn
# from torchvision import datasets, transforms
# from tqdm import tqdm
#
# mnist_train = datasets.MNIST(
#     root='./datasets', train=True, transform=transforms.ToTensor(), download=True
# )
#
# mnist_test = datasets.MNIST(
#     root='./datasets', train=False, transform=transforms.ToTensor(), download=True
# )
#
# train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
# test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)
#
# input_size = 784
# hidden_sizes = [128, 64]
# output_size = 10
#
# model = nn.Sequential(
#     nn.Linear(input_size, hidden_sizes[0]),
#     nn.ReLU(),
#     nn.Linear(hidden_sizes[0], hidden_sizes[1]),
#     nn.ReLU(),
#     nn.Linear(hidden_sizes[1], output_size),
#     nn.LogSoftmax(dim=1)
# )
#
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.09)
#
# epochs = 15
# for e in range(epochs):
#     running_loss = 0
#
#     for images, labels in train_loader:
#         images = images.view(images.shape[0], -1)
#         optimizer.zero_grad()
#
#         # 모델 계산
#         output = model(images)
#         # 손실계산
#         loss = criterion(output, labels)
#         # 역전파
#         loss.backward()
#         # 최적화
#         optimizer.step()
#
#         running_loss += loss.item()
#     # running_loss는 여태 손실 값을 전부 저장한 거임 그러니까 훈련 데이터의 개수만큼 나누어서 평균을 출력하기 위해 사용됨
#     print("Epoch {} - Training loss : {}".format(e, running_loss/len(train_loader)))
#
# # 테스트
# correct = 0
# total = len(mnist_test)
# with torch.no_grad():
#
#     # 테스트 세트 미니배치 for 문
#     for images, labels in tqdm(test_loader):
#         # 순전파
#         x = images.view(images.shape[0], -1)
#         y = model(x)
#
#         predictions = torch.argmax(y, dim=1)
#         correct += torch.sum((predictions == labels).float())
#
# accuracy = correct / total * 100
# print("Test Accuracy: {:.2f}%".format(accuracy))


# 1. 패키지 임포트
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 2. GPU 사용 체크
is_cuda = False
if torch.cuda.is_available():
    is_cuda = True

# 3. mnist 데이터 다운로드
transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST('data/', train=True, transform=transformation, download=True)
test_dataset = datasets.MNIST('data/', train=False, transform=transformation, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)


# 4. 네트워크 정의
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x,p=0.1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 5. 모델 불러오기
model = Net()
if is_cuda:
    model.cuda()

# 6. 최적화 함수
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 7. 훈련 데이터 변수 준비
data, target = next(iter(train_loader))


# 8. 훈련 및 검증 함수
def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile), Variable(target)
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        # running_loss += F.nll_loss(output,target,size_average=False).data[0]
        running_loss += F.nll_loss(output, target, reduction='sum').item()
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)

    print(
        f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss, accuracy


# 9. 훈련
train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []
for epoch in range(1, 20):
    epoch_loss, epoch_accuracy = fit(epoch, model, train_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, model, test_loader, phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

# 10. 훈련 데이터와 검증 데이터의 손실 그래프
plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo', label='training loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, 'r', label='validation loss')
plt.legend()

# 11. 훈련 데이터와 검증 데이터의 정확도 그래프
plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, 'bo', label='train accuracy')
plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, 'r', label='val accuracy')
plt.legend()
