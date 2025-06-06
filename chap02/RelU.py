import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# CUDA 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST 데이터셋 다운로드 및 변환
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./datasets', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./datasets', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# CNN 모델 정의하기
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
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 모델 및 옵티마이저 설정
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 훈련 및 평가 함수
def fit(epoch, model, data_loader, phase='training'):
    if phase == 'training':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        # GPU 또는 CPU로 데이터 이동
        data, target = data.to(device), target.to(device)

        if phase == 'training':
            optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)

        if phase == 'training':
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * data.size(0)
        preds = output.argmax(dim=1)
        running_correct += preds.eq(target).sum().item()

    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)
    print(f'{phase.capitalize()} Epoch {epoch}: Loss={loss:.4f}, Accuracy={running_correct}/{len(data_loader.dataset)} ({accuracy:.2f}%)')
    return loss, accuracy

# 훈련 루프
train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
for epoch in range(1,20):
    epoch_loss, epoch_accuracy = fit(epoch,model,train_loader,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,model,test_loader,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

print('end')

# 10. 훈련 데이터와 검증 데이터의 손실 그래프
plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo', label='training loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, 'r', label='validation loss')
plt.legend()
plt.show()

# 11. 훈련 데이터와 검증 데이터의 정확도 그래프
plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, 'bo', label='train accuracy')
plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, 'r', label='val accuracy')
plt.legend()
plt.show()