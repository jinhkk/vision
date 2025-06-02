# ResNet 파이토지 예제
# import matplotlib
# matplotlib.use('Agg')  # 그래픽 없이 이미지 저장용 백엔드 사용

"""
**ResNet(Residual Network)**은 딥러닝에서 널리 쓰이는 이미지 인식용 신경망 구조입니다.
핵심 아이디어: "잔차 연결(skip connection)"을 통해 깊은 네트워크도 쉽게 학습할 수 있게 함.
장점: 기울기 소실 문제 해결, 더 깊은 모델 학습 가능, 다양한 분야에 활용 가능.
대표 모델: ResNet-50, ResNet-101, ResNet-152 등
"""
import certifi
import os  # 파일 및 디렉토리 경로 조작을 위한 표준 라이브러리
os.environ["SSL_CERT_FILE"] = certifi.where()
from torch.utils.data import Dataset  # PyTorch의 데이터셋 클래스 상속을 위한 모듈
import torchvision.transforms as transforms  # 이미지 전처리(transform) 기능 제공
from PIL import Image  # 이미지 파일을 열고 처리하기 위한 라이브러리
import torch  # PyTorch 메인 라이브러리

# PyTorch의 Dataset 클래스를 상속받아 커스텀 데이터셋 정의
class PyTorchCustomDataset(Dataset):
    def __init__(self,
                 root_dir='D:/vision/chap03/cats_and_dogs_filtered/train',  # 이미지가 저장된 루트 디렉토리
                 transform=None):  # 이미지 전처리(transform) 함수
        self.image_abs_path = root_dir  # 이미지 루트 경로 저장
        self.transform = transform  # transform 저장
        self.label_list = os.listdir(self.image_abs_path)  # 클래스 디렉토리 목록 (예: ['cats', 'dogs'])
        self.label_list.sort()  # 클래스 이름 정렬 (일관된 라벨링을 위해)
        self.x_list = []  # 이미지 파일 경로 리스트
        self.y_list = []  # 이미지에 해당하는 라벨 리스트

        # 각 클래스 디렉토리 순회
        for label_index, label_str in enumerate(self.label_list):
            img_path = os.path.join(self.image_abs_path, label_str)  # 클래스 디렉토리 경로
            img_list = os.listdir(img_path)  # 해당 클래스의 이미지 파일 목록
            for img in img_list:
                self.x_list.append(os.path.join(img_path, img))  # 전체 이미지 경로 저장
                self.y_list.append(label_index)  # 클래스 인덱스를 라벨로 저장
        pass

    def __len__(self):
        return len(self.x_list)  # 전체 이미지 개수 반환

    def __getitem__(self, idx):
        image = Image.open(self.x_list[idx])  # 이미지 파일 열기
        if image.mode != "RGB":  # 흑백 등 RGB가 아닌 경우
            image = image.convert('RGB')  # RGB로 변환
        if self.transform != None:
            image = self.transform(image)  # transform이 정의되어 있으면 적용
        return image, self.y_list[idx]  # 이미지와 라벨 반환

    def __save_label_map__(self, dst_text_path="label_map.txt"):
        label_list = self.label_list  # 클래스 이름 리스트
        f = open(dst_text_path, 'w')  # 텍스트 파일 열기
        for i in range(len(label_list)):
            f.write(label_list[i] + '\n')  # 각 클래스 이름을 한 줄씩 저장
        f.close()  # 파일 닫기
        pass

    def __num_classes__(self):
        return len(self.label_list)  # 전체 클래스 수 반환

# Model

# 네트워크 정의

# torchvision에서 사전 학습된 모델(resnet18)을 불러오기 위해 import
from torchvision import models

# 신경망 구성에 필요한 모듈들 import
import torch.nn as nn
import torch.nn.functional as F

# 사용자 정의 모델 클래스 정의 (PyTorch의 nn.Module 상속)
class MODEL(nn.Module):
    def __init__(self, num_classes):  # num_classes: 분류할 클래스 수
        super().__init__()  # 부모 클래스 초기화

        # 사전 학습된 ResNet18 모델 불러오기 (ImageNet으로 학습됨)
        self.network = models.resnet18(pretrained=True)

        # 출력층을 사용자 정의 분류기로 교체
        self.classifier = nn.Sequential(
            nn.Dropout(),  # 과적합 방지를 위한 드롭아웃
            nn.Linear(1000, num_classes),  # ResNet18의 출력(1000차원)을 원하는 클래스 수로 변환
            nn.Sigmoid()  # 출력값을 0~1 사이로 변환 (이진 분류나 멀티라벨 분류에 사용)
        )

    def forward(self, x):  # 순전파 정의
        x = self.network(x)  # 입력 이미지를 ResNet18에 통과시킴
        return self.classifier(x)  # 그 결과를 커스텀 분류기에 통과시켜 최종 출력

# main

import torch.optim as optim

# 학습 및 검증 손실/정확도 저장용 리스트
train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []

# 학습 메인 함수 정의
def trainmain():
    USE_CUDA = torch.cuda.is_available()  # CUDA(GPU) 사용 가능 여부 확인
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")  # 사용할 디바이스 설정

    img_width, img_height = 224, 224  # 입력 이미지 크기 설정 (ResNet 기본 입력 크기)
    EPOCHS = 12  # 학습 반복 횟수
    BATCH_SIZE = 32  # 배치 크기

    # 학습 데이터 전처리 정의
    transform_train = transforms.Compose([
        transforms.Resize(size=(img_width, img_height)),  # 이미지 크기 조정
        transforms.RandomRotation(degrees=15),  # 약간의 회전으로 데이터 증강
        transforms.ToTensor()  # 텐서로 변환
    ])

    # 검증 데이터 전처리 정의
    transform_test = transforms.Compose([
        transforms.Resize(size=(img_width, img_height)),  # 이미지 크기 조정
        transforms.ToTensor()  # 텐서로 변환
    ])

    # 커스텀 데이터셋 클래스 지정
    TrainDataset = PyTorchCustomDataset
    TestDataset = PyTorchCustomDataset

    # 학습 및 검증 데이터셋 생성
    train_data = TrainDataset(root_dir='D:/vision/chap03/cats_and_dogs_filtered/train', transform=transform_train)
    test_data = TestDataset(root_dir='D:/vision/chap03/cats_and_dogs_filtered/validation', transform=transform_test)

    # DataLoader로 배치 단위 데이터 로딩
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True  # 데이터 섞기
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # 클래스 이름을 텍스트로 저장
    train_data.__save_label_map__()

    # 클래스 수 가져오기
    num_classes = train_data.__num_classes__()

    # 모델 생성 및 디바이스로 이동
    model = MODEL(num_classes).to(DEVICE)

    # 모델 저장 파일명 지정
    model_str = "PyTorch_Classification_Model.pt"

    # 옵티마이저 및 학습률 스케줄러 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Adam 옵티마이저 사용
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 10 에폭마다 학습률 감소
    acc = 0.0

    # 에포크 만큼 훈련, 검증

    for epoch in range(1, EPOCHS + 1):  # 1부터 EPOCHS까지 반복 (학습 반복)
        model.train()  # 모델을 학습 모드로 설정 (Dropout, BatchNorm 등이 학습 모드로 작동)
        tr_loss = 0.0  # 에폭 단위의 총 손실 초기화
        tr_correct = 0.0  # 에폭 단위의 정답 예측 개수 초기화

        for data, target in train_loader:  # train_loader에서 배치 단위로 데이터를 불러옴
            data, target = data.to(DEVICE), target.to(DEVICE)  # 데이터를 GPU나 CPU로 이동
            optimizer.zero_grad()  # 이전 배치의 gradient 초기화

            output = model(data)  # 모델에 입력 → 예측값 출력
            loss = F.cross_entropy(output, target)  # Cross Entropy Loss 계산 (softmax 포함됨)

            # NLL Loss는 log_softmax + nll이므로, 비교용으로 쓸 수 있음
            tr_loss += F.nll_loss(output, target, reduction='sum').item()  # 배치 손실 누적

            pred = output.data.max(dim=1, keepdim=True)[1]  # 가장 확률 높은 클래스 선택 (예측)
            tr_correct += pred.eq(target.view_as(pred)).sum().item()  # 예측이 정답과 같은지 체크해서 맞춘 개수 누적

            loss.backward()  # 역전파 수행: gradient 계산
            optimizer.step()  # 가중치 업데이트

        scheduler.step()  # learning rate 조정 스케줄러 적용

        # 에폭별 손실 및 정확도 계산
        tr_ep_loss = tr_loss / len(train_loader.dataset)  # 평균 훈련 손실
        tr_ep_accuracy = 100. * tr_correct / len(train_loader.dataset)  # 훈련 정확도 (%)

        # === 평가 단계 (Validation Phase) ===
        model.eval()  # 모델을 평가 모드로 설정
        te_loss = 0  # 검증 손실 초기화
        te_correct = 0  # 검증 정답 수 초기화

        with torch.no_grad():  # 평가에서는 gradient 계산 안 함 (메모리 절약)
            for data, target in test_loader:  # test_loader로부터 배치 추출
                data, target = data.to(DEVICE), target.to(DEVICE)  # GPU/CPU로 이동
                output = model(data)  # 예측 수행
                loss = F.cross_entropy(output, target)  # 손실 계산
                te_loss += F.cross_entropy(output, target, reduction='sum').item()  # 손실 누적
                pred = output.max(1, keepdim=True)[1]  # 예측 결과
                te_correct += pred.eq(target.view_as(pred)).sum().item()  # 정답 개수 누적

            te_ep_loss = te_loss / len(test_loader.dataset)  # 평균 검증 손실
            te_ep_accuracy = 100. * te_correct / len(test_loader.dataset)  # 검증 정확도

            # 학습 및 검증 결과 출력
            print('[{}] Train Loss: {:.4f}, Train Accuracy: {:.2f}% Test Loss: {:.4f}, '
                  'Test Accuracy: {:.2f}%'.format(
                epoch, tr_ep_loss, tr_ep_accuracy, te_ep_loss, te_ep_accuracy))

            # 최고 성능 갱신 시 모델 저장
            if acc < te_ep_accuracy:
                acc = te_ep_accuracy
                torch.save(model.state_dict(), model_str)  # 모델 저장
                print("model saved!")

            # 손실 및 정확도 값 저장 (그래프용)
            train_losses.append(tr_ep_loss)
            train_accuracy.append(tr_ep_accuracy)
            val_losses.append(te_ep_loss)
            val_accuracy.append(te_ep_accuracy)


# 훈련 메인 함수 호출
if __name__ == '__main__':
    # 여기서 실제 실행할 함수 호출
    trainmain()

# 훈련 데이터와 검증 데이터의 손실 그래프
import matplotlib.pyplot as plt

plt.plot(range(1,len(train_losses)+1),train_losses,'bo',label = 'training loss')
plt.plot(range(1,len(val_losses)+1),val_losses,'r',label = 'validation loss')
plt.legend()

# 훈련 데이터와 검증 데이터의 정확도 그래프
plt.plot(range(1,len(train_accuracy)+1),train_accuracy,'bo',label = 'train accuracy')
plt.plot(range(1,len(val_accuracy)+1),val_accuracy,'r',label = 'val accuracy')
plt.legend()

# 테스트 이미지 로딩 및 전처리부터 모델 추론까지 전체 과정을 포함한 예제 코드
# 각 부분에 대해 자세한 설명을 주석으로 달아두었으므로, 코드 전체를 복사해 사용하실 수 있습니다.

import os
from PIL import Image        # 이미지 처리 (Pillow 라이브러리)
import cv2                   # OpenCV: 이미지 파일 읽기 및 색상 변환 등에 사용
import numpy as np           # 수치 연산 및 배열 처리
import matplotlib.pyplot as plt  # 이미지 및 데이터 시각화
import torch                 # PyTorch
import torchvision         # PyTorch의 torchvision 모듈
import torchvision.transforms as transforms  # 이미지 전처리 함수들

##############################################
# 1. 테스트 이미지 로딩 및 경로 설정
##############################################
# 기본 디렉터리를 지정합니다.
PATH = "D:/vision/chap03/cats_and_dogs_filtered/validation"

# 검증(validation) 이미지 내에서 고양이(cats)와 개(dogs) 이미지가 있는 하위 폴더 경로를 설정합니다.
validation_cats_dir = PATH + '/cats'   # 고양이 이미지 폴더
validation_dogs_dir = PATH + '/dogs'     # 개 이미지 폴더

# 각 폴더 내의 파일 이름 목록을 가져옵니다.
list_of_test_cats_images = os.listdir(validation_cats_dir)
list_of_test_dogs_images = os.listdir(validation_dogs_dir)

# 파일 이름에 전체 경로를 추가하여 파일의 절대 경로 리스트를 만듭니다.
for idx in range(len(list_of_test_cats_images)):
    list_of_test_cats_images[idx] = validation_cats_dir + '/' + list_of_test_cats_images[idx]
for idx in range(len(list_of_test_dogs_images)):
    list_of_test_dogs_images[idx] = validation_dogs_dir + '/' + list_of_test_dogs_images[idx]

# 고양이와 개 이미지 경로 리스트를 하나로 합쳐 전체 테스트 이미지 리스트를 생성합니다.
list_of_test_images = list_of_test_cats_images + list_of_test_dogs_images

# 중간 확인을 위해 일부 이미지 경로를 출력합니다.
print("예시 고양이 이미지 경로:", list_of_test_cats_images[10])
print("예시 전체 이미지 경로:", list_of_test_images[501])

##############################################
# 2. 이미지 입출력 및 추론 관련 함수 정의
##############################################

# 라벨맵 로딩 함수: 텍스트 파일에서 탭('\t') 기준으로 라벨 정보를 읽어 문자열 배열로 반환합니다.
def load_label_map(textFile):
    return np.loadtxt(textFile, str, delimiter='\t')

# 이미지 읽기 함수: OpenCV를 사용해 주어진 경로의 이미지를 읽고, 경로를 출력합니다.
def cv_image_read(image_path):
    print("읽는 이미지 경로:", image_path)
    return cv2.imread(image_path)

# 이미지 시각화 함수: OpenCV는 BGR로 이미지를 읽기 때문에, RGB로 변환 후 matplotlib로 출력합니다.
def show_image(cv_image):
    rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)  # BGR → RGB 변환
    plt.figure()
    plt.imshow(rgb)
    plt.show(block=True)

# 추론 결과 출력 함수: 모델의 추론 결과에서 가장 높은 확률을 가진 클래스 인덱스를 찾아 라벨맵에서 해당 클래스 이름 출력
def print_result(inference_result, class_map):
    class_text = class_map[np.argmax(inference_result)]
    print("추론 결과 배열:", inference_result)
    print("예측 클래스:", class_text)

# 이미지 추론 함수: OpenCV 이미지 → PIL 이미지 변환 → 전처리(transform) → 배치 차원 추가 → 모델 추론 실행
def inference_image(opencv_image, transform_info, model, DEVICE):
    image = Image.fromarray(opencv_image)      # NumPy 배열을 PIL 이미지로 변환
    image_tensor = transform_info(image)         # 전처리 (리사이즈, ToTensor 등)
    image_tensor = image_tensor.unsqueeze(0)     # 배치 차원 추가 (1, C, H, W)
    image_tensor = image_tensor.to(DEVICE)         # 텐서를 지정한 디바이스(CPU/GPU)로 이동
    result = model(image_tensor)                   # 모델에 입력 후 추론 결과 반환
    return result

##############################################
# 3. 테스트 이미지 시각화로 함수 동작 확인
##############################################
# 리스트의 특정 이미지들을 읽어와 화면에 출력합니다.
show_image(cv_image_read(list_of_test_images[10]))
show_image(cv_image_read(list_of_test_images[501]))

##############################################
# 4. 테스트 메인 함수: 전체 추론 파이프라인 수행
##############################################
# 이 함수는 주어진 이미지 경로를 받아 다음을 수행합니다:
#   1. 디바이스 설정(CUDA 사용 가능 여부 확인)
#   2. 이미지 전처리(리사이즈, ToTensor 등) 설정
#   3. 라벨 맵 파일("label_map.txt") 로드
#   4. 분류 모델(MODEL)을 생성하고 저장된 가중치(pt 파일)를 불러온 후, 평가 모드로 설정
#   5. 이미지를 읽고 모델 추론 후 결과를 출력 및 시각화
def testmain(image_path):
    # 1. CUDA 사용 가능 여부를 확인하고 DEVICE 설정
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    # 2. 이미지 전처리: 224x224 크기로 리사이즈하고 텐서로 변환
    img_width, img_height = 224, 224
    transform_info = transforms.Compose([
        transforms.Resize(size=(img_width, img_height)),
        transforms.ToTensor()
    ])

    # 3. 라벨 파일 로드 ("label_map.txt" 파일 내의 탭 구분 라벨 정보)
    class_map = load_label_map('label_map.txt')
    num_classes = len(class_map)

    # 4. 분류 모델 로딩:
    #    MODEL은 미리 정의된 모델 클래스 또는 함수여야 합니다.
    #    "PyTorch_Classification_Model.pt" 파일로부터 가중치를 불러와 모델에 적용합니다.
    model = MODEL(num_classes).to(DEVICE)
    model_str = "PyTorch_Classification_Model" + ".pt"
    model.load_state_dict(torch.load(model_str))
    model.eval()  # 평가 모드로 전환 (dropout, batchnorm 등 변경 방지)

    # 5. 이미지 읽기 및 추론:
    #    OpenCV로 이미지를 읽고, 정의된 inference_image 함수를 통해 모델 추론을 진행합니다.
    opencv_image = cv_image_read(image_path)
    inference_result = inference_image(opencv_image, transform_info, model, DEVICE)

    # 6. 추론 결과 후처리: 결과 텐서를 CPU로 이동, gradient 추적 해제 후 NumPy 배열로 변환
    inference_result = inference_result.cpu().detach().numpy()

    # 7. 추론 결과 출력 및 원본 이미지 시각화
    print_result(inference_result, class_map)
    show_image(opencv_image)

##############################################
# 5. 테스트 이미지로 테스트 메인 함수 실행
##############################################
# 테스트용으로 두 개의 이미지 경로를 이용하여 전체 추론 파이프라인을 실행합니다.

# 테스트 1: 리스트에서 10번째 이미지로 추론을 실행
image_path = list_of_test_images[10]
testmain(image_path)

# 테스트 2: 리스트에서 501번째 이미지로 추론을 실행
image_path = list_of_test_images[501]
testmain(image_path)