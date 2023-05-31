'''
MNIST: handwritten digits dataset, 28*28, 1channel gray image
epoch: training set 전체가 한번 순회하면 epoch은 1
batch_size: training set을 나누는 크기. 메모리와 속도 측면에서 training set을 나눠서 학습하는게 효율적.
iteration: batch를 학습에 사용한 횟수
ex) 1000장을 학습할 때 batch size가 500이면 2iteration을 거쳐야 epoch이 1
'''

import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets #MNIST를 불러옴
import matplotlib.pyplot as plt
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu' # GPU 또는 CPU사용

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# parameters
training_epochs = 15
batch_size = 100


# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/', #MNIST의 경로
                          train=True, #MNIST의 train set을 불러옴
                          transform=transforms.ToTensor(), #이미지를 불러올 때 어떤 transform을 이용할 것인지, ToTensor()를 통해 이미지를 pytorch에 맞게 변형
                          download=True)# 데이터가 없을 시 다운받음

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False, #MNIST의 test set을 불러옴
                         transform=transforms.ToTensor(),
                         download=True)

# dataset loader, 데이터셋을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체로 감쌈
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True, #데이터를 무작위로 불러옴
                                          drop_last=True) # training set % batch_size != 0 인 경우 남는 데이터는 사용하지 않음

# 선형 변환, MNIST data image of shape 28 * 28 = 784
linear = torch.nn.Linear(784, 10, bias=True).to(device)

# define cost/loss & optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)    # Softmax is internally computed.
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1) #optimizer로 SGD알고리즘 사용


for epoch in range(training_epochs): #15번 순회
    avg_cost = 0
    total_batch = len(data_loader) #batch 수

    for X, Y in data_loader:
        # reshape input image into [batch_size by 784]
        # label is not one-hot encoded
        X = X.view(-1, 28 * 28).to(device) #X에 data image 저장, view를 이용해 (batchsize,1,28,28)을 (batchsize,784)로 변환
        Y = Y.to(device) #label 0~9

        optimizer.zero_grad() # gradients를 0으로 초기화
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')



# Test the model using test sets
with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # Get one and predict
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = linear(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()