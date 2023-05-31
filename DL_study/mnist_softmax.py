#sigmoid의 문제점: 그래프의 양 끝의 gradient가 0에 가까움, back propagation에서 vanishing gradient발생
#ReLU: f(x) = max(0,x), 음수의 영역에선 gradient가 0
#torch.opitm.SGD/Adadelta/Adagrad/Adam/SparseAdam/ASGD/LBGGs?RMSprop/Rprop

#pytorch에서 mnist를 어떻게 읽는가

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random

device = 'cuba' if torch.cuba.is_available() else 'cpu'

#for reproductibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuba':
    torch.cuba.manual_seed_all(777)

#parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

#MNIST dataset
mnist_train = dsets.MNIST(root  = 'MNIST_data/',
                          train = True,
                          transform = transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root = 'MNIST_data/',
                         train = False,
                         transform = transforms.ToTensor(),
                         download = True)

# dataset loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True) # 마지막 배치를 버림



# MNIST data image of shape 28 * 28 = 784
linear = torch.nn.Linear(784, 10, bias=True).to(device)
'''mnist__nn 코드
linear1 = torch.nn.Linear(784, 256, bias = True).to(device)
linear2 = torch.nn.Linear(256, 256, bias = True).to(device)
linear3 = torch.nn.Linear(256, 10, bias = True).to(device)
relu = torch.nn.ReLU() 
'''



# Initialization
torch.nn.init.normal_(linear.weight)
'''mnist_nn 코드
torch.nn.init.normal_(linear1.weight)
torch.nn.init.normal_(linear2.weight)
torch.nn.init.normal_(linear3.weight)
'''




# define cost/loss & optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)    # Softmax is internally computed.
optimizer = torch.optim.Adam(linear.parameters(), lr=learning_rate) #SGD 대신 Adam사용
'''mnist_nn 코드
model = torch.nn.Sequential(linear1, relu, linear2, relu, linear3).to(device) #세번째 layer에선 CrossEntropyLoss() 사용
critetrion = torch.nn.CrossEntropyLoss().to(device) # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
'''




#학습 코드
total_batch = len(data_loader)
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader: #data의 이미지와 레이블 불러오기
        # reshape input image into [batch_size by 784]
        # label is not one-hot encoded
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)
        optimizer.zero_grad()
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


#mnist_nn 코드(layer가 3개, Relu함수 사용)가  학습결과 성능이 더 좋음