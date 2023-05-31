import torch
from torch import device
from MNist import data_loader, mnist_test
import torchvision.datasets as dsets

linear = torch.nn.Linear(784,10,bias = True).to(device) #Linear의 입력은 784, 출력은 10
training_epochs = 15
batch_size = 100

criterion = torch.nn.CrossEntropyLoss().to(device) #crossentropy 함수가 softmax 계산
optimizer = torch.optim.SGD(linear.parameters(), lr = 0.1) #parametersm는 weight와 bias



for epoch in range(training_epochs): #학습을 15번 반복
    avg_cost = 0
    total_batch = len(data_loader)
    for X, Y in data_loader:
        X = X.view(-1, 28*28).to(device) # 28*28을 batch_size by 784로 변경
        optimizer.zero_grad()
        hypothesis = linear(X) #분류결과 저장
        cost = criterion(hypothesis, Y) #분류결과와 레이블을 비교해 cost 계산
        cost.backward() #gradient 계산
        optimizer.step()
        avg_cost += cost / total_batch

print("Epoch:", "%04d" % (epoch+1), "cost =", "{:.9f}". format(avg_cost))


#test

with torch.no_grad(): # gradient를 계산 안하겠다
    X_test = mnist_test.test_data.view(-1, 28*28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = linear(X_test) #학습한 linear모델에 X_test 입력
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.floar().mean()
    print("Accuracy: ", accuracy.item())