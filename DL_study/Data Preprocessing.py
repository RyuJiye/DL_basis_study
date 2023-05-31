import torch
import torch.nn as nn
import torch.optim as optim



x_train = torch.FloatTensor([[73,80,75],[93,88,93],[89,91,90],[96,98,100],[73,66,70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

mu = x_train.mean(dim = 0) #모평균
sigma = x_train.std(dim = 0)#모표준편차
norm_x_train = (x_train - mu) / sigma #정규화
print(norm_x_train)

class MultivariateLinearRegressionModel(nn.Module):
    # |x_train| = (m,3)
    # |prediction| = (m,1)
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1) #3개의 element를 받아서 하나의 element 반환

    def forward(self, x):
        return self.linear(x)

model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr = 1e-1)

