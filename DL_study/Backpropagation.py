import torch
from torch import device

#XOR 예제
X = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]]).to(device) #4개의 데이터 종류
Y = torch.FloatTensor([[0],[1],[1],[0]]).to(device) #정답


#Linear을 사용하지 않고 가중치와 편향을 직접 설정
#nn.Linear을 2개 설정한 것과 같음
w1 = torch.Tensor(2,2).to(device)
b1 = torch.Tensor(2).to(device)
w2 = torch.Tensor(2,1).to(device)
b2 = torch.Tensor.to(device)

def  sigmoid(X):
    return 1.0 / (1.0 + torch.exp(-x))

#sigmoid 한수를 미분한 함수
def sigmoid_prime(X):
    return sigmoid(X) * (1 - sigmoid(X))


for step in range(10001):
    #forward
    l1 = torch.add(torch.matmul(X, w1), b1)
    a1 = sigmoid(l1)
    l2 = torch.add(torch.matmul(a1, w2), b2)
    Y_pred = sigmoid(l2)

    #loss 계산, binary cross entropy
    cost = -torch.mean(Y * torch.log(Y_pred) + (1 - Y) * torch.log(1 - Y_pred))

    #back prop(chain rule)
    #Loss derivative
    # bias cross entropy 미분식
    d_Y_pred = (Y_pred - Y) / (Y_pred * (1.0 - Y_pred) + 1e-7) # 1e-7은 분모가 0이 되는걸 막는 역할
    # Layer 2
    d_l2 = d_Y_pred * sigmoid_prime(l2)
    d_b2 = l2 #편향에 대한 미분
    d_w2 = torch.matmul(torch.transpose(a1,0,1), d_b2) #가중치에 대한 미분, transpose로 a1행렬의 0번 축과 1번 축 전치
    #Layer 1
    d_a1 = torch.matmul(d_b2, torch.transpose(w2,0,1))
    d_l1 = d_a1 * sigmoid_prime(l1)
    d_b1 = d_l1
    d_w1 = torch.matmul(torch.transpose(X, 0, 1), d_b1)

    #Weight update, gradient minimization
    w1 = w1 - learning_rate * d_w1
    b1 = b1 - learning_rate * torch.mean(d_b1, 0)
    w2 = w2 - learning_rate * d_w2
    b2 = b2 - learning_rate * torch.mean(d_b2, 0)

    if step % 100 == 0: #100번째 스텝마다 cost 출력
        print(step, cost.item())



# 학습이 될 수록 cost감소
