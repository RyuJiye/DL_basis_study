import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#for reproducibility
torch.manual_seed(1)

#(m,3)
x_train = torch.FloatTensor([[1,2,1],[1,3,2],[1,3,4],[1,5,5],[1,7,5],[1,2,5],[1,6,6],[1,7,7]])
#(m,)
y_train = torch.LongTensor([2,2,2,1,1,1,0,0])
#(m',3)
x_test = torch.FloatTensor([[2,1,1],[3,1,2],[3,3,4]])
#(m',)
y_test = torch.LongTensor([2,2,2])



class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,3)
    def forward(self, x):
        return self.linear(x) #|x| = (m,3) => (m,3)


model = SoftmaxClassifierModel()

optimizer = optim.SGD(model.parameters(), lr=1e5)
#lr은 발산하면(cost의 극솟값을 발견하지 못하고 지나치면) 작게, cost가 줄어들지 않으면 크게 조정

def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):

       #H(x)
       prediction = model(x_train)

       #cost
       cost = F.cross_entropy(prediction, y_train)

       #cost로 H(x)개선
       optimizer.zero_grad()
       cost.backward()
       optimizer.step()

       print('Epoch {:4d}/{} Cost: {:.6f}'.format(
           epoch, nb_epochs, cost.item()
       ))


def test(medoel, optimizer,x_test, y_test):
    prediction = model(x_test)
    predicted_classes = prediction.max(1)[1]
    correct_count = (predicted_classes == y_test).sum().item()
    cost = F.cross_entropy(prediction, y_test)

    print('Accuracy: {}% Cost: {:.6f}'.format(correct_count / len(y_test) * 100, cost.item()))

train(model, optimizer, x_train, y_train)
test(model, optimizer, x_test, y_test)

#train set에 대해선 비용이 줄어들지만, test set에 대해선 비용이 증가. 즉 overfitting된 상태 --> lr 조정


