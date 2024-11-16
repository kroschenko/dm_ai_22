import torch.nn
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from torch import Tensor, tensor
from torcheval.metrics import BinaryConfusionMatrix


from model import *

cardiotocography = fetch_ucirepo(id=193)

X = cardiotocography.data.features
y = cardiotocography.data.targets


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=423)

X_tr_tensor = Tensor(X_train.values)
X_te_tensor = Tensor(X_test.values)


y_tr_tensor1 = Tensor(y_train['CLASS'].values).type(torch.int64) - 1
y_te_tensor1 = Tensor(y_test['CLASS'].values).type(torch.int64) - 1

num_classes = 10

y_tr_tensor = torch.nn.functional.one_hot(y_tr_tensor1, num_classes).type(torch.float64)
y_te_tensor = torch.nn.functional.one_hot(y_te_tensor1, num_classes).type(torch.float64)




lr = 1e-4
epochs = 25


rbm1 = RBM(21, 21)
optim1 = torch.optim.Adam(rbm1.parameters(), lr=lr)
torch.optim.SGD(rbm1.parameters(), lr=lr)
loss_fn1 = torch.nn.MSELoss()
rbm1.train()

for epoch in range(epochs):
    y_predicted = rbm1(X_tr_tensor)
    loss1 = loss_fn1(y_predicted, X_tr_tensor)
    optim1.zero_grad()
    loss1.backward()
    optim1.step()
    print(f'#{epoch:5d}  error = {loss1:.5f}')
print(f'-' * 50)


X_tr_tensor1 = torch.clone(rbm1.sample_h(X_tr_tensor)[0])
X_tr_tensor1 = X_tr_tensor1.detach().clone()

rbm2 = RBM(21, 42)
optim2 = torch.optim.Adam(rbm2.parameters(), lr=lr)
loss_fn2 = torch.nn.MSELoss()
rbm2.train()

for epoch in range(epochs):
    y_predicted = rbm2(X_tr_tensor)
    loss2 = loss_fn2(y_predicted, X_tr_tensor)
    optim2.zero_grad()
    loss2.backward()
    optim2.step()
    print(f'#{epoch:5d}  error = {loss2:.5f}')
print(f'-' * 50)

model = Model(rbm1.w, rbm1.b, rbm2.w, rbm2.b)


optim = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = torch.nn.CrossEntropyLoss()
model.train()

for epoch in range(2*epochs):
    y_predicted = model(X_tr_tensor)
    loss = loss_fn(y_predicted, y_tr_tensor)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f'#{epoch:5d}  error = {loss:.5f}')

model.test(X_te_tensor, y_te_tensor1)


