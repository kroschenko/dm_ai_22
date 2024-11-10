from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from torch import Tensor


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


model = Model()

lr = 1e-2
epochs = range(100)

model.set_train_layers(0)
optim = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()
model.train()

for epoch in epochs:
    y_predicted = model(X_tr_tensor) #torch.softmax(model(X_tr_tensor), dim=1)
    loss = loss_fn(y_predicted, y_tr_tensor)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f'#{epoch:5d}  error = {loss:.5f}')

model.test(X_te_tensor, y_te_tensor1)
