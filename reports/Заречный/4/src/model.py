import torch
from torcheval.metrics import MulticlassConfusionMatrix

torch.manual_seed(42)


class Model(torch.nn.Module):

    def __init__(self, w1, t1, w2, t2):

        super().__init__()

        self.layer1 = torch.nn.Linear(in_features=21, out_features=21)
        self.layer1.weight = torch.nn.Parameter(w1.T)
        self.layer1.bias = t1


        self.layer2 = torch.nn.Linear(in_features=21, out_features=42)
        self.layer2.weight = torch.nn.Parameter(w2.T)
        self.layer2.bias = t2



        self.layer3 = torch.nn.Linear(in_features=42, out_features=10)
        self.layers = torch.nn.Sequential(
            self.layer1,
            torch.nn.ReLU(),
            self.layer2,
            torch.nn.ReLU(),
            self.layer3
        )

    def forward(self, x):
        return self.layers(x)

    def test(self, x_te_tensor: torch.Tensor, y_te_tensor1: torch.Tensor):
        with torch.inference_mode():
            self.eval()
            y_predicted = torch.softmax(self(x_te_tensor), dim=1)
            _, y_predicted = torch.max(y_predicted, 1)
            matrix = MulticlassConfusionMatrix(10)
            matrix.update(y_predicted, y_te_tensor1)
            print(matrix.compute())

            eye = torch.eye(10).type(torch.bool)
            print(f'Accuracy - {matrix.compute()[eye].sum() / 426 :.5f}')


class RBM(torch.nn.Module):

    def __init__(self, nv, nh):
        super(RBM, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(nv, nh) * 0.01)
        self.a = torch.nn.Parameter(torch.zeros(nv))
        self.b = torch.nn.Parameter(torch.zeros(nh))

    def sample_h(self, v):
        phv = torch.relu(torch.matmul(v, self.w) + self.b)
        h = phv #torch.bernoulli(phv)

        return h, phv

    def sample_v(self, h):
        pvh = torch.relu(torch.matmul(h, self.w.t()) + self.a)
        v = pvh # torch.bernoulli(pvh)

        return v, pvh

    def forward(self, v):
        h, phv = self.sample_h(v)
        k = 0
        # gibbs sampling
        for i in range(k):
            v, pvh = self.sample_v(h)
            h, phv = self.sample_h(v)
        v, pvh = self.sample_v(phv)

        return pvh

    def free_energy(self, v):
        vt = torch.matmul(v, self.a)
        ht = torch.sum(torch.log(1 + torch.exp(torch.matmul(v, self.w) + self.b)), dim=1)

        return -(vt + ht)