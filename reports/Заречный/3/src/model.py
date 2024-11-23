import torch
from torcheval.metrics import MulticlassConfusionMatrix

torch.manual_seed(42)


class Model(torch.nn.Module):
    # layer1: torch.nn.Linear
    # layer2: torch.nn.Linear
    # layer3: torch.nn.Linear


    def __init__(self):

        super().__init__()

        self.layer1 = torch.nn.Linear(in_features=21, out_features=21).requires_grad_(False)
        self.layer2 = torch.nn.Linear(in_features=21, out_features=42).requires_grad_(False)
        self.layer3 = torch.nn.Linear(in_features=42, out_features=10).requires_grad_(True)



        self.layer1_ae = torch.nn.Linear(in_features=21, out_features=21).requires_grad_(True)


        self.layer2_ae = torch.nn.Linear(in_features=42, out_features=21).requires_grad_(True)




        self.layers = torch.nn.Sequential(
            self.layer1,
            torch.nn.ReLU(),
            self.layer2,
            torch.nn.ReLU(),
            self.layer3
        )

    def set_train_layers(self, num: int | None = None):

        match num:
            case None:
                self.layers = torch.nn.Sequential(
                    self.layer1,
                    torch.nn.ReLU(),

                    self.layer2,
                    torch.nn.ReLU(),

                    self.layer3,

                )
                self.layer1.requires_grad_(False)
                self.layer2.requires_grad_(False)

            case 0:
                self.layers = torch.nn.Sequential(
                    self.layer1,
                    torch.nn.ReLU(),

                    self.layer2,
                    torch.nn.ReLU(),

                    self.layer3,

                )
                self.layer1.requires_grad_(True)
                self.layer2.requires_grad_(True)


            case 1:
                self.layers = torch.nn.Sequential(
                    self.layer1,
                    torch.nn.ReLU(),
                    self.layer1_ae
                )

                self.layer1.requires_grad_(True)
                self.layer2.requires_grad_(False)

            case 2:
                self.layers = torch.nn.Sequential(
                    self.layer2,
                    torch.nn.ReLU(),
                    self.layer2_ae,
                )

                self.layer1.requires_grad_(False)
                self.layer2.requires_grad_(True)



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
