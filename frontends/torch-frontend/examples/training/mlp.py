import torch
from torch import nn
from byteir_backend import byteir_compile_fx

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 20)
        self.linear3 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        x = self.linear3(x)
        return x


model = MLP().cuda()
model1 = MLP().cuda()

model1.load_state_dict(model.state_dict())

compiled_model = torch.compile(model1, backend=byteir_compile_fx)

x = torch.randn(2, 10, device='cuda', requires_grad=True)
x1 = x.detach().clone().requires_grad_()

output = compiled_model(x)
expected_output = model(x1)

print(output)
print(expected_output)
torch.testing.assert_close(output, expected_output)

output.sum().backward()
expected_output.sum().backward()

for param1, param2 in zip(model.parameters(), compiled_model.parameters()):
    torch.testing.assert_close(param1.grad, param2.grad)

torch.testing.assert_close(x.grad, x1.grad)
