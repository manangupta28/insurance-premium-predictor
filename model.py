import torch.nn as nn

class InsuranceModel(nn.Module):
  def __init__(self,input_dim):
    super(InsuranceModel , self).__init__()

    self.fc1 = nn.Linear(input_dim , 64)
    self.fc2 = nn.Linear(64,32)
    self.output = nn.Linear(32,1)
    self.relu = nn.ReLU()

  def forward(self,x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.output(x)

    return x
