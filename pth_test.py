import torch
import torch.nn as nn

# Define the model structure (must match the structure of the model when the .pth file was saved)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(in_features=10, out_features=20)
        # Add more layers or match the architecture

    def forward(self, x):
        x = self.layer1(x)
        # Implement forward pass consistent with the saved model
        return x

# Create an instance of the model
model = MyModel()

# Load the state dictionary from the .pth file
state_dict = torch.load('policy_network.pth')

print(state_dict)

# Load the state dictionary into the model
# model.load_state_dict(state_dict)

# Now the model can be used for inference or further training