import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.input_layer = nn.Linear(2, 128)
        self.hidden_layer = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        return self.output_layer(x)



    def save_the_model(self, weights_filename='models/latest.pt'):
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(self.state_dict(), weights_filename)

    def load_the_model(self, weights_filename='models/latest.pt'):
        try:
            self.load_state_dict(torch.load(weights_filename))
            print("Model Load Sucessfully")
        except:
            print("failed to load the model")