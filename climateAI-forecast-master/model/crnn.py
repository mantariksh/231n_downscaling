
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Convolutional recurrent neural net class, loss and metrics.
'''

class CRNN(nn.Module):

    def __init__(self, d=128):

        # convolution layers arguments: input_channels, output_channels, filter_size, stride, padding.
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

        self.lstm1 = nn.LSTM(input_size=1, hidden_size=64) # input_dim = 1, output_dim = 64
        self.lstm2 = nn.LSTM(input_size=64,hidden_size=1) # input_dim = 64, output_dim = 1

    def forward(self, input):
        x = F.relu(self.conv1(input), 0.2)
        x = F.relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.relu(self.conv5(x))

        # lstm inputs: 3D, 1st sequence (time_steps), 2nd minibatch instances, 3rd elements of the inputs (vectors of each time_step).
        lstm1_out, hidden_state1 = self.lstm1(input, (h0, c0))
        lstm2_out, hidden_state2 = self.lstm2(input, (h0, c0))

        return lstm1_out


def loss_fn():

    return nn.MSELoss()


'''
Here we can add the metrics we decide to use.
'''
