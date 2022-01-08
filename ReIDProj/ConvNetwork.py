import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, in_feature_2d, out_feature):
        self.in_feature_2d = in_feature_2d
        super(ConvNet, self).__init__()
        self.layer_2d = []
        # Internal protected method
        self. _set_network(in_feature_2d, out_feature)

    def _set_network(self, in_feature_2d, out_feature):
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(
            5, 5), stride=1, padding="same")
        self.layer_2d.append(self.conv1)

        self.act1 = nn.ReLU()
        self.layer_2d.append(self.act1)

        self.pooling1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.layer_2d.append(self.pooling1)

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), padding="same")
        self.layer_2d.append(self.conv2)

        self.act2 = nn.ReLU()
        self.layer_2d.append(self.act2)

        self.pooling2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.layer_2d.append(self.pooling2)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=(3, 3), padding="same")
        self.layer_2d.append(self.conv3)

        self.act3 = nn.ReLU()
        self.layer_2d.append(self.act3)

        # global max pooling with pooling size of the input
        self.pooling3 = nn.MaxPool2d(kernel_size=(7, 7))
        self.layer_2d.append(self.pooling3)

        self.featureFC = nn.Linear(in_features=32, out_features=2, bias=True)
        self.outputFC = nn.Linear(
            in_features=2, out_features=out_feature, bias=True)

    def forward(self, x):
        feature = self.feature_extraction(x)
        return self.outputFC(feature)

    def feature_extraction(self, x):
        input = x
        for layer in self.layer_2d:
            output = layer(input)
            input = output
        output = self.featureFC(output.squeeze())
        return output

    def predict_prob(self, x):
        output = self.forward(x)
        return output.softmax(dim=1)  # apply softmax to each row, each batch

    def predict(self, x):
        prob = self.predict_prob(x)
        return prob.argmax(dim=1)


class MLPNet(nn.Module):
    def __init__(self, in_feature_2d, out_feature):
        self.in_feature_2d = in_feature_2d
        super(MLPNet, self).__init__()
        self.layers = nn.Sequential([
            nn.Linear(in_features=in_feature_2d[0]
                      * in_feature_2d[1], out_features=200),
            nn.Linear(in_features=200, out_features=200),
            nn.Linear(in_features=200, out_features=10)
        ])

    def forward(self, x):
        return self.layers(x)

    def predict_prob(self, x):
        output = self.forward(x)
        return output.softmax(dim=1)  # apply softmax to each row, each batch

    def predict(self, x):
        prob = self.predict_prob(x)
        return prob.argmax(dim=1)
