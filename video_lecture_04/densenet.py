"""
Implementation of DenseNet from scratch

Note that prebuilt and optimized densenets can be found in torchvision

"""
import torch
from torch import nn

class FirstBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """First convolution block of a Densenet. All hyperparameters are fixed.

        Parameters
        ----------
        in_channels : int
            Normally 3 for color images
        out_channels : int
            Number of initial features

        Returns
        -------
        Module
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=7,
                      stride=2,
                      padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class DenseLayer(nn.Module):
    def __init__(self,
                 num_input_features,
                 num_output_features,
                 k,
                 bottleneck_size=4):
        """Generic layer in a dense block

        Parameters
        ----------
        num_input_features : int
            Number of features in the input map
        k : int
            growth rate
        bottleneck_size : int
            Bottleneck size (4 in the paper)

        Returns
        -------
        Tensor
            output feature map of shape (bottleneck_size*k,:,:)

        """
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.bottleneck = nn.Conv2d(in_channels=num_input_features,
                                    out_channels=bottleneck_size*k,
                                    kernel_size=1,
                                    padding='same')
        self.bn2 = nn.BatchNorm2d(bottleneck_size*k)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=bottleneck_size*k,
                              out_channels=num_output_features,
                              kernel_size=3,
                              padding='same')

    def forward(self, x):
        h = x
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.bottleneck(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv(h)
        return h


class DenseBlock(nn.Module):
    def __init__(self, num_layers, k0, k):
        super().__init__()
        self.layers = nn.Sequential()
        for ell in range(num_layers-1):
            self.layers.append(DenseLayer(
                num_input_features=k0 + ell*k,
                num_output_features=k,
                k=k,
            ))
        self.layers.append(DenseLayer(
            num_input_features=k0 + (num_layers-1)*k,
            num_output_features=k0 + num_layers*k,
            k=k,
        ))

    def forward(self, x):
        h = [x]
        for ell, la in enumerate(self.layers):
            inp = torch.cat(h,1)
            res = la(inp)
            h.append(res)
        return res


class TransitionLayer(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=num_features,
                              out_channels=num_features//2,
                              kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))

    def forward(self, x):
        h = x
        h = self.norm(h)
        h = self.relu(h)
        h = self.conv(h)
        h = self.pool(h)
        return h

class ClassificationHead(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(num_features, num_classes)

    def forward(self, x):
        h = x
        h = self.pool(h)
        h = self.flatten(h)
        h = self.dense(h)
        return h


class DenseNet(nn.Module):
    def __init__(
            self,
            input_size,
            num_layers,
            k,
            num_dense_blocks,
            num_classes,
    ):
        super().__init__()
        self.first_block = FirstBlock(in_channels=3, out_channels=64)
        self.blocks = nn.Sequential()
        k0 = 64  # for the first dense block, this is the same as the number of out channels of the first block
        for i in range(num_dense_blocks):
            self.blocks.add_module(f"dense block {i}", DenseBlock(num_layers[i], k0, k))
            k0 = (k0 + num_layers[i]*k)//2  # for the next block
            if i < num_dense_blocks-1:
                self.blocks.add_module(f"transition {i}", TransitionLayer(num_features=k0*2))
        self.head = ClassificationHead(k0*2, num_classes)

    def forward(self, x):
        h = x
        h = self.first_block(h)
        # h = self.blocks(h)
        for block in self.blocks:
            h = block(h)
        h = self.head(h)
        return h


def main(opts):
    from visualize_network import visualize
    batch_size = 4
    input_size = 224
    input_data = torch.randn(batch_size,3,input_size,input_size)
    num_dense_blocks = 4
    num_layers = [6, 12, 24, 16]  # This corresponds to densenet121
    k = 32  # growth rate
    num_classes = 10
    model = DenseNet(input_size=input_size,
                     num_layers=num_layers,
                     k=k,
                     num_dense_blocks=num_dense_blocks,
                     num_classes=num_classes)
    visualize(model, 'DenseNet', input_data)
    # torch.onnx.export(model=model,
    #                   args=input_data,       # Dummy input
    #                   f="DenseNet.onnx",   # Filenane
    #                   export_params=False,       # Don't save parameters
    #                   # opset_version=17,          # Latest ONNX version
    #                   # verbose=True,              # Print extra info when saving
    #                   input_names=['input'],     # the model's input names
    #                   # output_names=['output'],   # the model's output names
    #                   # dynamic_axes={'input': {0: 'batch_size'},  # Axis 0 for batch
    #                   #               'output': {0: 'batch_size'}}
    #                   dynamo=True,
    #                   )


if __name__ == "__main__":
    import argparse
    from ipdb import launch_ipdb_on_exception
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=lambda prog:
                                     argparse.ArgumentDefaultsHelpFormatter(
                                         prog,max_help_position=52,width=90))
    parser.add_argument('--dataset', type=str,
                        choices=['galaxy'],
                        default='galaxy',
                        help='Which dataset')
    with launch_ipdb_on_exception():
        main(parser.parse_args())

