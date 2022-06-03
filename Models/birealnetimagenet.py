import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


__all__ = ['birealnet18', 'birealnet34']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        size = input.size()
        input = input.sign()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class SignWithnGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fc_out):
        '''
            x.shape : [Cout,Cin,k,k]
        '''
        return torch.sign(x)

    @staticmethod
    def backward(ctx, g):
        '''
            g.shape:[N,1]
        '''
        g = g.view(-1,1)
        g_tensor = torch.cat([g for _ in range(2)],axis=1) # be same with net_out_features 
        return None, g_tensor
class GatedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        '''
        '''
        ctx.input = x
        return x

    @staticmethod
    def backward(ctx, g):
        '''
            g.shape:[N,1]
        '''
        x = ctx.input
        g = g * x.lt(1) * x.gt(-1)
        return g

class MetaConv_v2(nn.Module):
    '''
    w -> multiple values for gradient bp; only for 1bit quantization
    '''
    def __init__(self, hidden_size = 100, num_layers = 2, features = 1, use_nonlinear='tanh', nbits=1):
        super(MetaConv_v2, self).__init__()
        self.nbits = nbits
        net_out_features = 2 ## use 8 params to avg the gradient

        self.use_nonlinear = use_nonlinear
        self.network = nn.Sequential()
        for layer_idx in range(num_layers):
            in_features = features if layer_idx == 0 else hidden_size
            out_features = net_out_features if layer_idx == (num_layers-1) else hidden_size
            self.network.add_module('Linear%d' %layer_idx, nn.Linear(in_features=in_features, out_features=out_features, bias=False))
            # self.network.add_module('Tanh%d' %layer_idx, nn.Tanh())
            if layer_idx != (num_layers-1):
                if self.use_nonlinear == 'relu':
                    self.network.add_module('ReLU%d' %layer_idx, nn.ReLU())
                elif self.use_nonlinear == 'tanh':
                    self.network.add_module('Tanh%d' %layer_idx, nn.Tanh())
                else:
                    # raise NotImplementedError
                    pass
        # self.network[0].weight.data = torch.tensor([[0.5],[0.5]])

    def forward(self, x):
        # if self.network[0].weight.data[0][0] > 50:
        #     self.network[0].weight.data[0][0] = 50
        # if self.network[0].weight.data[1][0] > 50:
        #     self.network[0].weight.data[1][0] = 50

        xshape = x.shape
        x_flatten = x.view(-1,1)
        fc_out = self.network(x_flatten) ## shape[N,8]
        fc_out = GatedLinear.apply(fc_out)
        out = SignWithnGradient.apply(x, fc_out)

        return out

def linearFunc(x, nbits):
    '''
        for 1bit :y=2x-1, y(0)=-1, y(1)=1
        else: y=x-2**(nbits-1)
    '''
    if nbits == 1:
        return 2*x - 1
    else:
        return x-2**(nbits-1)

class ArgmaxWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, nbits):
        '''
            x.shape : [N,2]
        '''
        v,i = torch.max(x,axis=1)
        argmax_res = i.view(-1,1).float()
        # argmax_res = x.argmax(axis=1).view(-1,1).float()
        ctx.input = x
        ctx.nbits = nbits
        return argmax_res

    @staticmethod
    def backward(ctx, g):
        '''
            g.shape:[N,1]
        '''
        g_tensor = torch.cat([g for _ in range(2**ctx.nbits)],axis=1)
        return g_tensor, None


class MetaConv_v3(nn.Module):

    def __init__(self, hidden_size = 100, num_layers = 1, features = 1, use_nonlinear='none', nbits=1):
        super(MetaConv_v3, self).__init__()
        self.nbits = nbits
        # net_out_features = 2 ** nbits ## 2-class classification problem
        net_out_features = 2

        self.use_nonlinear = use_nonlinear
        self.network = nn.Sequential()
        for layer_idx in range(num_layers):
            in_features = features if layer_idx == 0 else hidden_size
            out_features = net_out_features if layer_idx == (num_layers-1) else hidden_size
            self.network.add_module('Linear%d' %layer_idx, nn.Linear(in_features=in_features, out_features=out_features, bias=False))
            if layer_idx != (num_layers-1):
                if self.use_nonlinear == 'relu':
                    self.network.add_module('ReLU%d' %layer_idx, nn.ReLU())
                elif self.use_nonlinear == 'tanh':
                    self.network.add_module('Tanh%d' %layer_idx, nn.Tanh())
                else:
                    # raise NotImplementedError
                    pass

    def forward(self, x):
        xshape = x.shape
        x = x.view((-1, 1))
        out = self.network(x)  ## shape : [N, out_features]
        # out = out.softmax(dim=1)
        # out = out * 1
        # out = softargmax(out)  ## shape : [N, 1]

        ## either argmax
        out = ArgmaxWithGradient.apply(out, self.nbits)
        ## or gumble-softmax
        # out = gumbel_softmax(out, hard=True)
        # out = ArgmaxWithGradient.apply(out, self.nbits)
        # out = softargmax(out)

        out = linearFunc(out, self.nbits)
        out = out.view(xshape)

        return out

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out



class HardBinaryConvMeta(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, use_meta = 'Conv'):
        super(HardBinaryConvMeta, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        # self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.weights = nn.Parameter(torch.rand((out_chn, in_chn, kernel_size, kernel_size)), requires_grad=True)

        nn.init.xavier_uniform_(self.weights)

    def forward(self, x, meta_net):
        w = self.weights
        real_weights_center = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)

        real_weights = meta_net(real_weights_center)
        real_weights = real_weights.view(self.shape)


        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights_center),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        # binary_weights_no_grad = torch.sign(real_weights)

        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)


        return y


class MetaConv2d(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=False):
        super(MetaConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        # self.activator = BinaryActivation()
        # self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.weights = nn.Parameter(torch.rand((out_chn, in_chn, kernel_size, kernel_size)), requires_grad=True)
        
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x, meta_net):
        w = self.weights
        real_weights_center = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights_center),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights = meta_net(real_weights_center)
        binary_weights = binary_weights * scaling_factor

        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)


        return y

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.binary_activation = BinaryActivation()
        self.binary_conv = MetaConv2d(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.nonlinear = nn.PReLU(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x, meta_net):
        residual = x

        out = self.binary_activation(x)
        out = self.binary_conv(out, meta_net)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlinear(out)

        return out

class BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(BiRealNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.nonlinear = nn.PReLU(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.meta_net = MetaConv_v2()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nonlinear(x)
        x = self.maxpool(x)

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                x = block(x, self.meta_net)


        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def birealnet18(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = BiRealNet(BasicBlock, [4, 4, 4, 4], **kwargs)
    return model


def birealnet34(pretrained=False, **kwargs):
    """Constructs a BiRealNet-34 model. """
    model = BiRealNet(BasicBlock, [6, 8, 12, 6], **kwargs)
    return model

