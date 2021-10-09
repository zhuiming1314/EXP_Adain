import functools
import torch
import torch.nn as nn
import net_module as net
from torch.optim import lr_scheduler
import net_module as net
import math
import torch.nn.init as init

###############################################################
#-------------------------Encoders----------------------------#
###############################################################
class ContentEncoder(nn.Module):
    def __init__(self, input_dim):
        super(ContentEncoder, self).__init__()
        # content encoder
        enc_c = []
        n_in = input_dim
        n_out = 64

        enc_c += [net.Conv2dBlock(n_in, n_out, 7, 1, 3, norm="in", activation="relu", pad_type="reflect")]
        # downsampling blocks
        for _ in range(1, 3):
            n_in = n_out
            n_out *= 2
            enc_c += [net.Conv2dBlock(n_in, n_out, 4, 2, 1, norm="in", activation="relu", pad_type="reflect")]
        # residual blocks
        enc_c += [net.ResBlocks(4, n_out, norm="in", activation="relu", pad_type="reflect")]
        self.output_dim = n_out

        self.enc_c = nn.Sequential(*enc_c)
    
    def forward(self, x):
        return self.enc_c(x)

class StyleEncoder(nn.Module):
    def __init__(self, input_dim, output_nc):
        super(StyleEncoder, self).__init__()

        # style encoder of domain a
        enc_s = []
        n_in = input_dim
        n_out = 64

        enc_s += [net.Conv2dBlock(n_in, n_out, 7, 1, 3, norm="in", activation="relu", pad_type="reflect")]
        for _ in range(1, 3):
            n_in = n_out
            n_out *= 2
            enc_s += [net.Conv2dBlock(n_in, n_out, 4, 2, 1, norm="in", activation="relu", pad_type="reflect")]
        
        n_in = n_out
        for _ in range(1, 3):
            enc_s += [net.Conv2dBlock(n_in, n_out, 4, 2, 1, norm="in", activation="relu", pad_type="reflect")]
        enc_s += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        enc_s += [nn.Conv2d(n_out, output_nc, 1, 1, 0)]

        self.enc_s = nn.Sequential(*enc_s)

    def forward(self, x):
        return self.enc_s(x)

##############################################################
#-----------------Generators/Decoders------------------------#
##############################################################
class Generator(nn.Module):
    def __init__(self, n_in, output_dim):
        super(Generator, self).__init__()
        
        gen = []
        # AdaIN residual blocks
        gen += [net.ResBlocks(4, n_in, norm="adain", activation="relu", pad_type="reflect")]
        # upsampling blocks
        for _ in range(1, 3):
            n_out = n_in // 2
            gen += [nn.Upsample(scale_factor=2),
                           net.Conv2dBlock(n_in, n_out, 5, 1, 2, norm='ln', activation="relu", pad_type="reflect")]
            n_in = n_out

        # use reflection padding in the last conv layer
        gen += [net.Conv2dBlock(n_in, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type="reflect")]
        self.gen = nn.Sequential(*gen)
        self.mlp = net.MLP(8, self.get_num_adain_params(), 256, 3, norm='none', activ="relu")
    
    def forward(self, content, style):
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params)
        return self.gen(content)

    def assign_adain_params(self, adain_params):
        # assign the adain_params to the AdaIN layers in model
        for m in self.gen.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]
    def get_num_adain_params(self):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in self.gen.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

#############################################################
#--------------------Discriminator--------------------------#
#############################################################
class Discriminator(nn.Module):
    def __init__(self, n_in, n_scale=3, n_layer=4, norm="None"):
        super(Discriminator, self).__init__()

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.Diss = nn.ModuleList()
        n_out = 64
        for _ in range(n_scale):
            self.Diss.append(self._make_net(n_in, n_out, n_layer, norm))
        
    def _make_net(self, n_in, n_out, n_layer, norm):
        model = []

        model.append(net.Conv2dBlock(n_in, n_out, kernel_size=4, stride=2, padding=1, norm=norm))

        for _ in range(1, n_layer):
            n_in = n_out
            n_out *= 2
            model.append(net.Conv2dBlock(n_in, n_out, kernel_size=4, stride=2, padding=1, norm=norm))
       
        model.append(nn.Conv2d(n_out, 1, kernel_size=1, stride=1, padding=0))

        return nn.Sequential(*model)

    def forward(self, x):
        outs = []
        for dis in self.Diss:
            outs.append(dis(x))
            x = self.downsample(x)

        return outs


###############################################################
#---------------------------Basic Functions-------------------#
###############################################################
def weights_init(init_type="gaussian"):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

def get_scheduler(optimizer, opts, cur_ep=-1):
    if opts.lr_policy == "lambda":
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    else:
        return NotImplementedError("no such learn rate policy")
    return scheduler