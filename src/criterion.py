import torch
import torch.nn as nn
import torch.nn.functional as F

class CalcRecLoss():
    def __init__(self):
        super(CalcRecLoss, self).__init__()
        self.loss = nn.L1Loss()
    def __call__(self, origin, output):
        return self.loss(origin, output)

class CalcGanLoss():
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_lable=0.0, gpu=0):
        super(CalcGanLoss, self).__init__()

        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_lable

        self.gan_mode = gan_mode
        self.gpu = gpu

        if self.gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_mode in ["wgan", "wgangp", "hinge", "logistic"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %d not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            if not hasattr(self, "target_real_tensor"):
                self.target_real_tensor = torch.ones_like(prediction).cuda(self.gpu)
            target_tensor = self.target_real_tensor
        else:
            if not hasattr(self, "target_fake_tensor"):
                self.target_fake_tensor = torch.zeros_like(prediction).cuda(self.gpu)
            target_tensor = self.target_fake_tensor
        return target_tensor


    def __call__(self, prediction, target_is_real, is_updating_d=None):
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode.find("wgan") != -1:
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == "hinge":
            if target_is_real:
                loss = F.relu(1 - prediction) if is_updating_d else -prediction
            else:
                loss = F.relu(1 + prediction) if is_updating_d else prediction
            loss = loss.mean()
        elif self.gan_mode == 'logistic':
            pass
        else:
            print("error")

        return loss