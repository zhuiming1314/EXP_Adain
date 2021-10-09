import os
import torchvision
from tensorboardX import SummaryWriter
import numpy as np
import PIL


class Saver():
    def _mkdir_not_exists(self, dir):
        # make dir
        if not os.path.exists(dir):
            os.makedirs(dir)

    def __init__(self, args):
        self.display_dir = os.path.join(args.display_dir, args.output_name)
        self.model_dir = os.path.join(args.checkpoints, args.output_name)
        self.image_dir = os.path.join(args.img_save_dir, args.output_name)
        self.display_freq = args.display_freq
        self.img_save_freq = args.img_save_freq
        self.model_save_freq = args.checkpoint_freq

        self._mkdir_not_exists(self.display_dir)
        self._mkdir_not_exists(self.model_dir)
        self._mkdir_not_exists(self.image_dir)

        # create tensorboard writer
        self.writer = SummaryWriter(logdir=self.display_dir)

    # write losses and images to tensorboard
    def write_display(self, total_iter, model):
        if (total_iter + 1) % self.display_freq == 0:
            # write loss
            members = [attr for attr in dir(model) if not callable(getattr(model, attr)) and not attr.startswith("__") and "loss" in attr]
            for m in members:
                self.writer.add_scalar(m, getattr(model, m), total_iter)
            
            # write image
            images = torchvision.utils.make_grid(model.image_display, nrow=model.image_display.size(0)//2)/2 + 0.5
            self.writer.add_image("image", images, total_iter)

    def write_img(self, ep, model):
        if (ep + 1) % self.img_save_freq == 0:
            assembled_images = model.assemble_outputs()
            img_filename = "%s/gen_%05d.jpg" % (self.image_dir, ep)

            torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)

        elif ep == -1:
            assembled_images = model.assemble_outputs()
            img_filename = "%s/gen_last.jpg" % self.image_dir

            torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)

    # save model
    def write_model(self, ep, total_iter, model):
        print("\n------------save model at ep %d --------" %(ep))
        if ep == -1:
            model.save_model("%s/last.pth" % self.model_dir, ep, total_iter)
        elif (ep + 1) % self.model_save_freq == 0:
            model.save_model("%s/%05d.pth" % (self.model_dir, ep), ep, total_iter)

