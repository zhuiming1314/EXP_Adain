import argparse

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def _add_args(self):
        self.parser.add_argument("--dataroot", type=str, required=True, help="path of dataset")
        self.parser.add_argument("--phase", type=str, default="train", help="train or test for loading data")
        self.parser.add_argument("--batch_size", type=int, default=2, help="batch size")
        self.parser.add_argument("--resize_size", type=int, default=256, help="resize size of image")
        self.parser.add_argument("--crop_size", type=int, default=216, help="crop size of imgae")
        self.parser.add_argument("--input_dim_a", type=int, default=3, help="input channel of images in domain a")
        self.parser.add_argument("--input_dim_b", type=int, default=3, help="input channel of images in domain b")
        self.parser.add_argument("--n_threads", type=int, default=8, help="thread number for dataloader")
        self.parser.add_argument("--no_flip", action="store_true", help="specified if no flipping")

        #output setting
        self.parser.add_argument("--output_name", type=str, required=True, help="output folder name")
        self.parser.add_argument("--img_save_dir", type=str, default="../result", help="path for saving output")
        self.parser.add_argument("--img_save_freq", type=int, default=10, help="freq of saving output image")
        self.parser.add_argument("--display_dir", type=str, default="../logs", help="path for saving display result")
        self.parser.add_argument("--display_freq", type=int, default=1, help="freq (iteration) of display")
        self.parser.add_argument("--checkpoints", type=str, default="../checkpoints", help="path for saving checkpoints")
        self.parser.add_argument("--checkpoint_freq", type=int, default=10, help="checkpoint (epoch) of saving model")

        #train setting
        self.parser.add_argument("--dis_scale", type=int, default=3, help="scale of discriminator")
        self.parser.add_argument("--lr_policy", type=str, default="lambda", help="type of learn rate decay")
        self.parser.add_argument("--n_ep", type=int, default=1200, help="number of epochs")
        self.parser.add_argument("--n_ep_decay", type=int, default=600, help="epoch start decay learning rate, set -1 if no decay")
        self.parser.add_argument("--resume", type=str, help="resume training")
        self.parser.add_argument("--d_iter", type=int, default=3, help="# of iterations for updating content discriminator")
        self.parser.add_argument("--gpu", type=int, default=0, help="gpu device")


    def parse(self):
        self._add_args()
        self.args  = self.parser.parse_args()
        return self.args
