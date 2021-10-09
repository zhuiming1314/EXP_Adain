import os
import torch.utils.data as data
import PIL
import torchvision.transforms as transforms
import random

class Dataset(data.Dataset):
    def __init__(self, args):
        self.dataroot = args.dataroot

        # domain A
        images_a = os.listdir(os.path.join(self.dataroot, args.phase + 'A'))
        self.imgs_a = [os.path.join(self.dataroot, args.phase + 'A', x) for x in images_a]

        # domain B
        images_b = os.listdir(os.path.join(self.dataroot, args.phase + 'B'))
        self.imgs_b = [os.path.join(self.dataroot, args.phase + 'B', x) for x in images_b]

        self.size_a = len(self.imgs_a) # A域图像数量N
        self.size_b = len(self.imgs_b) # B域图像数量M
        self.size = max(self.size_a, self.size_b)
        self.input_dim_a = args.input_dim_a #图像的维度
        self.input_dim_b = args.input_dim_b
        #transform
        self.transforms = transforms.Compose([
            transforms.Resize(args.resize_size, PIL.Image.BICUBIC),
            transforms.RandomCrop(args.crop_size) if args.phase == "train" \
                else transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.size == self.size_a:
            img_a = self.load_img(self.imgs_a[index], self.input_dim_a)
            img_b = self.load_img(self.imgs_b[random.randint(0, self.size_b - 1)], self.input_dim_b)
        else:
            img_a = self.load_img(self.imgs_a[random.randint(0, self.size_a - 1)], self.input_dim_a)
            img_b = self.load_img(self.imgs_b[index], self.input_dim_b)

        return img_a, img_b

    def load_img(self, img_path, input_dim):
        '''inner func to help load img'''
        img = self.transforms(PIL.Image.open(img_path).convert("RGB"))
        if input_dim == 1:
            # 转为1通道
            img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
            img = img.unsqueeze(0) # 在指定的位置增加一个维度
        return img
