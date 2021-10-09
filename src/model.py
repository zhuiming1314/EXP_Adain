from torch._C import _last_executed_optimized_graph
from torch.nn.modules.loss import L1Loss
import network
import criterion
import torch
import torch.nn as nn

class TwinsNet(nn.Module):
    def __init__(self, opts):
        super(TwinsNet, self).__init__()
        
        # parameters
        lr = 0.0001
        betas = (0.5, 0.999)
        weight_decay = 0.0001
        self.nz = 8

        # encoders
        self.encA_c = network.ContentEncoder(opts.input_dim_a)
        self.encB_c = network.ContentEncoder(opts.input_dim_b)
        self.encA_s = network.StyleEncoder(opts.input_dim_a, self.nz)
        self.encB_s = network.StyleEncoder(opts.input_dim_b, self.nz)

        # generator
        self.genA = network.Generator(self.encA_c.output_dim, opts.input_dim_a)
        self.genB = network.Generator(self.encB_c.output_dim, opts.input_dim_b)

        # discirminators
        self.disA = network.Discriminator(opts.input_dim_a, opts.dis_scale)
        self.disB = network.Discriminator(opts.input_dim_b, opts.dis_scale)

        # criterion
        self.calc_rec_loss = criterion.CalcRecLoss()
        self.calc_gan_loss = criterion.CalcGanLoss("lsgan", opts.gpu)

        # optimizers
        self.encA_c_opt = torch.optim.Adam(self.encA_c.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.encB_c_opt = torch.optim.Adam(self.encB_c.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.encA_s_opt = torch.optim.Adam(self.encA_s.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.encB_s_opt = torch.optim.Adam(self.encB_s.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.genA_opt = torch.optim.Adam(self.genA.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.genB_opt = torch.optim.Adam(self.genB.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    def set_gpu(self, gpu):
        self.gpu = gpu
        self.encA_c.cuda(self.gpu)
        self.encB_c.cuda(self.gpu)
        self.encA_s.cuda(self.gpu)
        self.encB_s.cuda(self.gpu)
        self.genA.cuda(self.gpu)
        self.genB.cuda(self.gpu)
        self.disA.cuda(self.gpu)
        self.disB.cuda(self.gpu)

    def initialize(self):
        self.encA_c.apply(network.weights_init("kaiming"))
        self.encB_c.apply(network.weights_init("kaiming"))
        self.encA_s.apply(network.weights_init("kaiming"))
        self.encB_s.apply(network.weights_init("kaiming"))
        self.genA.apply(network.weights_init("kaiming"))
        self.genB.apply(network.weights_init("kaiming"))
        self.disA.apply(network.weights_init("gaussian"))
        self.disB.apply(network.weights_init("gaussian"))

    def set_scheduler(self, opts, last_ep=0):
        self.encA_c_sch = network.get_scheduler(self.encA_c_opt, opts, last_ep)
        self.encB_c_sch = network.get_scheduler(self.encB_c_opt, opts, last_ep)
        self.encA_s_sch = network.get_scheduler(self.encA_s_opt, opts, last_ep)
        self.encB_s_sch = network.get_scheduler(self.encB_s_opt, opts, last_ep)
        self.genA_sch = network.get_scheduler(self.genA_opt, opts, last_ep)
        self.genB_sch = network.get_scheduler(self.genB_opt, opts, last_ep)
        self.disA_sch = network.get_scheduler(self.disA_opt, opts, last_ep)
        self.disB_sch = network.get_scheduler(self.disB_opt, opts, last_ep)

    def forward(self):
        # get real content encode
        self.real_content_a = self.encA_c.forward(self.input_a)
        self.real_content_b = self.encB_c.forward(self.input_b)
        # get real style encode
        self.real_style_a = self.encA_s.forward(self.input_a)
        self.real_style_b = self.encB_s.forward(self.input_b)
        # generate imgae of content and style 
        self.output_fake_a = self.genA.forward(self.real_content_b, self.real_style_a)
        self.output_fake_b = self.genB.forward(self.real_content_a, self.real_style_b)
        # get content encode from b and style encode from a
        self.fake_content_b = self.encB_c(self.output_fake_a)
        self.fake_style_a = self.encA_s(self.output_fake_a)
        # get content encode from a and style encode from b
        self.fake_content_a = self.encA_c.forward(self.output_fake_b)
        self.fake_style_b = self.encB_s.forward(self.output_fake_b)

        # generate image of real content a and fake style a
        self.rec_a1 = self.genA.forward(self.fake_content_a, self.real_style_a)
        self.rec_a2 = self.genA.forward(self.real_content_a, self.fake_style_a)

        # generate image of fake b content and real style b
        self.rec_b1 = self.genB.forward(self.fake_content_b, self.real_style_b)
        self.rec_b2 = self.genB.forward(self.real_content_b, self.fake_style_b)

        self.image_display = torch.cat((self.input_a[0:1].detach().cpu(), self.input_b[0:1].detach().cpu(),
                                        self.rec_a1[0:1].detach().cpu(), self.rec_a2[0:1].detach().cpu(),
                                        self.rec_b1[0:1].detach().cpu(), self.rec_b2[0:1].detach().cpu(),
                                        self.output_fake_b[0:1].detach().cpu(),
                                        self.output_fake_a.detach().cpu()))
    def backward_gen(self):
        loss_gen_a = 0
        loss_gen_b = 0

        outs_fake_a = self.disA.forward(self.output_fake_a)
        outs_fake_b = self.disB.forward(self.output_fake_b)
        loss_gen = 0
        for _, (out_a, out_b) in enumerate(zip(outs_fake_a, outs_fake_b)):
            o_a = nn.functional.sigmoid(out_a)
            all_ones_a = torch.ones_like(o_a).cuda(self.gpu)
            loss_gen_a += nn.functional.binary_cross_entropy(o_a, all_ones_a)

            o_b = nn.functional.sigmoid(out_b)
            all_ones_b = torch.ones_like(o_b).cuda(self.gpu)
            loss_gen_b += nn.functional.binary_cross_entropy(o_b, all_ones_b)

        #loss_gen *= 0.5
        print("loss_gen_a: {}, loss_gen_b:{}".format(loss_gen_a, loss_gen_b))
        return loss_gen_a, loss_gen_b

    def backward_rec(self):
        loss_rec_a = 0.5 * (nn.L1Loss()(self.input_a, self.rec_a1)\
                            + nn.L1Loss()(self.input_a, self.rec_a2))
        loss_rec_b = 0.5 * (nn.L1Loss()(self.input_b, self.rec_b1)\
                            + nn.L1Loss()(self.input_b, self.rec_b2))
        print("loss_rec_a: {}, loss_rec_b: {}".format(loss_rec_a, loss_rec_b))
        return loss_rec_a, loss_rec_b


    def update_enc_gen(self):
        self.encA_c_opt.zero_grad()
        self.encB_c_opt.zero_grad()
        self.encA_s_opt.zero_grad()
        self.encB_s_opt.zero_grad()
        self.genA_opt.zero_grad()
        self.genB_opt.zero_grad()

        loss_gen_a, loss_gen_b = self.backward_gen()
        loss_rec_a, loss_rec_b = self.backward_rec()
        loss_a = loss_gen_a + loss_rec_a
        loss_b = loss_gen_b + loss_rec_b

        loss_a.backward(retain_graph=True)
        loss_b.backward(retain_graph=True)

        self.encA_c_opt.step()
        self.encA_s_opt.step()
        self.genA_opt.step()
        self.encB_c_opt.step()
        self.encB_s_opt.step()
        self.genB_opt.step()

    def backward_dis(self):
        loss_dis_a = 0
        loss_dis_b = 0
        pred_fake_a = self.disA.forward(self.output_fake_a)
        pred_real_a = self.disA.forward(self.input_a)

        for _, (fake, real) in enumerate(zip(pred_fake_a, pred_real_a)):
            out_fake = nn.functional.sigmoid(fake)
            out_real = nn.functional.sigmoid(real)
            all_zeros = torch.zeros_like(out_fake).cuda(self.gpu)
            all_ones = torch.ones_like(out_real).cuda(self.gpu)
            loss_dis_fake = nn.functional.binary_cross_entropy(out_fake, all_zeros)
            loss_dis_real = nn.functional.binary_cross_entropy(out_real, all_ones)

            loss_dis_a += 0.5* (loss_dis_fake + loss_dis_real)


        pred_fake_b = self.disB.forward(self.output_fake_b)
        pred_real_b = self.disB.forward(self.input_b)


        for _, (fake, real) in enumerate(zip(pred_fake_b, pred_real_b)):
            out_fake = nn.functional.sigmoid(fake)
            out_real = nn.functional.sigmoid(real)
            all_zeros = torch.zeros_like(out_fake).cuda(self.gpu)
            all_ones = torch.ones_like(out_real).cuda(self.gpu)
            loss_dis_fake = nn.functional.binary_cross_entropy(out_fake, all_zeros)
            loss_dis_real = nn.functional.binary_cross_entropy(out_real, all_ones)

            loss_dis_b += 0.5 * (loss_dis_fake + loss_dis_real)

        print("loss_dis_a:{}, loss_dis_b:{}".format(loss_dis_a, loss_dis_b))
        return loss_dis_a, loss_dis_b

    def update_dis(self, input_a, input_b):
        self.input_a = input_a
        self.input_b = input_b

        # encode and generate first
        self.forward()

        # update dis
        self.disA_opt.zero_grad()
        self.disB_opt.zero_grad()
        loss_dis_a, loss_dis_b = self.backward_dis()
        loss_dis_a.backward(retain_graph=True)
        loss_dis_b.backward(retain_graph=True)
        self.disA_opt.step()
        self.disB_opt.step()

    def save_model(self, filename, ep, total_iter):
        state = {
            "encA_c": self.encA_c.state_dict(),
            "encA_s": self.encA_s.state_dict(),
            "genA": self.genA.state_dict(),
            "encB_c": self.encB_c.state_dict(),
            "encB_s": self.encB_s.state_dict(),
            "genB": self.genB.state_dict(),
            "disA": self.disA.state_dict(),
            "disB": self.disB.state_dict(),
            "encA_c_opt": self.encA_c_opt.state_dict(),
            "encA_s_opt": self.encA_s_opt.state_dict(),
            "genA_opt": self.genA_opt.state_dict(),
            "encB_c_opt": self.encB_c_opt.state_dict(),
            "encB_s_opt": self.encB_s_opt.state_dict(),
            "genB_opt": self.genB_opt.state_dict(),
            "disA_opt": self.disA_opt.state_dict(),
            "disB_opt": self.disB_opt.state_dict(),
            "ep": ep,
            "total_iter": total_iter
        }
        torch.save(state, filename)


    def resume(self, filename, train=True):
        checkpoint = torch.load(filename)

        if train:
            self.encA_c.load_state_dict(checkpoint["encA_c"])
            self.encA_s.load_state_dict(checkpoint["encA_s"])
            self.genA.load_state_dict(checkpoint["genA"])
            self.encB_c.load_state_dict(checkpoint["encB_c"])
            self.encB_s.load_state_dict(checkpoint["encB_s"])
            self.genB.load_state_dict(checkpoint["genB"])
            self.disA.load_state_dict(checkpoint["disA"])
            self.disB.load_state_dict(checkpoint["disB"])
            self.encA_c_opt.load_state_dict(checkpoint["encA_c_opt"])
            self.encA_s_opt.load_state_dict(checkpoint["encA_s_opt"])
            self.genA_opt.load_state_dict(checkpoint["genA_opt"])
            self.encB_c_opt.load_state_dict(checkpoint["encB_c_opt"])
            self.encB_s_opt.load_state_dict(checkpoint["encB_s_opt"])
            self.genB_opt.load_state_dict(checkpoint["genB_opt"])
            self.genA_opt.load_state_dict(checkpoint["disA_opt"])
            self.disB_opt.load_state_dict(checkpoint["disB_opt"])

            return checkpoint["ep"], checkpoint["total_iter"]
    
    def assemble_outputs(self):
        img_a = self.input_a.detach()
        img_b = self.input_b.detach()
        img_b_to_a = self.output_fake_a.detach()
        img_a_to_b = self.output_fake_b.detach()
        rec_a1= self.rec_a1.detach()
        rec_a2= self.rec_a2.detach()
        rec_b1 = self.rec_b1.detach()
        rec_b2 = self.rec_b2.detach()
        row1 = torch.cat((img_a[0:1, ::], img_b[0:1, ::], 
                            rec_a1[0:1, ::], rec_a2[0:1, ::], 
                            rec_b1[0:1, ::], rec_b2[0:1, ::],
                            img_a_to_b[0:1, ::], img_b_to_a[0:1, ::]), 3)
        return row1

    def forward_transfer(self, input_a, input_b):
        content_a, content_b = self.enc_c(input_a, input_b)
        style_a, style_b = self.enc_s(input_a, input_b)

        output = self.gen.forward_b(content_a, style_b)

        return output




