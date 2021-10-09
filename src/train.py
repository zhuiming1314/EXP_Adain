import torch
from options import TrainOptions
from model import TwinsNet
from dataset import Dataset
from saver import Saver

def main():
    parser = TrainOptions()
    args = parser.parse()

    print("\n--------load dataset---------")
    dataset = Dataset(args)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)

    print("\n--------load model-----------")
    model = TwinsNet(args)
    model.set_gpu(args.gpu)
    if args.resume is None:
        model.initialize()
        ep0 = -1
        total_iter = 0
    else:
        ep0, total_iter = model.resume(args.resume)
    
    ep0 += 1
    print("\nstart training at epoch %d" % ep0)

    # saver for display and output
    saver = Saver(args)

    # train
    print("\n-----------train------------")
    max_iter = 500000
    for ep in range(ep0, args.n_ep):
        for it, (input_a, input_b) in enumerate(train_loader):
            if input_a.size(0) != args.batch_size or input_b.size(0) != args.batch_size:
                continue

            # input
            input_a = input_a.cuda(args.gpu).detach()
            input_b = input_b.cuda(args.gpu).detach()

            # update model
            model.update_dis(input_a, input_b)
            model.update_enc_gen()

            #save to display file
            saver.write_display(total_iter, model)

            print("total_it: %d (ep %d, iter %d), lr %08f" % (total_iter, ep, it, model.genA_opt.param_groups[0]["lr"]))
            total_iter += 1

            if total_iter >= max_iter:
                saver.write_img(-1, model)
                saver.write_model(-1, total_iter, model)
                break

        if total_iter >= max_iter:
            break
        # save result image
        saver.write_img(ep, model)

        # save network weights
        saver.write_model(ep, total_iter, model)

if __name__ == "__main__":
    main()
