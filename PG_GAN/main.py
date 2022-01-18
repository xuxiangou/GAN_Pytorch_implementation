from argparse import ArgumentParser
import torch
from dataset import PGDataset
from torch.utils.data import DataLoader
from model import Discriminator, Generator
from torch.optim import AdamW
from tqdm import tqdm
from Regularization import calculate_gradient
from utils import to_cpu, load_weight, save_weight, save_images, generate_noise


def creat_opt(known=False):
    parser = ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=list, default=[128, 128, 64, 32, 16, 8, 4], help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--num_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=512, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=list, default=[4, 8, 16, 32, 64, 128, 256],
                        help="size of each image dimension")
    parser.add_argument("--data", type=str, default="../imageset/face/face",
                        help="the dataset of celebA")
    parser.add_argument("--channels", type=int, default=3, help="the channels of image")
    parser.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help="the device to train")
    parser.add_argument("--LAMBDA", type=int, default=10, help="the regular term of penalty")
    parser.add_argument("--n_critics", type=int, default=1)
    parser.add_argument("--weight_disc", type=str, default="./weight/Discriminator_SN_WGAN_GP.pt",
                        help="the weight of discriminator")
    parser.add_argument("--weight_gen", type=str, default="./weight/Generator_SN_WGAN_GP.pt",
                        help="the weight of generator")
    parser.add_argument("--weight_penalty", type=int, default=0.001)
    parser.add_argument("--scale_iteration", type=list,
                        default=[20, 50, 70, 100, 150, 200, 250], help="each image scale iteration times")

    # it is very important for convergence of model
    parser.add_argument("--update_alpha", type=list, default=
    [
        {},
        {},
        {0: 1, 800: 0.9, 1500: 0.75, 2500: 0.5, 3000: 0.25, 3500: 0},
        {0: 1, 1500: 0.9, 3000: 0.75, 5000: 0.5, 6000: 0.25, 7000: 0},
        {0: 1, 2000: 0.9, 5000: 0.75, 8000: 0.5, 10000: 0.25, 16000: 0},
        {0: 1, 8000: 0.9, 16000: 0.75, 32000: 0.5, 48000: 0.25, 54000: 0},
        {0: 1, 10000: 0.9, 20000: 0.75, 40000: 0.5, 60000: 0.25, 80000: 0},
    ]
                        )
    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt


def main(load_model=False):
    opt = creat_opt()

    D = Discriminator(opt.latent_dim, opt.channels, opt.device).to(opt.device)
    G = Generator(opt.latent_dim, opt.channels, opt.img_size[-1], opt.device).to(opt.device)
    if load_model:
        G, D = load_weight(G, D, opt.weight_gen, opt.weight_disc)

    optim_D = AdamW(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optim_G = AdamW(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    for i in range(len(opt.scale_iteration)):
        print(f"\nnow start the size: {opt.img_size[i]}\n")
        TrainData = DataLoader(PGDataset(opt.data, opt.img_size[i]),
                               batch_size=opt.batch_size[i],
                               shuffle=True,
                               pin_memory=True,
                               num_workers=opt.num_cpu,
                               drop_last=True)
        # used to record the iter times
        iteration = 0

        for j in range(opt.scale_iteration[i]):
            print(f"\nimage_size:{opt.img_size[i]}, epoch: {j + 1}\n")
            loop = tqdm(TrainData)

            D_loss = 0.
            G_loss = 0.
            discriminator_loss = 0.
            generator_loss = 0.

            for batch_index, (real_data) in enumerate(loop):
                # -----------------
                # update the alpha
                # -----------------

                if iteration in opt.update_alpha[i]:
                    new_alpha = opt.update_alpha[i][iteration]
                    D.update_alpha(new_alpha)
                    G.update_alpha(new_alpha)

                # --------------------
                # train discriminator
                # --------------------

                optim_D.zero_grad()
                real_data = real_data.to(opt.device)

                noise = generate_noise((opt.batch_size[i], opt.latent_dim), opt.device)
                fake_data = G.forward(noise)

                # fake image
                D_fake = D.forward(G.forward(noise).detach())
                # real image
                D_real = D.forward(real_data)

                # if use hinge loss here
                gradient_penalty = calculate_gradient(D, real_data, fake_data, opt.batch_size[i], opt.channels,
                                                      opt.img_size[i], opt.device, opt.LAMBDA)

                # wasserstein
                D_loss = -torch.mean(D_real) + torch.mean(D_fake) + gradient_penalty + \
                         opt.weight_penalty * ((D_real ** 2).mean())

                D_loss.backward(retain_graph=True)
                optim_D.step()

                optim_G.zero_grad()
                D_loss = to_cpu(D_loss)

                if batch_index % opt.n_critics == 0:
                    # ----------------
                    # train generator
                    # ----------------
                    fake_image = G.forward(noise)
                    # loss of generator
                    G_fake = D(fake_image)

                    # wasserstein loss
                    G_loss = -torch.mean(G_fake)

                    G_loss.backward()
                    optim_G.step()

                    # 将损失放置到cpu上，保存到generator中
                    G_loss = to_cpu(G_loss)

                discriminator_loss += D_loss
                generator_loss += G_loss
                loop.set_postfix(discriminator_loss=discriminator_loss / (batch_index + 1 + 1e-8),
                                 generator_loss=generator_loss / (batch_index + 1 + 1e-8))

                iteration += 1

        # ---------------------
        # add layer to D and G
        # ---------------------

        save_images(G, i, opt.latent_dim, opt.device, opt.batch_size[i])
        in_channels, out_channels = G.add_layer()
        # the in_channels for D is out_channels
        D.add_layer(out_channels, in_channels)

        # ---------------------
        # build optimizer again
        # ---------------------

        optim_D = AdamW(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optim_G = AdamW(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    save_weight(G, D, opt.weight_gen, opt.weight_disc)


if __name__ == '__main__':
    main()
