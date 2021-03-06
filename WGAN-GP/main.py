import torch
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

from model import Discriminator, Generator
from utils import generate_noise, to_cpu, save_images, save_weight, load_weight
from Regularization import calculate_gradient


def creat_opt(known=False):
    parser = ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--num_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--data", type=str, default="../imageset",
                        help="the dataset of celebA")
    parser.add_argument("--dim", type=int, default=128, help="the dimension of G and D")
    parser.add_argument("--channels", type=int, default=3, help="the channels of image")
    parser.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help="the device to train")
    parser.add_argument("--penalty_lambda", type=int, default=10, help="the regular term of penalty")
    parser.add_argument("--n_critics", type=int, default=5)
    parser.add_argument("--weight_disc", type=str, default="./weight/Discriminator_SN_WGAN_GP.pt",
                        help="the weight of discriminator")
    parser.add_argument("--weight_gen", type=str, default="./weight/Generator_SN_WGAN_GP.pt",
                        help="the weight of generator")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(load_model=False):
    opt = creat_opt()

    # ????????????
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    TrainData = DataLoader(CIFAR10(opt.data, train=True, transform=transform, download=True),
                           batch_size=opt.batch_size,
                           shuffle=True,
                           pin_memory=True,
                           num_workers=opt.num_cpu,
                           drop_last=True)

    D = Discriminator(opt.latent_dim, opt.batch_size, opt.dim, opt.channels).to(opt.device)
    G = Generator(opt.latent_dim, opt.batch_size, opt.dim, opt.channels).to(opt.device)
    if load_model:
        G, D = load_weight(G, D, opt.weight_gen, opt.weight_disc)

    optim_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optim_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    schedule_D = torch.optim.lr_scheduler.StepLR(optim_D, gamma=0.92, step_size=8, verbose=True)
    schedule_G = torch.optim.lr_scheduler.StepLR(optim_G, gamma=0.92, step_size=8, verbose=True)

    for i in range(opt.n_epochs):
        print("")
        print(f"epoch: {i + 1}")
        loop = tqdm(TrainData)

        discriminator_loss = 0.
        generator_loss = 0.
        for batch_index, (real_data, _) in enumerate(loop):
            optim_D.zero_grad()

            # --------------------
            # train discriminator
            # --------------------
            real_data = real_data.to(opt.device)

            noise = generate_noise((opt.batch_size, opt.latent_dim), device=opt.device)
            fake_data = G.forward(noise)

            # fake image
            D_fake = D.forward(fake_data.detach())
            # real image
            D_real = D.forward(real_data)

            gradient_penalty = calculate_gradient(D, real_data, fake_data, opt.batch_size, opt.channels, opt.img_size,
                                                  opt.device, opt.penalty_lambda)

            D_loss = -torch.mean(D_real) + torch.mean(D_fake) + gradient_penalty
            D_loss.backward()
            optim_D.step()

            if batch_index % opt.n_critics == 0:
                # ----------------
                # train generator
                # ----------------
                optim_G.zero_grad()
                fake_image = G.forward(noise)

                # loss of generator
                G_fake = D(fake_image)
                G_loss = -torch.mean(G_fake)

                G_loss.backward()
                optim_G.step()
                generator_loss += G_loss.item()

            discriminator_loss += D_loss.item()
            loop.set_postfix(discriminator_loss=discriminator_loss / (batch_index + 1 + 1e-8),
                             generator_loss=generator_loss / ((batch_index + 1) // opt.n_critics + 1))

        # ??????????????????cpu???????????????discriminator???
        D.collect_loss(discriminator_loss * opt.batch_size / 50000)
        # ??????????????????cpu???????????????generator???
        G.collect_loss(generator_loss * opt.batch_size * opt.n_critics / 50000)
        save_images(G, i, opt.latent_dim, opt.device, opt.batch_size)

        # ???????????????
        schedule_D.step()
        schedule_G.step()

    D.plot_loss()
    G.plot_loss()
    save_weight(G, D, opt.weight_gen, opt.weight_disc)


if __name__ == '__main__':
    main()
