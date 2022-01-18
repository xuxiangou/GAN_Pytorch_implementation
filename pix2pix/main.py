import random
import torch
import torch.nn as nn
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import to_cpu, generate_noise, load_weight, save_weight, save_images
from model import Generator, Discriminator
from dataset import CityscapesDataset
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")


def creat_opt(known=False):
    parser = ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--num_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--train_data", type=str, default="../imageset/edges2shoes/train")
    parser.add_argument("--valid_data", type=str, default="../imageset/edges2shoes/val")
    parser.add_argument("--dim", type=int, default=64, help="the dimension of G and D")
    parser.add_argument("--channels", type=int, default=3, help="the channels of image")
    parser.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help="the device to train")
    parser.add_argument("--n_critics", type=int, default=1, help="the discriminator train per generator train")
    parser.add_argument("--patch_size", type=int, default=30, help="the patch size")
    parser.add_argument("--Lambda_pixel", type=int, default=100)
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(load_model=False):
    opt = creat_opt()

    both_transform = A.Compose(
        # [A.Resize(width=256, height=256), ],
        [],
        additional_targets={"image0": "image"},
    )

    transform_only_input = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.2),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
            ToTensorV2(),
        ]
    )

    transform_only_mask = A.Compose(
        [
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
            ToTensorV2(),
        ]
    )

    transform_only_valid = A.Compose(
        [
            # A.Resize(width=512, height=512),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
            ToTensorV2(),
        ],
        additional_targets={"image0": "image"}
    )

    TrainData = DataLoader(
        CityscapesDataset(opt.train_data, opt.train_data, opt.img_size, True,
                          (both_transform, transform_only_input, transform_only_mask)),
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=opt.num_cpu,
        drop_last=True
    )

    Valid_data = CityscapesDataset(opt.valid_data, opt.valid_data, opt.img_size, False,
                                   transform_only_valid)

    D = Discriminator(opt.channels, opt.batch_size, opt.dim).to(opt.device)
    G = Generator(opt.channels, opt.dim).to(opt.device)
    # G = UnetGenerator(3, 3, 8, 64, norm_layer=nn.InstanceNorm2d, use_dropout=True).to(opt.device)

    if load_model:
        G, D = load_weight(G, D, opt.weight_gen, opt.weight_disc)

    optim_D = torch.optim.AdamW(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optim_G = torch.optim.AdamW(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    criterion_GAN = nn.BCEWithLogitsLoss()
    pixel = nn.L1Loss()

    for i in range(opt.n_epochs):
        print(f"\nepoch: {i + 1}\n")
        loop = tqdm(TrainData)

        D_loss = 0.
        G_loss = 0.
        discriminator_loss = 0.
        generator_loss = 0.
        for batch_index, (image, mask) in enumerate(loop):

            # --------------------
            # train discriminator
            # --------------------

            valid = torch.ones((image.shape[0], 1, opt.patch_size, opt.patch_size)).to(opt.device)
            fake = torch.zeros((image.shape[0], 1, opt.patch_size, opt.patch_size)).to(opt.device)

            image, mask = image.to(opt.device), mask.to(opt.device)

            # fake image
            D_fake = D.forward(G.forward(mask).detach(), mask)
            # real image
            D_real = D.forward(image, mask)

            # Use BCEWithLogitsLoss
            D_loss = 0.5 * (criterion_GAN(D_real, valid) + criterion_GAN(D_fake, fake))

            optim_D.zero_grad()
            D_loss.backward()
            optim_D.step()

            D_loss = to_cpu(D_loss)

            if batch_index % opt.n_critics == 0:
                # ----------------
                # train generator
                # ----------------

                # loss of generator
                fake_image = G.forward(mask)
                pred_fake = D.forward(fake_image, mask)
                G_loss = criterion_GAN(pred_fake, valid) + opt.Lambda_pixel * pixel(fake_image, image)

                optim_G.zero_grad()
                G_loss.backward()
                optim_G.step()

                # 将损失放置到cpu上，保存到generator中
                G_loss = to_cpu(G_loss)

            discriminator_loss += D_loss
            generator_loss += G_loss
            loop.set_postfix(discriminator_loss=discriminator_loss / (batch_index + 1 + 1e-8),
                             generator_loss=generator_loss / (batch_index + 1 + 1e-8))

        # 将损失放置到cpu上，保存到discriminator中
        D.collect_loss(D_loss)
        # 将损失放置到cpu上，保存到generator中
        G.collect_loss(G_loss)

        with torch.no_grad():
            random_value = [random.randint(0, len(Valid_data) - 1) for _ in range(1)]
            for image_seed in random_value:
                _, mask = Valid_data[image_seed]
                mask = mask.to(opt.device)
                save_images(G, i, mask, image_seed, opt.img_size)

    D.plot_loss()
    G.plot_loss()


if __name__ == '__main__':
    main()
