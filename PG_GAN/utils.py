import os

import torch
from torchvision.utils import save_image


def to_cpu(tensor):
    return tensor.detach().cpu().numpy()


# 生成一个[-1,1)之间的随机数
def generate_noise(shape, device):
    return torch.rand(shape, device=device, requires_grad=True) * 2 - 1


@torch.no_grad()
def save_images(G, epochs, latent_dim, device, batch_size):
    noise = generate_noise((batch_size, latent_dim), device).to(device)
    image = G.forward(noise)
    for i in range(5):
        save_image(image[i] * 0.5 + 0.5, f"../image/{epochs}_{i}.png", normalize=False)


def save_weight(G, D, gen_dir, disc_dir):
    torch.save(G.state_dict(), gen_dir)
    torch.save(D.state_dict(), disc_dir)
    print("保存成功！")


def load_weight(G, D, gen_dir, disc_dir):
    G.load_state_dict(torch.load(gen_dir))
    D.load_state_dict(torch.load(disc_dir))
    return G, D
