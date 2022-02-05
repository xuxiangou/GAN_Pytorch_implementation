import math

import torch
from torchvision.utils import save_image
import numpy as np


def to_cpu(tensor):
    return tensor.detach().cpu().numpy()


def generate_noise(shape, device):
    return torch.randn(shape, device=device)


@torch.no_grad()
def save_images(G, epochs, latent_dim, device, batch_size):
    noise = generate_noise((batch_size, latent_dim), device).to(device)
    image = G.forward(noise)
    save_image(image * 0.5 + 0.5, f"./image/{epochs}.jpg", nrow=int(math.sqrt(batch_size)), normalize=False)


def save_weight(G, D, gen_dir, disc_dir):
    torch.save(G.state_dict(), gen_dir)
    torch.save(D.state_dict(), disc_dir)
    print("保存成功！")


def load_weight(G, D, gen_dir, disc_dir):
    G.load_state_dict(torch.load(gen_dir))
    D.load_state_dict(torch.load(disc_dir))
    return G, D
