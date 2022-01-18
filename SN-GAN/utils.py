import math
import torch
from torchvision.utils import save_image


def to_cpu(tensor):
    return tensor.detach().cpu().numpy()


def generate_noise(shape):
    return torch.randn(shape)


@torch.no_grad()
def save_images(G, epochs, latent_dim, device, batch_size):
    noise = generate_noise((batch_size, latent_dim)).to(device)
    image = G.forward(noise)
    # image = to_cpu(image.permute(0, 2, 3, 1).contiguous())
    save_image(image, f"./image/{epochs}.png", nrow=int(math.sqrt(batch_size)), normalize=False)


def save_weight(G, D, gen_dir, disc_dir):
    # for discriminator has spectral
    torch.save(G.state_dict(), gen_dir)
    torch.save(D.state_dict(), disc_dir)
    print("保存成功！")


def load_weight(G, D, gen_dir, disc_dir):
    G.load_state_dict(torch.load(gen_dir))
    D.load_state_dict(torch.load(disc_dir))
    return G, D
