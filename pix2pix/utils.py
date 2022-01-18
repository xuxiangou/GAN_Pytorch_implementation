import torch
from torchvision.utils import save_image
import warnings

warnings.filterwarnings("ignore")


def to_cpu(tensor):
    return tensor.detach().cpu().numpy()


def generate_noise(shape):
    return torch.randn(shape)


@torch.no_grad()
def save_images(G, epochs, mask, image_seed, image_size):
    # _, height, width = mask.shape
    # for i in range(height // image_size):
    #     for j in range(width // image_size):
    #         mask[:, i * image_size:(i + 1) * image_size, j * image_size: (j + 1) * image_size] = G.forward(
    #             mask[:, i * image_size:(i + 1) * image_size, j * image_size: (j + 1) * image_size].unsqueeze(dim=0)
    #         )[0]
    mask = G.forward(mask.unsqueeze(dim=0))[0]
    save_image(mask * 0.5 + 0.5, f"./images/{epochs}_{image_seed}.png")


def save_weight(G, D, gen_dir, disc_dir):
    # for discriminator has spectral
    torch.save(G.state_dict(), gen_dir)
    torch.save(D.state_dict(), disc_dir)
    print("保存成功！")


def load_weight(G, D, gen_dir, disc_dir):
    G.load_state_dict(torch.load(gen_dir))
    D.load_state_dict(torch.load(disc_dir))
    return G, D
