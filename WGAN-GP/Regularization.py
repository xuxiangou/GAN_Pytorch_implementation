import torch
from torch.autograd import grad


def calculate_gradient(D, real_data, fake_data, batch_size, channels, img_size, device, LAMBDA):
    # Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, fake_data.nelement() // batch_size).contiguous().view(batch_size, channels,
                                                                                           img_size,
                                                                                           img_size).to(device)

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.to(device)

    disc_interpolates = D(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones_like(disc_interpolates, requires_grad=False),
                     retain_graph=True, create_graph=True)[0]
    gradients = gradients.view(gradients.shape[0], -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return penalty
