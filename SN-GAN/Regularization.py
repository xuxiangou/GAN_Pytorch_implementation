import torch
import torch.nn as nn


# nodiff 更好理解
class SpectralNorm(nn.Module):
    def __init__(self, module, name: str = "weight", power_iterations: int = 1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_parameters():
            self._make_parameters()

    @staticmethod
    def l2normalize(v, eps=1e-12):
        return v / (v.norm() + eps)

    def _update_u_v(self):
        w = getattr(self.module, self.name)
        u = getattr(self.module, self.name + '_u')

        height = w.data.shape[0]

        """
        这是因为神经网络的每次迭代中参数 W 是在缓慢更新的，因此每两次迭代过程中可以认为 W的奇异值是近似相等的。
        因此虽然在网络的每一次迭代中，只进行一次power iteration，但是随着网络的训练，power iteration 对奇异值的估计会越来越准。
        """
        for _ in range(self.power_iterations):  # 一般而言power_iterations设置为1
            v = self.l2normalize(torch.mv(torch.t(w.view(height, -1).data), u))  # view不改变原始w，没有赋值
            u = self.l2normalize(torch.mv(w.view(height, -1).data, v.data))

        setattr(self.module, self.name + "_u", u)
        # dot代表两个向量相乘 -> 标量
        w.data = w.data / u.dot(torch.mv(w.view(height, -1), v))

    def _made_parameters(self):
        try:
            u = getattr(self.module, self.name + '_u')
            v = getattr(self.module, self.name + '_v')
            w = getattr(self.module, self.name + '_bar')
            return True
        except AttributeError:
            return False

    def _make_parameters(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        # 不需要计算u, v梯度，随机初始化
        u = self.l2normalize(w.data.new(height).normal_(0, 1))

        self.module.register_buffer(self.name + '_u', u)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


# WGAN-GP regularization
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
