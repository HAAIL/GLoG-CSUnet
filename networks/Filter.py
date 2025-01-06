import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import math

#
class GaborConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding="same"):
        super(GaborConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize learnable parameters
        self.theta = nn.Parameter(torch.Tensor(out_channels).uniform_(0, math.pi))
        self.Lambda = nn.Parameter(torch.Tensor(out_channels).uniform_(self.kernel_size/5 , self.kernel_size))
        self.sigma = nn.Parameter(torch.Tensor(out_channels).uniform_(self.kernel_size / 6, self.kernel_size ))
        self.psi = nn.Parameter(torch.Tensor(out_channels).uniform_(-math.pi, math.pi))
        self.gamma = nn.Parameter(torch.Tensor(out_channels).uniform_(0.2, 1))


        print(
            f"sigma: {self.sigma}\n theta: {self.theta}\n Lambda: {self.Lambda}\n psi: {self.psi}\n gamma: {self.gamma}")

    def forward(self, x):
        # Generate Gabor kernels based on current parameters
        xmax = self.kernel_size // 2
        ymax = self.kernel_size // 2
        add = 0 if self.kernel_size % 2 == 0 else 1
        xmin = -xmax
        ymin = -ymax
        grid_x, grid_y = torch.meshgrid(torch.arange(ymin, ymax + add), torch.arange(xmin, xmax + add), indexing='ij')
        grid_x = grid_x.to(x.device)
        grid_y = grid_y.to(x.device)

        kernels = []
        for i in range(self.out_channels):
            x_theta = grid_x * torch.cos(self.theta[i]) + grid_y * torch.sin(self.theta[i])
            y_theta = -grid_x * torch.sin(self.theta[i]) + grid_y * torch.cos(self.theta[i])

            sigma_x = self.sigma[i]
            sigma_y = self.sigma[i] / self.gamma[i]

            gb = torch.exp(-0.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * torch.cos(
                2 * np.pi / self.Lambda[i] * x_theta  + self.psi[i])
            epsilon = 1e-10
            gb = gb / ((torch.sum(gb ** 2)) + epsilon)  # L2 norm

            kernels.append(gb)

        kernels = torch.stack(kernels).unsqueeze(1)  # Shape: [out_channels, in_channels, H, W]

        return F.conv2d(x, kernels, stride=self.stride, padding=self.padding)


class LoGConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding="same"):
        super(LoGConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # self.pi = torch.tensor(torch.pi)
        self.sigma_max = 10
        self.sigma_min = 1e-3
        # Initialize learnable parametery
        self.sigma = nn.Parameter(torch.Tensor(out_channels).uniform_(0.5, 1.5))

        print(f"sigma: {self.sigma}")

        self.sigma_grads = []  # List to store gradient history

    def track_sigma_grad(self):
        if self.sigma.grad is not None:
            grad_norm = torch.norm(self.sigma.grad.data).item()
            grad_mean = self.sigma.grad.data.mean().item()
            grad_std = self.sigma.grad.data.std().item()
            value_mean = self.sigma.data.mean().item()
            self.sigma_grads.append({
                'value': value_mean,
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std,
                'min': self.sigma.grad.data.min().item(),
                'max': self.sigma.grad.data.max().item()
            })
            return grad_norm, grad_mean, grad_std, value_mean
        return None, None, None

    def get_sigma_stats(self):
        return {
            'value': self.sigma.data.cpu().numpy(),
            'grad_history': self.sigma_grads
        }



    def clip_sigma_grad(self):
        self.sigma.grad = torch.clamp(self.sigma.grad, -1, 1)



    def forward(self, x):
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-self.kernel_size // 2, self.kernel_size // 2, self.kernel_size),
            torch.linspace(-self.kernel_size // 2, self.kernel_size // 2, self.kernel_size),
            indexing="ij"
        )
        grid_x = grid_x.to(x.device)
        grid_y = grid_y.to(x.device)
        pi=torch.pi
        kernels = []
        self.clapm_sigma=torch.clamp(self.sigma, self.sigma_min, self.sigma_max)
        for i in range(self.out_channels):
            # pi = torch.tensor(torch.pi, device=device)
            # Compute squared distances
            dist_sq = grid_x ** 2 + grid_y ** 2

            # Gaussian component
            gaussian = torch.exp(-dist_sq / (2 * self.clapm_sigma[i] ** 2))

            # Laplacian component
            epsilon = 1e-10
            laplacian = (1 - (dist_sq / (2 * self.clapm_sigma[i] ** 2))) / (-pi * (self.clapm_sigma[i] ** 4+ epsilon))

            # Laplacian of Gaussian filter
            log_filter = laplacian * gaussian
            # normalize filter
            log_filter = log_filter / ((torch.sum(log_filter ** 2)) + epsilon)  # L2 norm

            kernels.append(log_filter)

        kernels = torch.stack(kernels).unsqueeze(1)  # Shape: [out_channels, in_channels, H, W]

        return F.conv2d(x, kernels, stride=self.stride, padding=self.padding)




# Example usage
if __name__ == "__main__":
    gabor_layer = GaborConv2D(in_channels=5, out_channels=90, kernel_size=3)
    input_tensor = torch.randn(1, 1, 28, 28).cuda()  # Example input tensor
    output_tensor = gabor_layer(input_tensor)
    print(output_tensor.shape)