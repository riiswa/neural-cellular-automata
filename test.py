import torch

from main import NeuralCellularAutomata
import numpy as np
import imageio


def create_gif(images, filename):
    imageio.mimsave(filename, images, fps=60)


if __name__ == "__main__":
    nca = NeuralCellularAutomata()
    nca.load()

    seed = torch.zeros(64, 64, 16)
    seed[64 // 2, 64 // 2, 3:] = 1
    seed[64 // 2, 64 // 2, 3:] = 1


    image_tensors = []

    def f(x):
        img = (x.squeeze(0)[:4, :, :].transpose(0, 2).clamp(0, 1).detach().numpy() * 255)
        image_tensors.append(img.astype(np.uint8))

    nca.update(seed.unsqueeze(0), 1000, 1000, hook=f)

    create_gif(image_tensors, "result.gif")