import torch
import torch.nn as nn
import requests
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os


def load_image_from_url(url, size=64):
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    img = img.resize((size, size))
    img = img.convert("RGBA")
    img_tensor = torch.from_numpy(np.array(img))
    return img_tensor / 255


def plot_images(images, writer, step, filename, rows=None, cols=None, figsize=None):  # Written by ChatGPT
    """
    Plot a batch of images in a grid.

    Parameters:
    - images: numpy array of shape (batch_size, height, width, channels)
    - rows: number of rows in the grid, defaults to sqrt(batch_size)
    - cols: number of columns in the grid, defaults to ceil(batch_size / rows)
    - figsize: size of the figure, default to None

    Returns: None
    """
    batch_size, height, width, channels = images.shape

    if rows is None:
        rows = int(np.ceil(np.sqrt(batch_size)))
    if cols is None:
        cols = int(np.ceil(batch_size / rows))
    if figsize is None:
        figsize = (cols * height / 10, rows * width / 10)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if i < batch_size:
            ax.imshow(images[i])
        ax.axis("off")
    plt.tight_layout()
    writer.add_figure("Batch images", fig, step)
    plt.close()


def generate_filename(num, directory="imgs", prefix="img_", padding=5):  # Written by ChatGPT
    """
    Generate a filename for an image with a number and padding.

    Parameters:
    - directory: directory where the image will be stored
    - prefix: prefix for the filename
    - num: number for the filename
    - padding: number of zeros to pad with

    Returns:
    - Filename in the form of directory/prefix + number with padding + .jpg
    """
    filename = "{}/{}{:0>{}}.jpg".format(directory, prefix, num, padding)
    return filename


def swap_positions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list


def alive_masking(state_grid):
    alive = torch.nn.functional.max_pool2d(state_grid[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1
    state_grid = state_grid * alive.float()
    return state_grid


def damage(image, radius):
    height, width = image.shape[:2]
    x_center = random.randint(0, width)
    y_center = random.randint(0, height)
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    mask = (x - x_center)**2 + (y - y_center)**2 <= radius**2
    image[mask, ..., :3] = 1


class NeuralCellularAutomata(nn.Module):
    def __init__(self):
        super(NeuralCellularAutomata, self).__init__()
        self.device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter()

        if not os.path.exists("imgs"):
            os.makedirs("imgs")

        sobel_x = torch.tensor([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]], dtype=torch.float) / 8

        self.conv_sobel_x = nn.Conv2d(16, 16, 3, padding='same')
        self.conv_sobel_x.weight = nn.Parameter(
            torch.tile(sobel_x, (16, 16, 1, 1)),
            requires_grad=False
        )

        self.conv_sobel_y = nn.Conv2d(16, 16, 3, padding='same')
        self.conv_sobel_x.weight = nn.Parameter(
            torch.tile(sobel_x.T, (16, 16, 1, 1)),
            requires_grad=False
        )
        self.dense1 = nn.Linear(48, 128)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(128, 16)
        self.dense2.weight.data.zero_()

        self.loss_fn = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5)
        self.to(self.device)

    def save(self):
        torch.save(self.state_dict(), 'model_weights.pth')

    def load(self):
        self.model.load_state_dict(torch.load('model_weights.pth'))

    def stochastic_update(self, state_grid, ds_grid):
        rand_mask = (torch.rand((ds_grid.size(0), ds_grid.size(1), ds_grid.size(2), 1)) < 0.5).to(self.device).float()
        ds_grid = ds_grid * rand_mask
        return state_grid + ds_grid.transpose(1, 3)

    def forward(self, x):
        x = torch.cat((self.conv_sobel_x(x), self.conv_sobel_y(x), x), 1)
        x = x.transpose(1, 3)
        x = self.relu(self.dense1(x))
        x = self.dense2(x)
        return x

    def update(self, x, min_epochs=64, max_epochs=96):
        x = x.transpose(1, 3)
        for _ in range(random.randint(min_epochs, max_epochs)):
            ds = self(alive_masking(x))
            x = self.stochastic_update(x, ds)
        return x.transpose(1, 3)

    def pool_training(self, target, epochs=50000, pool_size=1024, batch_size=32, monitoring_interval=50):
        seed = torch.zeros(target.shape[0], target.shape[1], 16)
        seed[target.shape[0] // 2, target.shape[1] // 2, 3:] = 1
        pool = [seed.clone() for _ in range(pool_size)]
        targets = target.repeat(batch_size, 1, 1, 1).to(self.device)

        def sample():
            return [list(x) for x in zip(*random.sample(list(enumerate(pool)), batch_size))]

        for i in tqdm(range(epochs)):
            idxs, batch = sample()
            with torch.no_grad():
                sorted_idx = sorted(range(batch_size),
                                    key=lambda x: self.loss_fn(
                                        batch[x][:, :, :4].transpose(0, 2).unsqueeze(0).to(self.device),
                                        target.transpose(0, 2).unsqueeze(0).clamp(0, 1).to(self.device)),
                                    reverse=True
                                    )
            idxs = [idxs[i] for i in sorted_idx]
            batch = [batch[i] for i in sorted_idx]
            batch[0] = seed.clone()

            for i in range(batch_size - int(round(batch_size * 0.2)), batch_size):
                damage(batch[i], random.randint(batch[i].shape[0] // 12, batch[i].shape[0] // 5))

            self.optimizer.zero_grad()
            state_grids = torch.stack(batch)
            outputs = self.update(state_grids.to(self.device))
            del state_grids
            loss = self.loss_fn(outputs[:, :, :, :4].transpose(1, 3), targets.transpose(1, 3))
            loss.backward()
            self.writer.add_scalar("Loss", loss.item(), i)
            self.optimizer.step()
            self.scheduler.step()
            outputs = outputs.cpu().detach()
            if i % monitoring_interval == 0:
                plot_images(torch.clamp(outputs[:, :, :, :4], min=0, max=1), self.writer, i, generate_filename(i))
            for idx, output in zip(idxs, outputs):
                pool[idx] = output.cpu().detach()
                del output


if __name__ == "__main__":
    nca = NeuralCellularAutomata()
    image_url = "https://static.vecteezy.com/system/resources/previews/003/240/508/original/beautiful-purple-daisy-flower-isolated-on-white-background-vector.jpg"
    nca.pool_training(load_image_from_url(image_url))
    nca.save()
