import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

args = None


class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(),
        )
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.reshape(-1, 128, 7, 7)
        return self.upconv(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.net(img)


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    print(" --- start training --- ")

    bce = nn.BCELoss()
    y_real = torch.ones(args.batch_size, 1)
    y_fake = torch.zeros(args.batch_size, 1)
    if torch.cuda.is_available():
        generator, discriminator = generator.cuda(), discriminator.cuda()
        bce = bce.cuda()
        y_real, y_fake = y_real.cuda(), y_fake.cuda()
    for epoch in range(args.n_epochs):
        total_G_loss = 0
        total_D_loss = 0
        for i, (imgs, _) in enumerate(dataloader):
            # prepare inputs
            noise = torch.rand((args.batch_size, args.latent_dim))
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                noise = noise.cuda()

            # train G
            optimizer_G.zero_grad()
            fake = generator(noise)
            #             print(imgs)
            #             print(fake)
            pred_fake = discriminator(fake)
            G_loss = bce(pred_fake, y_real)
            G_loss.backward()
            optimizer_G.step()

            total_G_loss += G_loss.item()

            if (epoch + 1) % 2 == 0:
                # train D with real images
                optimizer_D.zero_grad()
                pred_real = discriminator(imgs)
                D_real_loss = bce(pred_real, y_real)

                # train D with fake images
                fake = generator(noise)
                pred_fake = discriminator(fake)
                D_fake_loss = bce(pred_fake, y_fake)

                D_loss = D_real_loss + D_fake_loss
                D_loss.backward()
                optimizer_D.step()
                total_D_loss += D_loss.item()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                img = fake.view(args.batch_size, 1, 28, 28)
                save_image(img[:25],
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)

        total_D_loss /= len(dataloader)
        total_G_loss /= len(dataloader)
        print("Epoch: %d, D_loss: %.4f, G_loss: %.4f" % (epoch, total_D_loss, total_G_loss))


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataset = datasets.MNIST('./data/mnist', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,))]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Initialize models and optimizers
    generator = Generator(args.latent_dim)
    discriminator = Discriminator()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr * 10)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
