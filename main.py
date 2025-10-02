import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torchvision.transforms import ToTensor
import vae_model
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.manifold import TSNE

BATCH_SIZE = 32
LR_RATE = 1e-3
MNIST_DATA_SET_PATH = "../MNISTDataset/"
EPOCHS = 10

IMG_DIM = 28*28
HIDDEN_DIM = 400
Z_DIM = 40

mnist_train_ds = torchvision.datasets.MNIST(
        root=MNIST_DATA_SET_PATH,
        train=True,
        download=True,
        transform=ToTensor()
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_loop(dataloader: DataLoader, model: vae_model.VaeModel, optimizer: torch.optim.Adam, device, epoch):
    size = len(dataloader.dataset)
    model.train()

    for batch, (img, _) in enumerate(dataloader):
        img = img.to(device).view(img.shape[0], IMG_DIM)
        mu, logvar, recon = model(img)

        # loss compute
        recon_loss = nn.MSELoss(reduction="sum")(recon, img) / img.size(0)

        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / img.size(0)
        total_loss = recon_loss + kl_divergence

        total_loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = total_loss.item(), batch * BATCH_SIZE + len(img)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def train_model():
    model = vae_model.VaeModel(IMG_DIM, HIDDEN_DIM, Z_DIM)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
    data_loader = DataLoader(mnist_train_ds, BATCH_SIZE)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(data_loader, model, optimizer, device, epoch)

    torch.save(model.state_dict(), 'model_weights.pth')
    print("Done!")

def visualize_latent_tsne(model, device, dataset, n_samples=3000):
    model.eval()
    z_list, labels = [], []
    
    with torch.no_grad():
        for i in range(n_samples):
            img, label = dataset[i]
            mu, _, _ = model(img.to(device).view(1, IMG_DIM))
            z_list.append(mu.cpu().numpy().flatten())
            labels.append(label)
    
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(np.array(z_list))
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, ticks=range(10))
    plt.title(f'VAE Latent Space (t-SNE from {Z_DIM}D)')
    plt.show()


def generate_random_number(model, device):
    model.eval()  # set to eval mode

    # Ensure image is 4D: (1, 1, 28, 28)
    z = torch.randn(1, Z_DIM).to(device)

    with torch.no_grad():
        generated = model.decode(z)  # flatten to (1, 784)
        generated = generated.view(-1, 1, 28, 28)  # reshape back to image

    # Convert to CPU numpy for plotting
    generated_img = generated.view(28, 28).cpu().numpy()

    # Plot side by side
    fig, axes = plt.subplots(1, 1, figsize=(3, 3))

    axes.imshow(generated_img, cmap="gray")
    axes.set_title("Generated")
    axes.axis("off")

    plt.show()


def show_reconstruction(model, device, img):
    model.eval()  # set to eval mode

    # Ensure image is 4D: (1, 1, 28, 28)
    img = img.unsqueeze(0) if img.dim() == 3 else img
    img = img.to(device)

    with torch.no_grad():
        mu, logvar, recon = model(img.view(img.size(0), -1))  # flatten to (1, 784)
        recon = recon.view(-1, 1, 28, 28)  # reshape back to image

    # Convert to CPU numpy for plotting
    orig = img.view(28, 28).cpu().numpy()
    recon_img = recon.view(28, 28).cpu().numpy()

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(orig, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(recon_img, cmap="gray")
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    plt.show()


def main():
    model = vae_model.VaeModel(IMG_DIM, HIDDEN_DIM, Z_DIM).to(device)
    
    while True:
        print("\nChoose an action:")
        print("1: Train and load model")
        print("2: Only load model")
        print("3: Perform reconstruction on a random image")
        print("4: Generate a random number (image)")
        print("5: Visualize latent space (t-SNE)")
        print("6: Exit")

        choice = input("Enter choice (1/2/3/4/5/6): ").strip()

        if choice == "1":
            print("Training model...")
            train_model()
            model.load_state_dict(torch.load('model_weights.pth', map_location=device))
            print("Model loaded successfully.")
        elif choice == "2":
            file = input("Enter model weight file: ") or "model_weights.pth"
            model.load_state_dict(torch.load(file, map_location=device))
            print("Model loaded successfully.")
        elif choice == "3":
            idx = random.randint(0, len(mnist_train_ds) - 1)
            sample_img, _ = mnist_train_ds[idx]
            show_reconstruction(model, device, sample_img)
        elif choice == "4":
            generate_random_number(model, device)
        elif choice == "5":
            visualize_latent_tsne(model, device, mnist_train_ds, n_samples=3000)
        elif choice == "6":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()