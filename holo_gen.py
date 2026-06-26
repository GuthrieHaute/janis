import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import os
from tqdm import tqdm
import argparse

PHI = (1 + math.sqrt(5)) / 2

class MinimalHoloGen(nn.Module):
    """Minimal Holographic Generator - Formula 1 as measurement only"""
    def __init__(self, field_size=512, img_size=32, channels=3, num_classes=10):
        super().__init__()
        self.field_size = field_size
        self.img_size = img_size
        self.channels = channels

        # Reference beam (condition encoding)
        self.condition_embed = nn.Embedding(num_classes, field_size)

        # Learned holographic geometry
        self.G = nn.Parameter(torch.randn(field_size, field_size) * 0.02)

        # Phi binding - temporal coherence (reference frequency)
        self.phi_coupling = nn.Parameter(torch.tensor(0.18))
        self.phi_speed = nn.Parameter(torch.ones(field_size) * (2 * math.pi / PHI))

        # Memory (persistent field state)
        self.memory_rate = nn.Parameter(torch.ones(field_size) * 0.15)

        # Readout - measures the interference pattern
        self.readout = nn.Linear(field_size, img_size * img_size * channels)

        # Fibonacci boundary (golden angle container)
        indices = torch.arange(field_size)
        golden_angle = 2 * math.pi / (PHI * PHI)
        self.register_buffer('boundary_shape', 1.0 + 0.5 * torch.cos(indices * golden_angle))

    def forward(self, condition_ids, resonance_steps=3):
        batch = condition_ids.shape[0]
        device = condition_ids.device

        # Initial excitation from condition (reference beam)
        state = self.condition_embed(condition_ids)
        memory = state.clone()
        phi_t = torch.zeros(batch, self.field_size, device=device)

        # Accumulate states for mean pooling
        all_states = []

        for _ in range(resonance_steps):
            prev_state = state.clone()
            prev_memory = memory.clone()

            # Phi binding
            phi_wave = torch.sin(phi_t)
            state = state * (1.0 + self.phi_coupling * phi_wave)
            phi_t = phi_t + self.phi_speed

            # Memory update
            rate = torch.sigmoid(self.memory_rate)
            memory = (1 - rate) * memory + rate * state

            # === FORMULA 1 AS MEASUREMENT ===
            grad_memory = memory - prev_memory
            grad_state = state - prev_state
            E = torch.matmul(grad_memory, self.G)                    # interference
            coherence = torch.norm(E, dim=-1, keepdim=True) / math.sqrt(self.field_size)

            # Gentle feedback from measurement
            state = state + E * coherence * 0.08   # low strength - measurement, not primary driver

            # Fibonacci boundary containment
            state = F.normalize(state, dim=-1) * math.sqrt(self.field_size)
            state = state * self.boundary_shape
            
            all_states.append(state)

        # Mean pool - the "standing wave" is the average of the resonance
        # This prevents the model from "forgetting" early state information
        state = torch.stack(all_states, dim=1).mean(dim=1)

        # Final readout - measure the standing wave
        img_flat = self.readout(state)
        img = img_flat.view(batch, self.channels, self.img_size, self.img_size)
        return torch.sigmoid(img)


def get_dataloader(dataset_name, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    # Standard data root
    root = "data"

    if dataset_name.lower() == 'mnist':
        train_set = torchvision.datasets.MNIST(root=root, train=True, download=False, transform=transform)
        test_set = torchvision.datasets.MNIST(root=root, train=False, download=False, transform=transform)
        num_classes = 10
        img_size = 32

    elif dataset_name.lower() == 'fashion':
        train_set = torchvision.datasets.FashionMNIST(root=root, train=True, download=False, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(root=root, train=False, download=False, transform=transform)
        num_classes = 10
        img_size = 32

    elif dataset_name.lower() == 'cifar':
        train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=transform)
        num_classes = 10
        img_size = 32
    else:
        raise ValueError("Unsupported dataset")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader, num_classes, img_size


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar', choices=['mnist', 'fashion', 'cifar'])
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--field_size', type=int, default=512)
    parser.add_argument('--steps', type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, num_classes, img_size = get_dataloader(args.dataset, args.batch_size)

    model = MinimalHoloGen(
        field_size=args.field_size,
        img_size=img_size,
        channels=3 if args.dataset == 'cifar' else 1,
        num_classes=num_classes
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs("samples", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            generated = model(labels, resonance_steps=args.steps)

            loss = F.mse_loss(generated, images)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{total_loss / (pbar.n + 1):.6f}")

        scheduler.step()

        # Save sample grid
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                sample_labels = torch.arange(num_classes).to(device)
                samples = model(sample_labels, resonance_steps=args.steps)
                grid = torchvision.utils.make_grid(samples, nrow=int(math.sqrt(num_classes)), normalize=True)
                plt.imsave(f"samples/epoch_{epoch+1:02d}.png", grid.permute(1, 2, 0).cpu().numpy())
            print(f"→ Sample grid saved (epoch {epoch+1})")

        print(f"Epoch {epoch+1} completed. Avg Loss: {total_loss / len(train_loader):.6f}")

    torch.save(model.state_dict(), f"checkpoints/hologram_{args.dataset}_final.pth")
    print("Training finished!")


if __name__ == "__main__":
    train()