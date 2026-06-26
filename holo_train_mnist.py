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
    def __init__(self, field_size=512, img_size=32, channels=1, num_classes=10):
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

        # Accuracy head - projecting state to classes for metric printing
        self.classifier = nn.Linear(field_size, num_classes)

        # Fibonacci boundary (golden angle container)
        indices = torch.arange(field_size)
        golden_angle = 2 * math.pi / (PHI * PHI)
        self.register_buffer('boundary_shape', 1.0 + 0.5 * torch.cos(indices * golden_angle))

        # Persistent Standing Wave Pool (one per class)
        # This stores the "long-lived" state of the field
        self.register_buffer('standing_wave', torch.zeros(num_classes, field_size))
        # Keep track of exposure counts for true averaging (optional, EMA is often more stable)
        self.register_buffer('pool_rate', torch.tensor(0.01)) 

    def forward(self, condition_ids, resonance_steps=12):
        batch = condition_ids.shape[0]
        device = condition_ids.device

        # Initial excitation from condition (reference beam)
        state = self.condition_embed(condition_ids)
        memory = state.clone()
        phi_t = torch.zeros(batch, self.field_size, device=device)

        # Accumulate states for mean pooling within this resonance event
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

        # Mean pool - the "transient standing wave" of this batch
        transient_wave = torch.stack(all_states, dim=1).mean(dim=1)

        # Update the Persistent standing wave (EMA update)
        if self.training:
            with torch.no_grad():
                # For each class in the batch, blend the transient wave into the long-lived pool
                # This is the "pooling between epochs"
                for i in range(batch):
                    cid = condition_ids[i]
                    self.standing_wave[cid] = (1 - self.pool_rate) * self.standing_wave[cid] + self.pool_rate * transient_wave[i]

        # Use the Persistent standing wave for the final readout if available, 
        # otherwise use the transient wave (blended)
        # We blend them so gradients still flow back to the parameters
        final_state = 0.5 * transient_wave + 0.5 * self.standing_wave[condition_ids]

        # Image Readout
        img_flat = self.readout(final_state)
        img = img_flat.view(batch, self.channels, self.img_size, self.img_size)
        
        # Class Readout for accuracy tracking
        logits = self.classifier(final_state)
        
        return torch.sigmoid(img), logits


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
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--field_size', type=int, default=512)
    parser.add_argument('--steps', type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"HOLOGRAPHIC MNIST TRAINING - Device: {device}")

    train_loader, test_loader, num_classes, img_size = get_dataloader('mnist', args.batch_size)

    model = MinimalHoloGen(
        field_size=args.field_size,
        img_size=img_size,
        channels=1,
        num_classes=num_classes
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs("samples", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            generated, logits = model(labels, resonance_steps=args.steps)

            # Reconstruct loss + Classification loss (for semantic alignment)
            recon_loss = F.mse_loss(generated, images)
            class_loss = F.cross_entropy(logits, labels)
            loss = recon_loss + 0.1 * class_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += recon_loss.item()
            
            # Accuracy metric
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix(recon=f"{total_loss / (pbar.n + 1):.6f}", acc=f"{100 * correct / total:.2f}%")

        scheduler.step()

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                _, logits = model(labels, resonance_steps=args.steps)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1} Results -> Recon Loss: {total_loss / len(train_loader):.6f} | Test Acc: {val_acc:.2f}%")

        # Save sample grid every epoch
        with torch.no_grad():
            sample_labels = torch.arange(num_classes).to(device)
            samples, _ = model(sample_labels, resonance_steps=args.steps)
            grid = torchvision.utils.make_grid(samples, nrow=int(math.sqrt(num_classes)), normalize=True)
            plt.imsave(f"samples/mnist_epoch_{epoch+1:02d}.png", grid.permute(1, 2, 0).cpu().numpy())

    torch.save(model.state_dict(), f"checkpoints/holo_mnist_final.pth")
    print("Training finished!")


if __name__ == "__main__":
    train()
