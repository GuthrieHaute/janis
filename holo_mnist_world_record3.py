import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import math
import os
from tqdm import tqdm
import argparse

PHI = (1 + math.sqrt(5)) / 2

class HoloWorldRecord(nn.Module):
    def __init__(self, field_size=1024, img_size=32, num_classes=10):
        super().__init__()
        self.field_size = field_size
        self.img_size = img_size

        self.pixel_proj = nn.Linear(img_size * img_size, field_size)
        self.G = nn.Parameter(torch.randn(field_size, field_size) * (1.0 / math.sqrt(field_size)))
        self.feedback_strength = nn.Parameter(torch.ones(field_size) * 0.1)
        self.memory_rate = nn.Parameter(torch.ones(field_size) * 0.2)

        self.classifier = nn.Sequential(
            nn.Linear(field_size, field_size // 2),
            nn.GELU(),
            nn.LayerNorm(field_size // 2),
            nn.Linear(field_size // 2, num_classes)
        )

        indices = torch.arange(field_size, dtype=torch.float32)
        golden_angle = 2 * math.pi / (PHI * PHI)
        self.register_buffer('boundary_shape', 1.0 + 0.5 * torch.cos(indices * golden_angle))

    def formula_1(self, grad_C, grad_A):
        interference = grad_C * grad_A
        E = interference @ self.G
        coherence = E.norm(dim=-1, keepdim=True) / math.sqrt(self.field_size)
        return E, coherence

    def forward(self, images, resonance_steps=32):
        batch = images.shape[0]
        device = images.device
        sqrt_field = math.sqrt(self.field_size)

        x = images.view(batch, -1)
        state = self.pixel_proj(x)

        memory = state.clone()
        prev_state = state.clone()
        prev_memory = memory.clone()
        prev_grad = torch.zeros(batch, self.field_size, device=device)

        rate = torch.sigmoid(self.memory_rate)

        all_states = []

        for _ in range(resonance_steps):
            memory = (1 - rate) * memory + rate * state

            grad_state = state - prev_state
            grad_memory = memory - prev_memory
            accel = grad_state - prev_grad

            state_pre = state.clone()

            E, coherence = self.formula_1(grad_memory, grad_state + accel)
            state = state + E * coherence * self.feedback_strength

            prev_grad = grad_state.clone()

            magnitude = state.norm(dim=-1, keepdim=True) + 8e-1
            containment = sqrt_field / torch.sqrt(magnitude ** 2 + sqrt_field ** 2) * math.sqrt(2.0)
            state = state * containment * self.boundary_shape

            state = 2 * state - state_pre

            all_states.append(state)

            prev_state = state.clone()
            prev_memory = memory.clone()

        peak_state = torch.stack(all_states, dim=1).max(dim=1)[0]

        logits = self.classifier(peak_state)
        return logits

def get_dataloader(batch_size=1028):
    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    root = "data"
    train_set = torchvision.datasets.MNIST(root=root, train=True, download=False, transform=transform_train)
    test_set = torchvision.datasets.MNIST(root=root, train=False, download=False, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--field_size', type=int, default=256)
    parser.add_argument('--steps', type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[!!!] MNIST WORLD RECORD ATTEMPT v3 [!!!]")
    print(f"Device: {device} | Field: {args.field_size} | Steps: {args.steps}")
    print(f"Accel + Grover | Corrected Formula 1 | Soft Containment\n")

    train_loader, test_loader = get_dataloader(args.batch_size)

    model = HoloWorldRecord(
        field_size=args.field_size,
        img_size=32
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                            steps_per_epoch=len(train_loader),
                                            epochs=args.epochs)

    best_acc = 0.0
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
            logits = model(images, resonance_steps=args.steps)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{total_loss / (pbar.n + 1):.4f}", acc=f"{100 * correct / total:.2f}%")

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images, resonance_steps=args.steps)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/holo_mnist_record3_best.pth")
            marker = " [NEW RECORD!]"
        else:
            marker = ""

        print(f"Epoch {epoch+1} -> Test Acc: {val_acc:.4f}% | Best: {best_acc:.4f}%{marker}")

    print(f"\nTraining Complete. Final Best Accuracy: {best_acc:.4f}%")

if __name__ == "__main__":
    train()
