import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.transforms as T
import os
from tqdm import tqdm
import math
import argparse

PHI = (1 + math.sqrt(5)) / 2

class MinimalHoloAudio(nn.Module):
    """Minimal Holographic Audio Classifier - same core as image version"""
    def __init__(self, field_size=512, num_classes=12):
        super().__init__()
        self.field_size = field_size
        self.num_classes = num_classes

        # Reference beam: audio spectrogram → field excitation
        self.mel_spec = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=128,
            power=1.0
        )
        self.spec_to_field = nn.Linear(128, field_size)

        # Holographic geometry
        self.G = nn.Parameter(torch.randn(field_size, field_size) * 0.02)

        # Phi binding
        self.phi_coupling = nn.Parameter(torch.tensor(0.18))
        self.phi_speed = nn.Parameter(torch.ones(field_size) * (2 * math.pi / PHI))

        # Memory
        self.memory_rate = nn.Parameter(torch.ones(field_size) * 0.15)

        # Readout
        self.readout = nn.Linear(field_size, num_classes)

        # Fibonacci boundary
        indices = torch.arange(field_size)
        golden_angle = 2 * math.pi / (PHI * PHI)
        self.register_buffer('boundary_shape', 1.0 + 0.5 * torch.cos(indices * golden_angle))

    def forward(self, waveform, resonance_steps=12):
        batch = waveform.shape[0]
        device = waveform.device

        # Convert waveform → mel spectrogram → field excitation
        spec = self.mel_spec(waveform)                    # (B, 128, time)
        spec = spec.mean(dim=2)                           # average over time → (B, 128)
        state = self.spec_to_field(spec)                  # (B, field_size)

        memory = state.clone()
        phi_t = torch.zeros(batch, self.field_size, device=device)

        all_states = []

        for _ in range(resonance_steps):
            prev_state = state.clone()
            prev_memory = memory.clone()

            # Phi binding
            phi_wave = torch.sin(phi_t)
            state = state * (1.0 + self.phi_coupling * phi_wave)
            phi_t = phi_t + self.phi_speed

            # Memory
            rate = torch.sigmoid(self.memory_rate)
            memory = (1 - rate) * memory + rate * state

            # Formula 1 as measurement
            grad_memory = memory - prev_memory
            grad_state = state - prev_state
            E = torch.matmul(grad_memory, self.G)
            coherence = torch.norm(E, dim=-1, keepdim=True) / math.sqrt(self.field_size)

            # Gentle feedback
            state = state + E * coherence * 0.08

            # Fibonacci boundary
            state = F.normalize(state, dim=-1) * math.sqrt(self.field_size)
            state = state * self.boundary_shape

            all_states.append(state)

        # Max pooling over resonance steps — catch the resonance peak
        state = torch.stack(all_states, dim=1).max(dim=1).values

        # Final classification
        logits = self.readout(state)
        return logits


LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
          "bed", "bird", "cat", "dog", "eight", "five", "four", "happy", "house",
          "marvin", "nine", "one", "seven", "sheila", "six", "three", "tree", "two",
          "wow", "zero", "backward", "forward", "follow", "learn", "visual"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}


def collate_fn(batch):
    waveforms = []
    labels = []
    for waveform, sample_rate, label, *_ in batch:
        if label not in LABEL_TO_IDX:
            continue
        if waveform.shape[1] > 16000:
            waveform = waveform[:, :16000]
        else:
            waveform = F.pad(waveform, (0, 16000 - waveform.shape[1]))
        waveforms.append(waveform)
        labels.append(LABEL_TO_IDX[label])
    if len(waveforms) == 0:
        return torch.zeros(1, 16000), torch.zeros(1, dtype=torch.long)
    return torch.stack(waveforms).squeeze(1), torch.tensor(labels)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--field_size', type=int, default=512)
    parser.add_argument('--steps', type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_root = r"E:\janis5\trans\2\ML\motherfucker\janis\Masterclass\More_holo\data"

    train_set = torchaudio.datasets.SPEECHCOMMANDS(data_root, subset="training", download=False)
    test_set = torchaudio.datasets.SPEECHCOMMANDS(data_root, subset="testing", download=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=8, pin_memory=True)

    model = MinimalHoloAudio(field_size=args.field_size, num_classes=len(LABELS)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for waveforms, labels in pbar:
            waveforms, labels = waveforms.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(waveforms, resonance_steps=args.steps)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}", acc=f"{100*correct/total:.2f}%")

        scheduler.step()

        print(f"Epoch {epoch+1} completed. Avg Loss: {total_loss/len(train_loader):.4f} | Train Acc: {100*correct/total:.2f}%")

        # Quick validation
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for waveforms, labels in test_loader:
                    waveforms, labels = waveforms.to(device), labels.to(device)
                    logits = model(waveforms, resonance_steps=args.steps)
                    pred = logits.argmax(dim=1)
                    val_correct += (pred == labels).sum().item()
                    val_total += labels.size(0)
            print(f"Validation Accuracy: {100*val_correct/val_total:.2f}%")

    torch.save(model.state_dict(), "checkpoints/holo_audio_final.pth")
    print("Training finished!")


if __name__ == "__main__":
    train()