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
    def __init__(self, field_size=256, n_eigen=128, num_classes=12):
        super().__init__()
        self.field_size = field_size
        self.n_eigen = n_eigen
        self.num_classes = num_classes

        self.mel_spec = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=128,
            power=0.5
        )
        self.spec_to_field = nn.Linear(128, field_size)

        # Spectral components replacing monolithic G
        self.log_eigenvalues = nn.Parameter(torch.randn(n_eigen) * 0.1)
        self.eigenbasis = nn.Parameter(torch.randn(field_size, n_eigen) * (1.0 / math.sqrt(field_size)))
        
        self.feedback_strength = nn.Parameter(torch.ones(field_size) * 0.1)
        self.memory_rate = nn.Parameter(torch.ones(field_size) * 0.25)

        self.readout = nn.Linear(field_size, num_classes)

        indices = torch.arange(field_size, dtype=torch.float32)
        golden_angle = 2 * math.pi / (PHI * PHI)
        self.register_buffer('boundary_shape', 1.0 + 0.5 * torch.cos(indices * golden_angle))

    def hhl_projection(self, b):
        """Spectral inversion pulse replacing G-based formula_1."""
        eigenvalues = torch.exp(self.log_eigenvalues)
        betas = self.eigenbasis.T @ b.T
        weights = betas / (eigenvalues.unsqueeze(1) + 1e-4)
        x = self.eigenbasis @ weights
        return x.T

    def forward(self, waveform):
        batch = waveform.shape[0]
        device = waveform.device
        sqrt_field = math.sqrt(self.field_size)

        spec = self.mel_spec(waveform)
        frames = spec.transpose(1, 2)
        seq_len = frames.shape[1]

        state = torch.zeros(batch, self.field_size, device=device)
        memory = torch.zeros(batch, self.field_size, device=device)
        prev_state = state.clone()
        prev_memory = memory.clone()
        prev_grad = torch.zeros(batch, self.field_size, device=device)

        rate = torch.sigmoid(self.memory_rate)

        all_states = []

        for t in range(seq_len):
            state = state + self.spec_to_field(frames[:, t])

            memory = (1 - rate) * memory + rate * state

            # HHL-Proxy pulse
            E = self.hhl_projection(state)
            coherence = E.norm(dim=-1, keepdim=True) / sqrt_field
            state = state + E * coherence * self.feedback_strength

            magnitude = state.norm(dim=-1, keepdim=True) + 1e-8
            containment = sqrt_field / torch.sqrt(magnitude ** 2 + sqrt_field ** 2) * math.sqrt(2.0)
            state = state * containment * self.boundary_shape

            all_states.append(state)

        peak_state = torch.stack(all_states, dim=1).max(dim=1)[0]

        logits = self.readout(peak_state)
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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--field_size', type=int, default=512)
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

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for waveforms, labels in pbar:
            waveforms, labels = waveforms.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(waveforms)
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

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for waveforms, labels in test_loader:
                    waveforms, labels = waveforms.to(device), labels.to(device)
                    logits = model(waveforms)
                    pred = logits.argmax(dim=1)
                    val_correct += (pred == labels).sum().item()
                    val_total += labels.size(0)
            print(f"Validation Accuracy: {100*val_correct/val_total:.2f}%")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/holo_audio2_final.pth")
    print("Training finished!")


if __name__ == "__main__":
    train()
