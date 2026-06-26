import os, sys, traceback
import datasets
import math
import torch.optim as optim
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import argparse


PHI = (1 + math.sqrt(5)) / 2

class MinimalHoloGenFlickr(nn.Module):
    """Minimal Holographic Generator for Text-to-Image (Flickr30k)"""
    def __init__(self, field_size=1024, img_size=128, channels=3, vocab_size=10000):
        super().__init__()
        self.field_size = field_size
        self.img_size = img_size
        self.channels = channels

        # Reference beam: Text encoding
        # We use a simple embedding for "honesty", but this projects to the field
        self.text_embed = nn.Embedding(vocab_size, 256)
        self.text_proj = nn.Linear(256, field_size)

        # Learned holographic geometry
        self.G = nn.Parameter(torch.randn(field_size, field_size) * 0.01)

        # Phi binding
        self.phi_coupling = nn.Parameter(torch.tensor(0.15))
        self.phi_speed = nn.Parameter(torch.ones(field_size) * (2 * math.pi / PHI))

        # Memory
        self.memory_rate = nn.Parameter(torch.ones(field_size) * 0.1)

        # Readout
        self.readout = nn.Linear(field_size, img_size * img_size * channels)

        # Fibonacci boundary
        indices = torch.arange(field_size)
        golden_angle = 2 * math.pi / (PHI * PHI)
        self.register_buffer('boundary_shape', 1.0 + 0.5 * torch.cos(indices * golden_angle))

        # Persistent Standing Wave Pool
        # For text-to-image, we pool based on a hash of the text or a fixed number of "concept slots"
        # For now, we'll use a pool for "global concept memory"
        self.register_buffer('standing_wave', torch.zeros(1, field_size))
        self.register_buffer('pool_rate', torch.tensor(0.001))

    def forward(self, token_ids, resonance_steps=16):
        batch = token_ids.shape[0]
        device = token_ids.device

        # Initial excitation from text (reference beam)
        # Average token embeddings for the prompt
        embedded = self.text_embed(token_ids).mean(dim=1)
        state = self.text_proj(embedded)
        
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

            # Memory update
            rate = torch.sigmoid(self.memory_rate)
            memory = (1 - rate) * memory + rate * state

            # Formula 1
            grad_memory = memory - prev_memory
            grad_state = state - prev_state
            E = torch.matmul(grad_memory, self.G)
            coherence = torch.norm(E, dim=-1, keepdim=True) / math.sqrt(self.field_size)

            # Gentle feedback
            state = state + E * coherence * 0.05

            # Fibonacci boundary
            state = F.normalize(state, dim=-1) * math.sqrt(self.field_size)
            state = state * self.boundary_shape
            
            all_states.append(state)

        # Mean pool
        transient_wave = torch.stack(all_states, dim=1).mean(dim=1)

        # Inter-epoch pooling (Global concept memory)
        if self.training:
            with torch.no_grad():
                avg_transient = transient_wave.mean(dim=0, keepdim=True)
                self.standing_wave[:] = (1 - self.pool_rate) * self.standing_wave + self.pool_rate * avg_transient

        # Readout from the merged Standing Wave (current prompt + global memory)
        final_state = 0.7 * transient_wave + 0.3 * self.standing_wave.expand(batch, -1)

        img_flat = self.readout(final_state)
        img = img_flat.view(batch, self.channels, self.img_size, self.img_size)
        return torch.sigmoid(img)

class FlickrTextDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, vocab=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.vocab = vocab or {"<pad>": 0, "<unk>": 1}
        self.max_len = 24
        
        if vocab is None:
            print("Building vocab...")
            for item in tqdm(self.dataset):
                for caption in item['caption']:
                    for word in caption.lower().split():
                        if word not in self.vocab:
                            self.vocab[word] = len(self.vocab)
                        if len(self.vocab) >= 10000: break
                if len(self.vocab) >= 10000: break

    def tokenize(self, text):
        tokens = [self.vocab.get(w, 1) for w in text.lower().split()][:self.max_len]
        tokens += [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Pick a random caption from the 5 available
        import random
        caption = random.choice(item['caption'])
        tokens = self.tokenize(caption)
        
        return image, tokens, caption

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--field_size', type=int, default=1024)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--steps', type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"HOLOGRAPHIC TEXT-TO-IMAGE (FLICKR) - Device: {device}")

    print("Loading Flickr30k...")
    cache_dir = r"C:\Users\dingle\.cache\huggingface\datasets"
    raw_dataset = load_dataset("nlphuji/flickr30k", cache_dir=cache_dir, split='test')
    
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    
    dataset = FlickrTextDataset(raw_dataset, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = MinimalHoloGenFlickr(
        field_size=args.field_size,
        img_size=args.img_size,
        vocab_size=len(dataset.vocab)
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs("samples", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    test_captions = [
        "A dog running in the grass",
        "A person riding a bike",
        "A red car on the street",
        "A group of people standing outside"
    ]
    test_tokens = torch.stack([dataset.tokenize(c) for c in test_captions]).to(device)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, tokens, _ in pbar:
            images, tokens = images.to(device), tokens.to(device)

            optimizer.zero_grad()
            generated = model(tokens, resonance_steps=args.steps)

            loss = F.mse_loss(generated, images)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(recon=f"{total_loss / (pbar.n + 1):.6f}")

        scheduler.step()

        # Save sample grid every epoch
        model.eval()
        with torch.no_grad():
            samples = model(test_tokens, resonance_steps=args.steps)
            grid = torchvision.utils.make_grid(samples, nrow=2, normalize=True)
            plt.imsave(f"samples/flickr_text_epoch_{epoch+1:02d}.png", grid.permute(1, 2, 0).cpu().numpy())
        
        print(f"Epoch {epoch+1} Results -> Recon Loss: {total_loss / len(loader):.6f}")

    torch.save(model.state_dict(), f"checkpoints/holo_flickr_text_final.pth")
    print("Training finished!")

if __name__ == "__main__":
    train()
