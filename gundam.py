"""
GundamPilotJanis — 64-arm holonomic controller.

64 independent actuators tracking 64 known target trajectories.
Deterministic ground truth. Measurable per-arm error.
Prove she can coordinate 64 outputs through interference before
trusting her to drive the LLM.

Task: at each timestep, observe 64 current positions + 64 target positions
      → output 64 control signals → drive each arm to its target.

Targets are deterministic sinusoids with different frequencies/phases/amplitudes.
Known optimal. If she nails this, she's ready.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

PHI = (1 + math.sqrt(5)) / 2

N_ARMS =  2000
FIELD_SIZE = 512
RESONANCE_STEPS = 16
SEQ_LEN = 400
BATCH_SIZE = 32
EPOCHS = 300
LR = 1e-3


# ============================================================
# TARGET GENERATOR — 64 deterministic trajectories
# ============================================================
def generate_targets(batch_size, seq_len, n_arms=N_ARMS):
    """
    64 sinusoidal targets with different frequencies, phases, amplitudes.
    Fully deterministic for a given seed. Known ground truth.
    """
    t = torch.linspace(0, 4 * math.pi, seq_len).unsqueeze(0).unsqueeze(-1)  # (1, seq, 1)
    t = t.expand(batch_size, -1, -1)

    # Each arm gets a unique frequency, phase, amplitude
    freqs = torch.linspace(0.5, 3.0, n_arms).unsqueeze(0).unsqueeze(0)   # (1, 1, arms) — within plant bandwidth
    phases = torch.linspace(0, 2 * math.pi, n_arms).unsqueeze(0).unsqueeze(0)
    amps = torch.linspace(0.3, 1.0, n_arms).unsqueeze(0).unsqueeze(0)

    targets = amps * torch.sin(freqs * t + phases)  # (batch, seq, arms)
    return targets


# ============================================================
# PLANT — simple first-order system per arm
# ============================================================
def simulate_plant(controls, dt=0.05, damping=0.9):
    """
    64 independent first-order damped systems.
    position[t+1] = damping * position[t] + dt * control[t]

    controls: (batch, seq, n_arms)
    returns:  (batch, seq, n_arms) — actual positions
    """
    batch, seq_len, n_arms = controls.shape
    positions = torch.zeros(batch, seq_len, n_arms, device=controls.device)

    pos = torch.zeros(batch, n_arms, device=controls.device)
    for t in range(seq_len):
        pos = damping * pos + dt * controls[:, t, :]
        positions[:, t, :] = pos

    return positions


# ============================================================
# GUNDAM PILOT — holonomic field controller
# ============================================================
class GundamPilotJanis(nn.Module):
    """
    64-arm controller. Formula 1 resonance loop.
    Observes: current position (64) + target position (64) + error (64) = 192 inputs
    Outputs: 64 control signals
    """
    def __init__(self, n_arms=N_ARMS, field_size=FIELD_SIZE):
        super().__init__()
        self.field_size = field_size
        self.n_arms = n_arms

        # Input: position + target + error per arm
        self.input_proj = nn.Linear(n_arms * 3, field_size)

        # Dynamics
        self.G = nn.Parameter(torch.randn(field_size, field_size) * (1.0 / math.sqrt(field_size)))
        self.feedback_strength = nn.Parameter(torch.ones(field_size) * 0.1)
        self.memory_rate = nn.Parameter(torch.ones(field_size) * 0.15)

        # 64 arm outputs
        self.readout = nn.Linear(field_size, n_arms)

        # Boundary
        indices = torch.arange(field_size, dtype=torch.float32)
        golden_angle = 2 * math.pi / (PHI * PHI)
        self.register_buffer('boundary_shape', 1.0 + 0.5 * torch.cos(indices * golden_angle))

    def formula_1(self, grad_C, grad_A):
        interference = grad_C * grad_A
        E = interference @ self.G
        coherence = E.norm(dim=-1, keepdim=True) / math.sqrt(self.field_size)
        return E, coherence

    def forward(self, positions, targets):
        """
        positions: (batch, seq, n_arms) — current plant state
        targets:   (batch, seq, n_arms) — desired positions
        returns:   (batch, seq, n_arms) — control signals
        """
        batch, seq_len, _ = positions.shape
        device = positions.device

        state = torch.zeros(batch, self.field_size, device=device)
        memory = torch.zeros(batch, self.field_size, device=device)
        prev_state = state.clone()
        prev_memory = memory.clone()
        prev_grad = torch.zeros(batch, self.field_size, device=device)

        rate = torch.sigmoid(self.memory_rate)
        sqrt_field = math.sqrt(self.field_size)

        controls = []

        for t in range(seq_len):
            error = targets[:, t, :] - positions[:, t, :]
            obs = torch.cat([positions[:, t, :], targets[:, t, :], error], dim=-1)

            # Imprint observation
            excitation = self.input_proj(obs)
            state = state + excitation

            # Memory
            memory = (1 - rate) * memory + rate * state

            # Formula 1
            grad_state = state - prev_state
            grad_memory = memory - prev_memory
            accel = grad_state - prev_grad

            state_pre = state.clone()

            E, coherence = self.formula_1(grad_memory, grad_state + accel)
            state = state + E * coherence * self.feedback_strength

            prev_grad = grad_state.clone()

            # Containment
            magnitude = state.norm(dim=-1, keepdim=True) + 1e-8
            containment = sqrt_field / torch.sqrt(magnitude ** 2 + sqrt_field ** 2) * math.sqrt(2.0)
            state = state * containment * self.boundary_shape

            state = 2 * state - state_pre

            # Output 64 control signals
            ctrl = self.readout(state)
            controls.append(ctrl)

            prev_state = state.clone()
            prev_memory = memory.clone()

        return torch.stack(controls, dim=1)


# ============================================================
# DIFFERENTIABLE TRAINING LOOP
# The plant is differentiable so gradients flow:
# loss ← positions ← controls ← pilot
# ============================================================
def train_open_loop():
    """
    Open loop: pilot sees targets, outputs controls.
    Plant simulates. Loss = MSE(positions, targets).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  GUNDAM PILOT JANIS — {N_ARMS} Arms")
    print(f"  Field: {FIELD_SIZE} | Resonance: per-step")
    print(f"  Seq: {SEQ_LEN} | Batch: {BATCH_SIZE}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    pilot = GundamPilotJanis(n_arms=N_ARMS, field_size=FIELD_SIZE).to(device)
    params = sum(p.numel() for p in pilot.parameters())
    print(f"  Pilot params: {params:,}")

    optimizer = torch.optim.AdamW(pilot.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    targets = generate_targets(BATCH_SIZE, SEQ_LEN).to(device)

    best_mse = float('inf')
    best_per_arm = None

    for epoch in range(EPOCHS):
        pilot.train()
        t0 = time.time()

        # Closed-loop: pilot controls → plant → positions → pilot observes
        # But we need to unroll through time differentiably
        batch = targets.shape[0]
        pos = torch.zeros(batch, N_ARMS, device=device)
        all_positions = []
        all_controls = []

        state = torch.zeros(batch, FIELD_SIZE, device=device)
        memory = torch.zeros(batch, FIELD_SIZE, device=device)
        prev_state = state.clone()
        prev_memory = memory.clone()
        prev_grad = torch.zeros(batch, FIELD_SIZE, device=device)
        rate = torch.sigmoid(pilot.memory_rate)
        sqrt_field = math.sqrt(FIELD_SIZE)

        for t in range(SEQ_LEN):
            error = targets[:, t, :] - pos
            obs = torch.cat([pos, targets[:, t, :], error], dim=-1)

            excitation = pilot.input_proj(obs)
            state = state + excitation

            memory = (1 - rate) * memory + rate * state

            grad_state = state - prev_state
            grad_memory = memory - prev_memory
            accel = grad_state - prev_grad

            state_pre = state.clone()

            E, coherence = pilot.formula_1(grad_memory, grad_state + accel)
            state = state + E * coherence * pilot.feedback_strength

            prev_grad = grad_state.clone()

            magnitude = state.norm(dim=-1, keepdim=True) + 1e-8
            containment = sqrt_field / torch.sqrt(magnitude ** 2 + sqrt_field ** 2) * math.sqrt(2.0)
            state = state * containment * pilot.boundary_shape

            state = 2 * state - state_pre

            ctrl = pilot.readout(state)
            all_controls.append(ctrl)

            # Plant step — differentiable
            pos = 0.9 * pos + 0.05 * ctrl
            all_positions.append(pos.clone())

            prev_state = state.clone()
            prev_memory = memory.clone()

        positions = torch.stack(all_positions, dim=1)  # (batch, seq, arms)
        controls = torch.stack(all_controls, dim=1)

        loss = F.mse_loss(positions, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pilot.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - t0
        mse = loss.item()

        # Per-arm MSE
        with torch.no_grad():
            per_arm_mse = ((positions - targets) ** 2).mean(dim=(0, 1))  # (n_arms,)
            worst_arm = per_arm_mse.argmax().item()
            best_arm = per_arm_mse.argmin().item()
            median_arm = per_arm_mse.median().item()

        if mse < best_mse:
            best_mse = mse
            best_per_arm = per_arm_mse.clone()
            marker = " *"
        else:
            marker = ""

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | MSE: {mse:.6f} | Best: {best_mse:.6f} | "
                  f"Worst arm [{worst_arm}]: {per_arm_mse[worst_arm]:.4f} | "
                  f"Best arm [{best_arm}]: {per_arm_mse[best_arm]:.6f} | "
                  f"Median: {median_arm:.4f} | {elapsed:.1f}s{marker}")

    # Final report
    print(f"\n{'='*60}")
    print(f"  FINAL REPORT — {N_ARMS} Arms")
    print(f"{'='*60}")
    print(f"  Overall MSE:  {best_mse:.6f}")
    print(f"  Pilot params: {params:,}")

    # Sort arms by performance
    sorted_arms = best_per_arm.argsort()
    print(f"\n  Top 5 arms (lowest error):")
    for i in range(5):
        arm = sorted_arms[i].item()
        print(f"    Arm {arm:2d}: MSE {best_per_arm[arm]:.6f}")

    print(f"\n  Bottom 5 arms (highest error):")
    for i in range(5):
        arm = sorted_arms[-(i+1)].item()
        print(f"    Arm {arm:2d}: MSE {best_per_arm[arm]:.6f}")

    # Pass/fail per arm
    threshold = 0.01
    passed = (best_per_arm < threshold).sum().item()
    print(f"\n  Arms below {threshold} MSE: {passed}/{N_ARMS}")

    threshold2 = 0.05
    passed2 = (best_per_arm < threshold2).sum().item()
    print(f"  Arms below {threshold2} MSE: {passed2}/{N_ARMS}")

    threshold3 = 0.10
    passed3 = (best_per_arm < threshold3).sum().item()
    print(f"  Arms below {threshold3} MSE: {passed3}/{N_ARMS}")

    # Coordination metric: are neighboring arms correlated in their errors?
    arm_corr = torch.corrcoef(best_per_arm.unsqueeze(0).expand(2, -1))[0, 1].item()

    print(f"\n  Mean per-arm MSE:   {best_per_arm.mean():.6f}")
    print(f"  Std per-arm MSE:    {best_per_arm.std():.6f}")
    print(f"  Max/Min ratio:      {best_per_arm.max()/best_per_arm.min():.1f}x")

    print(f"\n{'='*60}")
    if passed == N_ARMS:
        print(f"  ALL {N_ARMS} ARMS PASSED. She's ready.")
    elif passed2 == N_ARMS:
        print(f"  ALL {N_ARMS} ARMS BELOW 0.05. Close.")
    else:
        print(f"  {N_ARMS - passed2} arms still above 0.05. Needs work.")
    print(f"{'='*60}")


if __name__ == '__main__':
    train_open_loop()
