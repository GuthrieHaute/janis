"""
JANIS² Nuclear — Joint Accelerated Nuclear Intelligence System  v4.1
=====================================================================
Coupled proton/neutron shell dynamics predicting nuclear ground state
stability, decay characteristics, shell closure signatures, and
continuous gradient position from known to undiscovered isotopes.

Shell model:   Woods-Saxon / spin-orbit ordering, 29 levels
Magic numbers: 2, 8, 20, 28, 50, 82, 126, 184

Classification:
  I    — Stable
  II   — Stable (energetic)
  III  — Resonant SHE
  IV   — Exotic bound
  V    — Unbound / unstable

Run modes:
  python nuclear_sd.py                        # autopilot: Massive Discovery Run (Full Z/N Sweep)
  python nuclear_sd.py single Z N [label]     # single isotope, verbose
  python nuclear_sd.py sweep                  # Z/N landscape → CSV
  python nuclear_sd.py island                 # island of stability → CSV
  python nuclear_sd.py gradient               # gradient anomaly report
  python nuclear_sd.py analyze <file.csv>     # rank and filter CSV results
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import math
import csv
import time
import sys
import os
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Shell model foundation
# ─────────────────────────────────────────────────────────────────────────────

MAGIC_NUMBERS = [2, 8, 20, 28, 50, 82, 126, 184]

SHELL_LEVELS = [
    (0,  "1s1/2",    2,    2),
    (1,  "1p3/2",    4,    6),
    (2,  "1p1/2",    2,    8),
    (3,  "1d5/2",    6,   14),
    (4,  "2s1/2",    2,   16),
    (5,  "1d3/2",    4,   20),
    (6,  "1f7/2",    8,   28),
    (7,  "2p3/2",    4,   32),
    (8,  "1f5/2",    6,   38),
    (9,  "2p1/2",    2,   40),
    (10, "1g9/2",   10,   50),
    (11, "1g7/2",    8,   58),
    (12, "2d5/2",    6,   64),
    (13, "2d3/2",    4,   68),
    (14, "3s1/2",    2,   70),
    (15, "1h11/2",  12,   82),
    (16, "1h9/2",   10,   92),
    (17, "2f7/2",    8,  100),
    (18, "2f5/2",    6,  106),
    (19, "3p3/2",    4,  110),
    (20, "3p1/2",    2,  112),
    (21, "1i13/2",  14,  126),
    (22, "1i11/2",  12,  138),
    (23, "2g9/2",   10,  148),
    (24, "2g7/2",    8,  156),
    (25, "3d5/2",    6,  162),
    (26, "3d3/2",    4,  166),
    (27, "4s1/2",    2,  168),
    (28, "1j15/2",  16,  184),
]

N_ORDERS = len(SHELL_LEVELS)

DECAY_MODES = {
    "stable":  "Stable",
    "alpha":   "α decay",
    "beta_m":  "β⁻ decay",
    "beta_p":  "β⁺ / EC",
    "sf":      "Spontaneous fission",
    "p_emit":  "Proton emission",
    "n_emit":  "Neutron emission",
    "unknown": "Unknown / theoretical",
}

CLASSIFICATION_LABELS = {
    "I":   "Stable",
    "II":  "Stable — energetic",
    "III": "Resonant SHE",
    "IV":  "Exotic bound state",
    "V":   "Unbound / unstable",
}

HALFLIFE_TIERS = {
    "I":   "Stable / cosmologically long",
    "II":  "Long-lived (>10⁶ yr) to geologically stable",
    "III": "Short to very short (<10³ yr predicted)",
    "IV":  "Unknown — candidate for measurement",
    "V":   "Very short (<ms) or unbound",
}

CLASSIFICATION_COLORS_ANSI = {
    "I": "\033[92m", "II": "\033[94m", "III": "\033[95m",
    "IV": "\033[96m", "V":  "\033[91m",
}
CLASSIFICATION_COLORS_HEX = {
    "I": "#2ecc71", "II": "#4a9eff", "III": "#b06fff",
    "IV": "#1abcb0", "V":  "#ff4757",
}
RESET = "\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# Shell oscillator parameters
# ─────────────────────────────────────────────────────────────────────────────

def shell_theta(shell_idx: int, A: float) -> float:
    omega_0  = 41.0 * (max(A, 1) ** (-1.0 / 3.0))
    j_factor = SHELL_LEVELS[shell_idx][3] / 184.0
    return omega_0 * (1.0 + 0.3 * j_factor) / 100.0


def shell_alpha(shell_idx: int, Z: int, N: int, is_proton: bool) -> float:
    nc         = Z if is_proton else N
    _, _, cap, _ = SHELL_LEVELS[shell_idx]
    prev_cum   = SHELL_LEVELS[shell_idx - 1][3] if shell_idx > 0 else 0
    dist_magic = min(abs(nc - m) for m in MAGIC_NUMBERS)
    proximity  = 1.0 - min(dist_magic / 20.0, 1.0)
    shell_occ  = nc - prev_cum
    fill_frac  = max(0.0, min(shell_occ / cap, 1.0)) if cap > 0 else 0.0
    return float(np.clip(0.15 + 0.4 * proximity + 0.2 * fill_frac, 0.05, 0.95))


# ─────────────────────────────────────────────────────────────────────────────
# Valley baseline
# ─────────────────────────────────────────────────────────────────────────────

def valley_centerline_N(Z: int) -> float:
    if Z <= 20:
        return float(Z)
    ratio = 1.0 + (0.711 * Z) / (4.0 * 23.7)
    return float(Z * min(ratio, 1.54))


def valley_BE_expected(A: int) -> float:
    if A < 2: return 0.0
    Z_opt = max(1.0, min(A / (1.0 + 0.0075 * A ** (2.0 / 3.0)), A - 1))
    BE  = 15.75 * A - 17.8 * (A ** (2.0 / 3.0))
    BE -= 0.711 * Z_opt * (Z_opt - 1) / (A ** (1.0 / 3.0))
    BE -= 23.7  * ((A - 2 * Z_opt) ** 2) / A
    BE += 34.0  / (A ** (3.0 / 4.0))
    return float(np.clip(BE / A, 0.0, 8.8))


# ─────────────────────────────────────────────────────────────────────────────
# Nuclear configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NuclearConfig:
    Z: int
    N: int
    label: str = ""

    @property
    def A(self) -> int:
        return self.Z + self.N

    @property
    def symbol(self) -> str:
        return self.label or f"Z={self.Z}  N={self.N}  A={self.A}"

    @property
    def is_magic_Z(self) -> bool: return self.Z in MAGIC_NUMBERS
    @property
    def is_magic_N(self) -> bool: return self.N in MAGIC_NUMBERS
    @property
    def is_doubly_magic(self) -> bool: return self.is_magic_Z and self.is_magic_N

    def pairing_type(self) -> str:
        if self.Z % 2 == 0 and self.N % 2 == 0: return "even-even"
        if self.Z % 2 == 1 and self.N % 2 == 1: return "odd-odd"
        return "odd-A"

    def shell_closure_note(self) -> str:
        notes = []
        if self.is_doubly_magic:
            notes.append("doubly magic")
        else:
            if self.is_magic_Z: notes.append(f"magic Z={self.Z}")
            if self.is_magic_N: notes.append(f"magic N={self.N}")
        if not notes:
            if min(abs(self.Z - m) for m in MAGIC_NUMBERS) <= 2: notes.append("near magic Z")
            if min(abs(self.N - m) for m in MAGIC_NUMBERS) <= 2: notes.append("near magic N")
        return ", ".join(notes) if notes else ""

    def bethe_weizsacker(self) -> float:
        """
        Binding energy per nucleon (MeV/u).
        Pairing term: 34/A^(3/4) for even-even (well-fit across full range).
        Odd-odd: blends 12/sqrt(A) (light nuclei) → 34/A^(3/4) (heavy) across A=15-40.
        This corrects the systematic underestimate of light odd-odd nuclei
        (6Li, 10B, 14N) without touching even-even or medium/heavy nuclei.
        """
        if self.Z <= 0 or self.N < 0 or self.A <= 0: return 0.0
        A, Z = self.A, self.Z
        BE  =  15.75 * A
        BE -= 17.8   * (A ** (2.0 / 3.0))
        BE -= 0.711  * Z * (Z - 1) / (A ** (1.0 / 3.0))
        BE -= 23.7   * ((A - 2 * Z) ** 2) / A

        if Z % 2 == 0 and self.N % 2 == 0:
            BE += 34.0 / (A ** (3.0 / 4.0))
        elif Z % 2 == 1 and self.N % 2 == 1:
            w   = float(np.clip((A - 15) / 25.0, 0.0, 1.0))
            BE -= w * 34.0 / (A ** (3.0 / 4.0)) + (1.0 - w) * 12.0 / (A ** 0.5)

        return round(float(np.clip(BE / A, 0.0, 8.8)), 3)

    def valley_displacement(self) -> float:
        return round(self.N - valley_centerline_N(self.Z), 2)

    def shell_proximity_Z(self) -> float:
        return round(float(1.0 - min(min(abs(self.Z - m) for m in MAGIC_NUMBERS) / 14.0, 1.0)), 3)

    def shell_proximity_N(self) -> float:
        return round(float(1.0 - min(min(abs(self.N - m) for m in MAGIC_NUMBERS) / 14.0, 1.0)), 3)

    def shell_proximity_combined(self) -> float:
        return round((self.shell_proximity_Z() + self.shell_proximity_N()) / 2.0, 3)

    def binding_gradient_deviation(self) -> float:
        return round(self.bethe_weizsacker() - valley_BE_expected(self.A), 3)

    def separation_energy_n(self) -> float:
        """One-neutron separation energy Sn (MeV). Negative = beyond neutron drip."""
        if self.N <= 0: return 0.0
        be_this  = self.bethe_weizsacker() * self.A
        other    = NuclearConfig(self.Z, self.N - 1)
        be_other = other.bethe_weizsacker() * other.A
        return round(be_this - be_other, 3)

    def separation_energy_p(self) -> float:
        """One-proton separation energy Sp (MeV). Negative = beyond proton drip."""
        if self.Z <= 1: return 0.0
        be_this  = self.bethe_weizsacker() * self.A
        other    = NuclearConfig(self.Z - 1, self.N)
        be_other = other.bethe_weizsacker() * other.A
        return round(be_this - be_other, 3)

    def separation_energy_2n(self) -> float:
        """Two-neutron separation energy S2n (MeV). Key for even-N shell gaps."""
        if self.N <= 1: return 0.0
        be_this  = self.bethe_weizsacker() * self.A
        other    = NuclearConfig(self.Z, self.N - 2)
        be_other = other.bethe_weizsacker() * other.A
        return round(be_this - be_other, 3)

    def separation_energy_2p(self) -> float:
        """Two-proton separation energy S2p (MeV). Key for even-Z shell gaps."""
        if self.Z <= 2: return 0.0
        be_this  = self.bethe_weizsacker() * self.A
        other    = NuclearConfig(self.Z - 2, self.N)
        be_other = other.bethe_weizsacker() * other.A
        return round(be_this - be_other, 3)

    def alpha_q_value(self) -> float:
        """Q-value for alpha decay (MeV). Positive = energetically allowed."""
        if self.Z < 2 or self.N < 2: return 0.0
        be_parent   = self.bethe_weizsacker() * self.A
        daughter    = NuclearConfig(self.Z - 2, self.N - 2)
        be_daughter = daughter.bethe_weizsacker() * daughter.A
        be_alpha    = 7.074 * 4
        return round(be_daughter + be_alpha - be_parent, 3)

    def drip_line_status(self) -> str:
        """Drip line proximity: 'bound', 'n-drip', 'p-drip', or 'both-drip'."""
        sn = self.separation_energy_n()
        sp = self.separation_energy_p()
        n_drip = sn < 0
        p_drip = sp < 0
        if n_drip and p_drip: return "both-drip"
        if n_drip:            return "n-drip"
        if p_drip:            return "p-drip"
        return "bound"

    def deformation_beta2(self) -> float:
        """Rough quadrupole deformation from shell filling. 0 = spherical (magic)."""
        dist_z = min(abs(self.Z - m) for m in MAGIC_NUMBERS)
        dist_n = min(abs(self.N - m) for m in MAGIC_NUMBERS)
        gap_z = gap_n = 10
        for i, m in enumerate(MAGIC_NUMBERS):
            if self.Z <= m:
                gap_z = m - (MAGIC_NUMBERS[i - 1] if i > 0 else 0)
                break
        for i, m in enumerate(MAGIC_NUMBERS):
            if self.N <= m:
                gap_n = m - (MAGIC_NUMBERS[i - 1] if i > 0 else 0)
                break
        frac_z = min(dist_z / max(gap_z / 2, 1), 1.0)
        frac_n = min(dist_n / max(gap_n / 2, 1), 1.0)
        return round(0.3 * (frac_z + frac_n) / 2.0, 3)

    def r_process_candidate(self) -> bool:
        """Flag nuclei plausibly on the astrophysical r-process path."""
        if self.A < 60: return False
        sn = self.separation_energy_n()
        nz = self.N / max(self.Z, 1)
        return nz > 1.3 and 0 < sn < 5.0

    def fissility(self) -> float:
        """Fissility parameter x = Z²/A / 50.88. x > 1 = spontaneous fission."""
        if self.A <= 0: return 0.0
        return round((self.Z ** 2 / self.A) / 50.88, 4)

    def isobar_competition(self) -> Dict:
        """Compare BE against neighboring isobars (same A, different Z).
        If a neighbor has higher total BE, this isotope loses and will beta-decay.
        Note: BW lacks shell corrections, so doubly magic nuclei are always isobar-stable."""
        if self.is_doubly_magic:
            return {"isobar_stable": True, "isobar_delta_minus": 0.0, "isobar_delta_plus": 0.0}
        A = self.A
        be_self = self.bethe_weizsacker() * A
        # (Z-1, N+1) — beta-minus daughter
        d_minus = d_plus = 0.0
        iso_stable = True
        if self.Z > 1 and self.N >= 0:
            nb_m = NuclearConfig(self.Z - 1, self.N + 1)
            be_m = nb_m.bethe_weizsacker() * A
            d_minus = round(be_m - be_self, 3)
            if d_minus > 0: iso_stable = False
        # (Z+1, N-1) — electron-capture / beta-plus daughter
        if self.N > 0:
            nb_p = NuclearConfig(self.Z + 1, self.N - 1)
            be_p = nb_p.bethe_weizsacker() * A
            d_plus = round(be_p - be_self, 3)
            if d_plus > 0: iso_stable = False
        return {
            "isobar_stable":      iso_stable,
            "isobar_delta_minus": d_minus,
            "isobar_delta_plus":  d_plus,
        }

    def Q_beta_minus(self) -> float:
        """Q-value for beta-minus decay (MeV). Positive = energetically allowed."""
        if self.N < 1: return 0.0
        be_self = self.bethe_weizsacker() * self.A
        daughter = NuclearConfig(self.Z + 1, self.N - 1)
        be_d = daughter.bethe_weizsacker() * self.A
        return round(be_d - be_self, 3)

    def Q_beta_plus(self) -> float:
        """Q-value for beta-plus / EC decay (MeV). Positive = energetically allowed."""
        if self.Z < 2: return 0.0
        be_self = self.bethe_weizsacker() * self.A
        daughter = NuclearConfig(self.Z - 1, self.N + 1)
        be_d = daughter.bethe_weizsacker() * self.A
        return round(be_d - be_self, 3)

    def Q_2beta(self) -> float:
        """Q-value for double-beta decay (MeV). Relevant for even-even nuclei."""
        if self.Z < 2 or self.N < 2: return 0.0
        be_self = self.bethe_weizsacker() * self.A
        daughter = NuclearConfig(self.Z + 2, self.N - 2)
        be_d = daughter.bethe_weizsacker() * self.A
        return round(be_d - be_self, 3)

    def shell_gap_n(self) -> float:
        """Neutron shell gap: Delta_n = S2n(Z,N) - S2n(Z,N+2). Large = shell closure."""
        if self.N <= 3: return 0.0
        s2n_here = self.separation_energy_2n()
        other = NuclearConfig(self.Z, self.N + 2)
        s2n_next = other.separation_energy_2n()
        return round(s2n_here - s2n_next, 3)

    def shell_gap_p(self) -> float:
        """Proton shell gap: Delta_p = S2p(Z,N) - S2p(Z+2,N). Large = shell closure."""
        if self.Z <= 3: return 0.0
        s2p_here = self.separation_energy_2p()
        other = NuclearConfig(self.Z + 2, self.N)
        s2p_next = other.separation_energy_2p()
        return round(s2p_here - s2p_next, 3)

    def pairing_gap_n(self) -> float:
        """Neutron pairing gap from 3-point formula (MeV)."""
        if self.N < 1 or self.N >= 183: return 0.0
        be_m = NuclearConfig(self.Z, self.N - 1).bethe_weizsacker() * (self.A - 1)
        be_0 = self.bethe_weizsacker() * self.A
        be_p = NuclearConfig(self.Z, self.N + 1).bethe_weizsacker() * (self.A + 1)
        sign = (-1) ** self.N
        return round(sign * (be_m - 2 * be_0 + be_p) / 2.0, 3)

    def pairing_gap_p(self) -> float:
        """Proton pairing gap from 3-point formula (MeV)."""
        if self.Z < 2: return 0.0
        be_m = NuclearConfig(self.Z - 1, self.N).bethe_weizsacker() * (self.A - 1)
        be_0 = self.bethe_weizsacker() * self.A
        be_p = NuclearConfig(self.Z + 1, self.N).bethe_weizsacker() * (self.A + 1)
        sign = (-1) ** self.Z
        return round(sign * (be_m - 2 * be_0 + be_p) / 2.0, 3)

    def nuclear_radius(self) -> float:
        """Nuclear charge radius in fm. R = 1.2 * A^(1/3)."""
        if self.A <= 0: return 0.0
        return round(1.2 * (self.A ** (1.0 / 3.0)), 3)

    def neutron_skin(self) -> float:
        """Estimated neutron skin thickness in fm.
        Uses linear isospin dependence: delta_R ≈ 0.90 * (N-Z)/A + 0.01 fm."""
        if self.A <= 4: return 0.0
        return round(0.90 * (self.N - self.Z) / self.A + 0.01, 3)

    def estimated_halflife_log10(self) -> Optional[float]:
        """Rough log10(half-life in seconds) estimate.
        Uses Geiger-Nuttall for alpha, Sargent's rule for beta. None if stable.
        Returns None for genuinely stable or effectively eternal (>10^25 s)."""
        if self.A < 12: return None  # BW unreliable for very light nuclei
        # Doubly magic nuclei: BW lacks shell corrections, beta Q-values unreliable
        if self.is_doubly_magic: return None
        q_a = self.alpha_q_value()
        q_bm = self.Q_beta_minus()
        q_bp = self.Q_beta_plus()
        iso = self.isobar_competition()

        # If isobar-stable and no energetically allowed alpha, stable
        if iso["isobar_stable"] and q_a <= 0:
            return None  # stable

        estimates = []
        # Alpha: Geiger-Nuttall (fitted to U-238, Po-210, Ra-226)
        # log10(t½/s) ≈ 1.20 * Z / sqrt(Qα) - 36.0
        if q_a > 0.5 and self.Z >= 52:
            gn = 1.20 * self.Z / math.sqrt(q_a) - 36.0
            estimates.append(gn)
        # Spontaneous fission (for very heavy / high fissility)
        if self.fissility() > 0.75 and self.A > 240:
            # Rough: log10(t½) drops steeply with fissility above threshold
            sf = max(-3.0, 25.0 - 40.0 * self.fissility())
            estimates.append(sf)
        # Beta-minus: Sargent's rule log10(t½/s) ≈ 4.0 - 5*log10(Q)
        if q_bm > 0.1:
            sr = 4.0 - 5.0 * math.log10(max(q_bm, 0.01))
            estimates.append(sr)
        # Beta-plus/EC
        if q_bp > 0.1:
            sr = 4.0 - 5.0 * math.log10(max(q_bp, 0.01))
            estimates.append(sr)

        if not estimates: return None
        result = round(min(estimates), 1)
        # If longer than 10^20 s (~300× age of universe), effectively stable
        if result > 20.0 and iso["isobar_stable"]:
            return None
        return result

    def spin_parity_estimate(self) -> str:
        """Ground-state spin-parity from last unpaired nucleon's shell."""
        def _valence_shell(nc):
            for idx, (_, label, cap, cum) in enumerate(SHELL_LEVELS):
                if nc <= cum:
                    return idx, label
            return len(SHELL_LEVELS) - 1, SHELL_LEVELS[-1][1]

        if self.Z % 2 == 0 and self.N % 2 == 0:
            return "0+"
        if self.Z % 2 == 1 and self.N % 2 == 1:
            _, lbl_p = _valence_shell(self.Z)
            _, lbl_n = _valence_shell(self.N)
            return f"({lbl_p}x{lbl_n})"

        # Odd-A: spin from the odd nucleon
        nc = self.Z if self.Z % 2 == 1 else self.N
        _, label = _valence_shell(nc)
        # Extract j from label like "1f7/2" → "7/2"
        j_str = label.split("/")[0][-1] + "/2"
        # Parity from orbital l: s=0(+), p=1(-), d=2(+), f=3(-), g=4(+), h=5(-), i=6(+), j=7(-)
        orb_char = label[1]
        l_map = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6, "j": 7}
        l = l_map.get(orb_char, 0)
        parity = "+" if l % 2 == 0 else "-"
        return f"{j_str}{parity}"

    def gradient_position(self) -> Dict:
        disp   = self.valley_displacement()
        sp_c   = self.shell_proximity_combined()
        be_dev = self.binding_gradient_deviation()
        valley_bonus = max(0.0, 1.0 - abs(disp) / 10.0)
        return {
            "valley_displacement":   disp,
            "shell_proximity_Z":     self.shell_proximity_Z(),
            "shell_proximity_N":     self.shell_proximity_N(),
            "shell_proximity":       sp_c,
            "binding_deviation_mev": be_dev,
            "deviation_score":       round(be_dev + sp_c * 0.5 + valley_bonus * 0.3, 3),
        }

    def predicted_decay_mode(self, classification: str) -> str:
        Z, N, A = self.Z, self.N, self.A
        nz = N / max(Z, 1)

        # --- Single nucleon or bare nucleus ---
        if A <= 1 or (Z == 1 and N == 0):
            return DECAY_MODES["stable"]

        # --- Light nuclei (A <= 10): physics-driven ---
        if A <= 10:
            if classification == "V" or classification not in ("I",):
                if Z == 1 and N == 2:           return DECAY_MODES["beta_m"]   # tritium
                if Z == 2 and N > 2:            return DECAY_MODES["beta_m"]   # He-6, He-8
                if N == 0 or nz < 0.5:          return DECAY_MODES["p_emit"]
                if nz > 2.0:                    return DECAY_MODES["n_emit"]
                if N > Z and nz > 1.3:          return DECAY_MODES["beta_m"]
                if Z > N:                       return DECAY_MODES["beta_p"]
                return DECAY_MODES["beta_m"]
            return DECAY_MODES["stable"]

        # --- Stable / exotic bound, sub-Bi ---
        if classification in ("I", "IV") and A < 209:
            disp = self.valley_displacement()
            if disp > 8:                        return DECAY_MODES["beta_m"]
            if disp < -5:                       return DECAY_MODES["beta_p"]
            return DECAY_MODES["stable"]

        # --- Stable energetic ---
        if classification == "II":
            if A > 209 or Z > 83:               return DECAY_MODES["alpha"]
            disp = self.valley_displacement()
            if disp > 8:                        return DECAY_MODES["beta_m"]
            if disp < -5:                       return DECAY_MODES["beta_p"]
            return DECAY_MODES["stable"]

        # --- SHE resonant ---
        if classification == "III":
            if A > 270:                          return DECAY_MODES["sf"]
            return DECAY_MODES["alpha"]

        # --- Unbound ---
        if classification == "V":
            if nz > 1.5:                        return DECAY_MODES["beta_m"]
            if Z > N:                           return DECAY_MODES["p_emit"]
            if nz < 0.7:                        return DECAY_MODES["beta_p"]
            return DECAY_MODES["unknown"]

        return DECAY_MODES["unknown"]


# ─────────────────────────────────────────────────────────────────────────────
# Shell oscillator
# ─────────────────────────────────────────────────────────────────────────────

class ShellOscillator:
    EMISSION_THRESHOLD = 2.5

    def __init__(self, theta: float, alpha: float, dt: float):
        self.theta = theta; self.alpha = alpha; self.dt = dt
        self.mu_r = 0.0; self.mu_i = 0.0
        self.emission_count = 0; self.total_emitted = 0.0

    @property
    def energy(self) -> float: return self.mu_r ** 2 + self.mu_i ** 2
    def norm(self) -> float:   return math.sqrt(max(self.energy, 0.0))

    def step(self, drive: float) -> Optional[float]:
        angle     = self.theta * self.dt
        cr, ci    = math.cos(angle), math.sin(angle)
        new_r     = self.mu_r * cr - self.mu_i * ci
        new_i     = self.mu_r * ci + self.mu_i * cr
        self.mu_r = new_r * (1.0 - self.alpha) + drive * self.alpha
        self.mu_i = new_i * (1.0 - self.alpha)
        if self.energy > self.EMISSION_THRESHOLD ** 2:
            emitted = self.energy
            self.emission_count += 1; self.total_emitted += emitted
            self.mu_r *= 0.3; self.mu_i *= 0.3
            return emitted
        return None


class ShellCoupler:
    def __init__(self, n: int, c: float = 0.12):
        self.n = n; self.c = c

    def forces(self, osc: List[ShellOscillator]) -> List[float]:
        f = [0.0] * self.n
        for k in range(self.n):
            left  = osc[k - 1].mu_r if k > 0          else 0.0
            right = osc[k + 1].mu_r if k < self.n - 1 else 0.0
            f[k]  = self.c * (left + right - 2.0 * osc[k].mu_r)
        return f


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify(r: Dict) -> str:
    var = r["_var"]; sync = r["_sync"]; em = r["_emissions"]
    cfg = r["config"]; A = cfg.A; Z = cfg.Z; N = cfg.N
    nz = N / max(Z, 1)

    # --- Superheavy resonant: high-emission, locked sync ---
    if sync > 0.999 and em > 500 and A > 200:
        return "III"

    # --- Doubly magic override: exceptionally stable ---
    if cfg.is_doubly_magic and em == 0:
        if A <= 209: return "I"
        return "II"

    # --- Light nuclei (A <= 10): dynamics too clean, use physics ---
    if A <= 10:
        if A <= 1:                              return "I"
        if Z == 1 and N > 1:                    return "V"
        if Z == 2 and N > 2:                    return "V"
        if nz > 2.0 or (Z > 2 and nz < 0.5):   return "V"
        if Z > 2 and N == 0:                    return "V"
        if Z > N and A > 5:                     return "V"
        if abs(N - Z) <= 1:                     return "I"
        if nz > 1.5:                            return "V"
        return "I"

    # --- Valley proximity and ratio checks ---
    valley_n = valley_centerline_N(Z)
    valley_dist = abs(N - valley_n)
    near_valley = valley_dist < max(6, Z * 0.12)

    if A <= 40:   extreme = nz > 2.0 or nz < 0.5
    elif A <= 100: extreme = nz > 1.8 or nz < 0.6
    else:          extreme = nz > 1.7 or nz < 0.65

    # --- Unbound: extreme ratio or chaotic dynamics ---
    if extreme and not near_valley:
        return "V"
    if var > 0.025 and sync < 0.60:
        return "V"

    # --- Stable (I): clean dynamics, near valley, A < 209 ---
    if A <= 209 and em == 0 and sync > 0.85 and near_valley:
        if (A <= 60 and var < 0.020) or (A > 60 and var < 0.016):
            return "I"

    # --- Stable energetic (II): heavy near-valley or mild emissions ---
    if near_valley:
        if A > 209 or Z > 83:
            return "II"
        if em > 0 and var < 0.022 and sync > 0.55:
            return "II"

    # --- Exotic bound (IV): clean dynamics, unusual position ---
    if var < 0.014 and em == 0 and sync > 0.85 and A > 20:
        if near_valley:
            return "I"
        return "IV"

    # --- Superheavy fallback ---
    if A > 210 and em > 0:
        return "III"

    # --- Remaining unbound ---
    if var > 0.018 and sync < 0.70:
        return "V"
    if extreme:
        return "V"

    # --- Fallbacks with valley awareness ---
    if A <= 60:
        return "I" if near_valley else "V"
    if A <= 210:
        return "II" if near_valley else "V"
    return "III"


# ─────────────────────────────────────────────────────────────────────────────
# Dynamics engine
# ─────────────────────────────────────────────────────────────────────────────

class NuclearDynamicsEngine:
    _DT_G = 3.16
    _DT_E = 4.36

    def __init__(self, config: NuclearConfig):
        self.config   = config
        self.n_str    = 2; self.n_ord = N_ORDERS
        self._osc_G   = self._build(self._DT_G)
        self._osc_E   = self._build(self._DT_E)
        self._coupler = ShellCoupler(self.n_ord)
        self._history: List[float] = []
        self._decays  = 0

    def _build(self, dt: float):
        return [
            [ShellOscillator(
                shell_theta(k, self.config.A),
                shell_alpha(k, self.config.Z, self.config.N, s == 0), dt)
             for k in range(self.n_ord)]
            for s in range(self.n_str)
        ]

    def _drive(self, strand: int, order: int) -> float:
        nc = self.config.Z if strand == 0 else self.config.N
        _, _, cap, _ = SHELL_LEVELS[order]
        prev_cum = SHELL_LEVELS[order - 1][3] if order > 0 else 0
        occ      = max(0.0, min(nc - prev_cum, cap))
        fill     = occ / cap if cap > 0 else 0.0
        if order == 0: return fill
        p_cap  = SHELL_LEVELS[order - 1][2]
        p_pcum = SHELL_LEVELS[order - 2][3] if order > 1 else 0
        p_occ  = max(0.0, min(nc - p_pcum, p_cap))
        p_fill = p_occ / p_cap if p_cap > 0 else 0.0
        if order == 1: return fill - p_fill
        return fill - 2.0 * p_fill

    def _coulomb(self) -> List[float]:
        Z, A = self.config.Z, self.config.A
        cf   = Z * (Z - 1) / (A ** (1.0 / 3.0) + 1e-8) / 200.0
        return [cf * (k / self.n_ord) for k in range(self.n_ord)]

    def _isospin(self) -> float:
        return 23.7 * (self.config.N - self.config.Z) / (self.config.A + 1e-8)

    def _step(self, osc) -> int:
        coulomb = self._coulomb(); isospin = self._isospin(); n_emit = 0
        for s in range(self.n_str):
            cf = self._coupler.forces(osc[s])
            for k in range(self.n_ord):
                drive = self._drive(s, k) + cf[k]
                if s == 0: drive -= coulomb[k] * osc[s][k].mu_r
                drive += isospin * osc[1 - s][k].mu_r * 0.05
                if osc[s][k].step(drive) is not None: n_emit += 1
        return n_emit

    def _coherence(self, osc) -> float:
        norms = np.array([osc[s][k].norm()
                          for s in range(self.n_str) for k in range(self.n_ord)])
        p = norms / (norms.sum() + 1e-8)
        return float(-(p * np.log(p + 1e-8)).sum())

    def _pn_alignment(self) -> float:
        dot = pn = nn = 0.0
        for k in range(self.n_ord):
            pr = self._osc_G[0][k].mu_r; pi = self._osc_G[0][k].mu_i
            nr = self._osc_G[1][k].mu_r; ni = self._osc_G[1][k].mu_i
            dot += pr*nr + pi*ni; pn += pr**2 + pi**2; nn += nr**2 + ni**2
        return float((dot / math.sqrt(max(pn, 1e-8) * max(nn, 1e-8)) + 1.0) / 2.0)

    def _freq_coherence(self) -> float:
        dot = gn = en = 0.0
        for s in range(self.n_str):
            for k in range(self.n_ord):
                gr = self._osc_G[s][k].mu_r; gi = self._osc_G[s][k].mu_i
                er = self._osc_E[s][k].mu_r; ei = self._osc_E[s][k].mu_i
                dot += gr*er + gi*ei; gn += gr**2 + gi**2; en += er**2 + ei**2
        return float(np.clip(dot / math.sqrt(max(gn, 1e-12) * max(en, 1e-12)), -1.0, 1.0))

    def _active_shells(self) -> Tuple[List[str], List[str]]:
        tp = sorted(range(self.n_ord), key=lambda k: -self._osc_G[0][k].norm())[:3]
        tn = sorted(range(self.n_ord), key=lambda k: -self._osc_G[1][k].norm())[:3]
        return [SHELL_LEVELS[k][1] for k in tp], [SHELL_LEVELS[k][1] for k in tn]

    def _decay_rate_signal(self) -> float:
        return float(np.clip(
            self._decays / (max(len(self._history), 1) * self.n_str * self.n_ord), 0.0, 1.0))

    def solve(self, n_steps: int = 500, verbose: bool = False) -> Dict:
        WINDOW = 30; TOL = 0.02
        window = []; converged_at = None

        for step in range(n_steps):
            n_emit = self._step(self._osc_G)
            self._step(self._osc_E)
            self._decays += n_emit
            coh = self._coherence(self._osc_G)
            self._history.append(coh)
            window.append(coh)
            if len(window) > WINDOW: window.pop(0)
            if (len(window) == WINDOW and float(np.std(window)) < TOL
                    and converged_at is None):
                converged_at = step
                if not verbose: break
            if verbose and step % 25 == 0:
                print(f"  step {step:4d} | coh={coh:.4f} | "
                      f"p-n={self._pn_alignment():.3f} | "
                      f"fc={self._freq_coherence():+.4f} | "
                      f"decays={self._decays}")

        var  = float(np.std(self._history[-WINDOW:]))
        sync = self._pn_alignment()
        fc   = self._freq_coherence()
        p_sh, n_sh = self._active_shells()
        dr   = self._decay_rate_signal()

        raw = {
            "config": self.config,
            "_converged": converged_at is not None,
            "_conv_at":   converged_at,
            "_coherence": self._history[-1] if self._history else 0.0,
            "_var": var, "_sync": sync, "_fc": fc,
            "_emissions": self._decays, "_decay_rate": dr,
            "_p_shells": p_sh, "_n_shells": n_sh,
            "_history": self._history,
        }

        cls   = classify(raw)
        gpos  = self.config.gradient_position()
        be    = self.config.bethe_weizsacker()
        decay = self.config.predicted_decay_mode(cls)
        sc    = self.config.shell_closure_note()

        if   cls in ("I", "IV") and be > 6.0: be_conf = "high"
        elif cls == "II"        and be > 5.0: be_conf = "high"
        elif cls == "III":                    be_conf = "theoretical"
        else:                                 be_conf = "moderate"

        dyn_stability = 1.0 - var
        grad_expected = (gpos["deviation_score"] + 1.0) / 2.0
        anomaly = round(dyn_stability - grad_expected, 3)

        cfg = self.config
        sn    = cfg.separation_energy_n()
        sp    = cfg.separation_energy_p()
        s2n   = cfg.separation_energy_2n()
        s2p   = cfg.separation_energy_2p()
        q_a   = cfg.alpha_q_value()
        q_bm  = cfg.Q_beta_minus()
        q_bp  = cfg.Q_beta_plus()
        q_2b  = cfg.Q_2beta()
        drip  = cfg.drip_line_status()
        beta2 = cfg.deformation_beta2()
        r_proc = cfg.r_process_candidate()
        fiss  = cfg.fissility()
        iso   = cfg.isobar_competition()
        sg_n  = cfg.shell_gap_n()
        sg_p  = cfg.shell_gap_p()
        pg_n  = cfg.pairing_gap_n()
        pg_p  = cfg.pairing_gap_p()
        r_fm  = cfg.nuclear_radius()
        nskin = cfg.neutron_skin()
        hl    = cfg.estimated_halflife_log10()
        jp    = cfg.spin_parity_estimate()

        # --- Discrepancy detection: dynamics vs isobar competition ---
        discrepancy = False
        disc_note   = ""
        if cls in ("I", "II") and not iso["isobar_stable"]:
            discrepancy = True
            losers = []
            if iso["isobar_delta_minus"] > 0:
                losers.append(f"beta- by {iso['isobar_delta_minus']:+.2f} MeV")
            if iso["isobar_delta_plus"] > 0:
                losers.append(f"EC/beta+ by {iso['isobar_delta_plus']:+.2f} MeV")
            disc_note = (f"JANIS dynamics: {CLASSIFICATION_LABELS[cls]} | "
                         f"Isobar analysis: loses {', '.join(losers)}")

        raw.update({
            "classification":        cls,
            "classification_label":  CLASSIFICATION_LABELS[cls],
            "halflife_tier":         HALFLIFE_TIERS[cls],
            "predicted_decay":       decay,
            "binding_energy_mev":    be,
            "be_confidence":         be_conf,
            "pairing":               cfg.pairing_type(),
            "shell_closure":         sc,
            "active_proton_shells":  p_sh,
            "active_neutron_shells": n_sh,
            "valley_displacement":   gpos["valley_displacement"],
            "shell_proximity":       gpos["shell_proximity"],
            "shell_proximity_Z":     gpos["shell_proximity_Z"],
            "shell_proximity_N":     gpos["shell_proximity_N"],
            "binding_deviation_mev": gpos["binding_deviation_mev"],
            "deviation_score":       gpos["deviation_score"],
            "gradient_anomaly":      anomaly,
            "nz_ratio":              round(cfg.N / max(cfg.Z, 1), 4),
            "Sn":                    sn,
            "Sp":                    sp,
            "S2n":                   s2n,
            "S2p":                   s2p,
            "alpha_q":               q_a,
            "Q_beta_minus":          q_bm,
            "Q_beta_plus":           q_bp,
            "Q_2beta":               q_2b,
            "drip_line":             drip,
            "deformation_beta2":     beta2,
            "r_process":             r_proc,
            "fissility":             fiss,
            "isobar_stable":         iso["isobar_stable"],
            "isobar_delta_minus":    iso["isobar_delta_minus"],
            "isobar_delta_plus":     iso["isobar_delta_plus"],
            "shell_gap_n":           sg_n,
            "shell_gap_p":           sg_p,
            "pairing_gap_n":         pg_n,
            "pairing_gap_p":         pg_p,
            "nuclear_radius_fm":     r_fm,
            "neutron_skin_fm":       nskin,
            "halflife_log10s":       hl,
            "spin_parity":           jp,
            "discrepancy":           discrepancy,
            "discrepancy_note":      disc_note,
        })
        return raw


def solve(config: NuclearConfig, n_steps: int = 400, verbose: bool = False) -> Dict:
    return NuclearDynamicsEngine(config).solve(n_steps=n_steps, verbose=verbose)


# ─────────────────────────────────────────────────────────────────────────────
# Isotope sets
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_STABLE = [
    NuclearConfig(1,   0,  "¹H"),
    NuclearConfig(1,   1,  "²H"),
    NuclearConfig(2,   1,  "³He"),
    NuclearConfig(2,   2,  "⁴He"),
    NuclearConfig(3,   3,  "⁶Li"),
    NuclearConfig(3,   4,  "⁷Li"),
    NuclearConfig(4,   5,  "⁹Be"),
    NuclearConfig(5,   5,  "¹⁰B"),
    NuclearConfig(5,   6,  "¹¹B"),
    NuclearConfig(6,   6,  "¹²C"),
    NuclearConfig(6,   7,  "¹³C"),
    NuclearConfig(7,   7,  "¹⁴N"),
    NuclearConfig(7,   8,  "¹⁵N"),
    NuclearConfig(8,   8,  "¹⁶O"),
    NuclearConfig(8,   9,  "¹⁷O"),
    NuclearConfig(8,  10,  "¹⁸O"),
    NuclearConfig(20, 20,  "⁴⁰Ca  — doubly magic"),
    NuclearConfig(20, 28,  "⁴⁸Ca  — doubly magic"),
    NuclearConfig(26, 30,  "⁵⁶Fe  — iron peak"),
    NuclearConfig(28, 28,  "⁵⁶Ni  — doubly magic"),
    NuclearConfig(28, 30,  "⁵⁸Ni"),
    NuclearConfig(50, 64,  "¹¹⁴Sn"),
    NuclearConfig(50, 82,  "¹³²Sn — doubly magic"),
    NuclearConfig(82, 124, "²⁰⁶Pb"),
    NuclearConfig(82, 125, "²⁰⁷Pb"),
    NuclearConfig(82, 126, "²⁰⁸Pb — doubly magic"),
]

REFERENCE_RADIOACTIVE = [
    NuclearConfig(1,   2,  "³H   — tritium"),
    NuclearConfig(2,   4,  "⁶He  — beta emitter"),
    NuclearConfig(4,   3,  "⁷Be  — electron capture"),
    NuclearConfig(6,   2,  "⁸C   — proton-rich"),
    NuclearConfig(43, 56,  "⁹⁹Tc  — no stable isotope"),
    NuclearConfig(61, 84,  "¹⁴⁵Pm — no stable isotope"),
    NuclearConfig(92, 143, "²³⁵U  — fissile"),
    NuclearConfig(92, 146, "²³⁸U  — fertile"),
    NuclearConfig(94, 144, "²³⁸Pu — heat source"),
]

NEUTRON_RICH = [
    NuclearConfig(2,   6,  "⁸He   — known unstable"),
    NuclearConfig(8,  16,  "²⁴O   — beyond n drip line"),
    NuclearConfig(14, 28,  "⁴²Si  — exotic neutron-rich"),
    NuclearConfig(20, 40,  "⁶⁰Ca  — beyond drip line"),
    NuclearConfig(28, 50,  "⁷⁸Ni  — doubly magic candidate"),
    NuclearConfig(36, 60,  "⁹⁶Kr  — neutron-rich"),
    NuclearConfig(50, 100, "¹⁵⁰Sn — neutron-rich, post-magic"),
]

SUPERHEAVY = [
    NuclearConfig(82,  184, "²⁶⁶Pb  — Z=82  N=184"),
    NuclearConfig(108, 162, "²⁷⁰Hs"),
    NuclearConfig(110, 171, "²⁸¹Ds"),
    NuclearConfig(112, 161, "²⁷³Cn"),
    NuclearConfig(113, 169, "²⁸²Nh"),
    NuclearConfig(114, 161, "²⁷⁵Fl"),
    NuclearConfig(114, 184, "²⁹⁸Fl  — theoretical island peak"),
    NuclearConfig(115, 173, "²⁸⁸Mc"),
    NuclearConfig(115, 193, "³⁰⁸Mc  — neutron-rich candidate"),
    NuclearConfig(116, 177, "²⁹³Lv"),
    NuclearConfig(117, 176, "²⁹³Ts"),
    NuclearConfig(117, 186, "³⁰³Ts  — neutron-rich candidate"),
    NuclearConfig(118, 176, "²⁹⁴Og  — heaviest confirmed"),
    NuclearConfig(118, 180, "²⁹⁸Og  — island candidate"),
    NuclearConfig(119, 184, "³⁰³E119 — undiscovered  Z=119  N=184"),
    NuclearConfig(120, 184, "³⁰⁴E120 — undiscovered  Z=120  N=184"),
    NuclearConfig(121, 184, "³⁰⁵E121 — undiscovered  Z=121  N=184"),
    NuclearConfig(122, 184, "³⁰⁶E122 — undiscovered  Z=122  N=184"),
    NuclearConfig(126, 184, "³¹⁰E126 — predicted doubly magic"),
]

ALL_CONFIGS = REFERENCE_STABLE + REFERENCE_RADIOACTIVE + NEUTRON_RICH + SUPERHEAVY


# ─────────────────────────────────────────────────────────────────────────────
# Terminal output
# ─────────────────────────────────────────────────────────────────────────────

def _col(cls: str) -> str:
    return CLASSIFICATION_COLORS_ANSI.get(cls, "")

def _bar(v: float, width: int = 12) -> str:
    filled = int(round(v * width))
    return "#" * filled + "-" * (width - filled)

def print_result(r: Dict, verbose: bool = False):
    cfg = r["config"]; cls = r["classification"]
    sc    = f"  [{r['shell_closure']}]" if r["shell_closure"] else ""
    anom  = r["gradient_anomaly"]
    anom_s = f"  (A){anom:+.3f}" if anom > 0.05 else ""
    sync_b = _bar(r["_sync"], 8)
    print(
        f"  {_col(cls)}[{cls}]{RESET} {cfg.symbol:<40} | "
        f"{r['classification_label']:<22} | "
        f"BE={r['binding_energy_mev']:5.2f} MeV/u | "
        f"p-n [{sync_b}] {r['_sync']:.3f} | "
        f"Dv={r['valley_displacement']:+6.1f}  sh={r['shell_proximity']:.2f} | "
        f"decay: {r['predicted_decay']:<18}"
        f"{sc}{anom_s}"
    )
    if verbose:
        print(f"         Half-life tier:        {r['halflife_tier']}")
        print(f"         Pairing:               {r['pairing']}")
        print(f"         BE deviation:          {r['binding_deviation_mev']:+.3f} MeV/u from valley")
        print(f"         Deviation score:       {r['deviation_score']:+.3f}")
        print(f"         Gradient anomaly:      {r['gradient_anomaly']:+.3f}")
        print(f"         Decay rate signal:     {r['_decay_rate']:.5f}")
        print(f"         Freq coherence:        {r['_fc']:+.4f}")
        print(f"         Ground state var:      {r['_var']:.5f}")
        print(f"         Active proton shells:  {', '.join(r['active_proton_shells'])}")
        print(f"         Active neutron shells: {', '.join(r['active_neutron_shells'])}")
        print(f"         Shell closure:         {r['shell_closure'] or '—'}")
        print(f"         N/Z ratio:             {r['nz_ratio']:.4f}")
        print(f"         Spin-parity (est):     {r.get('spin_parity','?')}")
        print(f"         Nuclear radius:        {r.get('nuclear_radius_fm',0):.3f} fm")
        print(f"         Neutron skin:          {r.get('neutron_skin_fm',0):+.3f} fm")
        print(f"       --- Separation energies ---")
        print(f"         Sn (1n sep):           {r.get('Sn',0):+.3f} MeV")
        print(f"         Sp (1p sep):           {r.get('Sp',0):+.3f} MeV")
        print(f"         S2n (2n sep):          {r.get('S2n',0):+.3f} MeV")
        print(f"         S2p (2p sep):          {r.get('S2p',0):+.3f} MeV")
        print(f"       --- Q-values ---")
        print(f"         Alpha Q-value:         {r.get('alpha_q',0):+.3f} MeV")
        print(f"         Q(beta-):              {r.get('Q_beta_minus',0):+.3f} MeV")
        print(f"         Q(beta+/EC):           {r.get('Q_beta_plus',0):+.3f} MeV")
        print(f"         Q(2beta):              {r.get('Q_2beta',0):+.3f} MeV")
        print(f"       --- Shell structure ---")
        print(f"         Shell gap (n):         {r.get('shell_gap_n',0):+.3f} MeV")
        print(f"         Shell gap (p):         {r.get('shell_gap_p',0):+.3f} MeV")
        print(f"         Pairing gap (n):       {r.get('pairing_gap_n',0):+.3f} MeV")
        print(f"         Pairing gap (p):       {r.get('pairing_gap_p',0):+.3f} MeV")
        print(f"       --- Stability analysis ---")
        print(f"         Drip line:             {r.get('drip_line','—')}")
        print(f"         Deformation beta2:     {r.get('deformation_beta2',0):.3f}")
        print(f"         Fissility:             {r.get('fissility',0):.4f}")
        iso_s = "YES" if r.get('isobar_stable') else "NO"
        print(f"         Isobar stable:         {iso_s}  (d-={r.get('isobar_delta_minus',0):+.3f}  d+={r.get('isobar_delta_plus',0):+.3f})")
        hl = r.get('halflife_log10s')
        hl_s = f"10^{hl:.1f} s" if hl is not None else "Stable"
        print(f"         Est. half-life:        {hl_s}")
        rp = "YES" if r.get('r_process') else "no"
        print(f"         r-process candidate:   {rp}")
        if r.get('discrepancy'):
            print(f"       --- DISCREPANCY ---")
            print(f"         {r['discrepancy_note']}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotly dashboard
# ─────────────────────────────────────────────────────────────────────────────

def build_plotly_dashboard(results: List[Dict], output_path: str = "nuclear_survey.html"):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly not available — skipping plots")
        return

    CLS_HEX = CLASSIFICATION_COLORS_HEX
    BG       = "#0d1117"
    GRID     = "#1e2530"
    TEXT     = "#c9d1d9"
    ACCENT   = "#58a6ff"

    def hover_text(r: Dict) -> str:
        cfg = r["config"]
        flags = []
        if r.get("r_process"):                flags.append("r-proc")
        if r.get("drip_line") != "bound":     flags.append(r["drip_line"])
        if r.get("discrepancy"):              flags.append("DISCREPANCY")
        if not r.get("isobar_stable", True):  flags.append("isobar-unstable")
        flag_str = f"<br><b>⚑ {', '.join(flags)}</b>" if flags else ""
        hl = r.get("halflife_log10s")
        hl_s = f"10^{hl:.1f} s" if hl is not None else "Stable"
        iso_s = "yes" if r.get("isobar_stable", True) else "NO"
        return (
            f"<b>{cfg.symbol}</b>  J={r.get('spin_parity','?')}<br>"
            f"Z={cfg.Z}  N={cfg.N}  A={cfg.A}  R={r.get('nuclear_radius_fm',0):.2f} fm<br>"
            f"Class: {r['classification']} — {r['classification_label']}<br>"
            f"BE: {r['binding_energy_mev']:.3f} MeV/u  β2≈{r.get('deformation_beta2',0):.3f}<br>"
            f"Decay: {r['predicted_decay']}  t½≈{hl_s}<br>"
            f"Sn={r.get('Sn',0):+.2f}  Sp={r.get('Sp',0):+.2f} MeV<br>"
            f"Qα={r.get('alpha_q',0):+.2f}  Qβ-={r.get('Q_beta_minus',0):+.2f}  "
            f"Qβ+={r.get('Q_beta_plus',0):+.2f}<br>"
            f"Isobar stable: {iso_s}  fiss={r.get('fissility',0):.3f}<br>"
            f"ΔSn={r.get('shell_gap_n',0):+.2f}  ΔSp={r.get('shell_gap_p',0):+.2f}<br>"
            f"p-n: {r['_sync']:.3f}  var: {r['_var']:.4f}<br>"
            f"Δvalley: {r['valley_displacement']:+.1f}  shell: {r['shell_proximity']:.2f}<br>"
            f"N/Z: {r['nz_ratio']:.3f}  {r['shell_closure'] or ''}"
            f"{flag_str}"
        )

    layout_base = dict(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family="monospace", size=11),
        margin=dict(l=60, r=30, t=50, b=50),
    )

    def axis_style(**kw):
        base = dict(gridcolor=GRID, zerolinecolor=GRID,
                    tickfont=dict(color=TEXT))
        if "tickfont" in kw:
            base["tickfont"] = {**base["tickfont"], **kw.pop("tickfont")}
        base.update(kw)
        return base

    figs_html = []

    # ── 1. Segré nuclear chart ─────────────────────────────────────────────
    fig1 = go.Figure()

    # Valley centerline
    z_line = list(range(1, 127))
    n_line = [valley_centerline_N(z) for z in z_line]
    fig1.add_trace(go.Scatter(
        x=n_line, y=z_line, mode="lines",
        line=dict(color="rgba(255, 255, 255, 0.2)", width=1, dash="dot"),
        name="Valley centerline", hoverinfo="skip",
    ))

    # Magic number grid lines
    for m in MAGIC_NUMBERS:
        fig1.add_vline(x=m, line=dict(color="rgba(255, 255, 255, 0.09)", width=1))
        fig1.add_hline(y=m, line=dict(color="rgba(255, 255, 255, 0.09)", width=1))

    # Isotopes by classification
    for cls in ["V", "IV", "III", "II", "I"]:
        subset = [r for r in results if r["classification"] == cls]
        if not subset: continue
        fig1.add_trace(go.Scatter(
            x=[r["config"].N for r in subset],
            y=[r["config"].Z for r in subset],
            mode="markers",
            name=f"{cls} — {CLASSIFICATION_LABELS[cls]}",
            marker=dict(
                color=CLS_HEX[cls],
                size=[max(6, r["binding_energy_mev"] * 1.2) for r in subset],
                opacity=0.88,
                line=dict(width=0.5, color="rgba(0, 0, 0, 0.37)"),
                symbol=["star" if r["config"].is_doubly_magic else "circle" for r in subset],
            ),
            text=[hover_text(r) for r in subset],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig1.update_layout(
        **layout_base,
        title=dict(text="Nuclear Chart — Segré  (★ doubly magic)", font=dict(size=15, color=ACCENT)),
        xaxis=axis_style(title="Neutron number N"),
        yaxis=axis_style(title="Proton number Z"),
        legend=dict(bgcolor=BG, bordercolor=GRID, borderwidth=1),
        height=600,
    )

    # ── 2. Binding energy vs A ────────────────────────────────────────────
    fig2 = go.Figure()

    # Smooth valley baseline
    a_vals  = list(range(1, max(r["config"].A for r in results) + 1))
    be_base = [valley_BE_expected(a) for a in a_vals]
    fig2.add_trace(go.Scatter(
        x=a_vals, y=be_base, mode="lines",
        line=dict(color="rgba(255, 255, 255, 0.15)", width=1.5, dash="dot"),
        name="Valley baseline (BW smooth)", hoverinfo="skip",
    ))

    # Magic number verticals on A-axis (approximate positions)
    for m in MAGIC_NUMBERS:
        for a_est in [m * 2, m * 2 + m // 2]:
            if a_est <= max(a_vals):
                fig2.add_vline(x=a_est, line=dict(color="rgba(255, 255, 255, 0.06)", width=1))

    for cls in ["V", "IV", "III", "II", "I"]:
        subset = sorted(
            [r for r in results if r["classification"] == cls],
            key=lambda r: r["config"].A)
        if not subset: continue
        fig2.add_trace(go.Scatter(
            x=[r["config"].A for r in subset],
            y=[r["binding_energy_mev"] for r in subset],
            mode="markers",
            name=f"{cls}",
            marker=dict(
                color=CLS_HEX[cls], size=9, opacity=0.9,
                symbol=["star" if r["config"].is_doubly_magic else "circle" for r in subset],
            ),
            text=[hover_text(r) for r in subset],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig2.update_layout(
        **layout_base,
        title=dict(text="Binding Energy per Nucleon vs Mass Number A", font=dict(size=15, color=ACCENT)),
        xaxis=axis_style(title="Mass number A"),
        yaxis=axis_style(title="BE / nucleon (MeV/u)", range=[0, 9.2]),
        height=450,
    )

    # ── 3. Gradient anomaly + deviation score ─────────────────────────────
    fig3 = go.Figure()

    ranked = sorted(results, key=lambda r: r["gradient_anomaly"])
    labels = [r["config"].symbol[:30] for r in ranked]
    anomalies = [r["gradient_anomaly"] for r in ranked]
    colors_bar = [CLS_HEX[r["classification"]] for r in ranked]
    hover3 = [hover_text(r) for r in ranked]

    fig3.add_trace(go.Bar(
        y=labels, x=anomalies, orientation="h",
        marker=dict(color=colors_bar, opacity=0.85),
        text=[f"{a:+.3f}" for a in anomalies],
        textposition="outside",
        textfont=dict(size=9, color=TEXT),
        hovertext=hover3, hovertemplate="%{hovertext}<extra></extra>",
        name="Gradient anomaly",
    ))
    fig3.add_vline(x=0.05, line=dict(color=ACCENT, width=1, dash="dash"))
    fig3.update_layout(
        **layout_base,
        title=dict(text="Gradient Anomaly — Dynamics vs Smooth Prediction", font=dict(size=15, color=ACCENT)),
        xaxis=axis_style(title="Anomaly (dynamics − gradient prediction)", zeroline=True, zerolinewidth=1),
        yaxis=axis_style(title="", tickfont=dict(size=9)),
        height=max(400, len(ranked) * 18),
        showlegend=False,
    )

    # ── 4. Shell proximity landscape (Z vs N, heat = shell prox) ─────────
    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(
        x=[r["config"].N for r in results],
        y=[r["config"].Z for r in results],
        mode="markers",
        marker=dict(
            color=[r["shell_proximity"] for r in results],
            colorscale=[
                [0.0, "#1a1f2e"], [0.4, "#2d4a7a"],
                [0.7, "#4a9eff"], [0.9, "#a0d4ff"], [1.0, "#ffffff"],
            ],
            size=12, opacity=0.9,
            colorbar=dict(
                title=dict(text="Shell proximity", font=dict(color=TEXT)),
                tickfont=dict(color=TEXT), bgcolor=BG,
            ),
            showscale=True,
        ),
        text=[hover_text(r) for r in results],
        hovertemplate="%{text}<extra></extra>",
        name="Shell proximity",
    ))

    z_line2 = list(range(1, 127))
    n_line2  = [valley_centerline_N(z) for z in z_line2]
    fig4.add_trace(go.Scatter(
        x=n_line2, y=z_line2, mode="lines",
        line=dict(color="rgba(255, 255, 255, 0.19)", width=1, dash="dot"),
        name="Valley centerline", hoverinfo="skip",
    ))

    fig4.update_layout(
        **layout_base,
        title=dict(text="Shell Proximity Landscape (Z vs N)", font=dict(size=15, color=ACCENT)),
        xaxis=axis_style(title="N"),
        yaxis=axis_style(title="Z"),
        height=550,
    )

    # ── 5. p-n alignment vs BE deviation (scatter) ───────────────────────
    fig5 = go.Figure()
    for cls in ["V", "IV", "III", "II", "I"]:
        subset = [r for r in results if r["classification"] == cls]
        if not subset: continue
        fig5.add_trace(go.Scatter(
            x=[r["_sync"] for r in subset],
            y=[r["binding_deviation_mev"] for r in subset],
            mode="markers",
            name=f"{cls} — {CLASSIFICATION_LABELS[cls]}",
            marker=dict(
                color=CLS_HEX[cls], size=10, opacity=0.85,
                symbol=["star" if r["config"].is_doubly_magic else "circle" for r in subset],
            ),
            text=[hover_text(r) for r in subset],
            hovertemplate="%{text}<extra></extra>",
        ))
    fig5.add_hline(y=0, line=dict(color="rgba(255, 255, 255, 0.12)", width=1))
    fig5.update_layout(
        **layout_base,
        title=dict(text="p-n Alignment vs Binding Deviation (★ doubly magic)", font=dict(size=15, color=ACCENT)),
        xaxis=axis_style(title="p-n alignment (sync)"),
        yaxis=axis_style(title="BE deviation from valley (MeV/u)"),
        height=450,
    )

    # ── 6. Decay Mode Map (Z vs N, colored by decay) ───────────────────
    fig6 = go.Figure()
    decay_colors = {
        "Stable": "#2ecc71", "α decay": "#e74c3c", "β⁻ decay": "#3498db",
        "β⁺ / EC": "#f39c12", "Spontaneous fission": "#9b59b6",
        "Proton emission": "#e67e22", "Neutron emission": "#1abc9c",
        "Unknown / theoretical": "#95a5a6",
    }
    for dmode, dcolor in decay_colors.items():
        subset = [r for r in results if r.get("predicted_decay") == dmode]
        if not subset: continue
        fig6.add_trace(go.Scatter(
            x=[r["config"].N for r in subset],
            y=[r["config"].Z for r in subset],
            mode="markers", name=dmode,
            marker=dict(color=dcolor, size=8, opacity=0.85),
            text=[hover_text(r) for r in subset],
            hovertemplate="%{text}<extra></extra>",
        ))
    for m in MAGIC_NUMBERS:
        fig6.add_vline(x=m, line=dict(color="rgba(255, 255, 255, 0.07)", width=1))
        fig6.add_hline(y=m, line=dict(color="rgba(255, 255, 255, 0.07)", width=1))
    fig6.update_layout(
        **layout_base,
        title=dict(text="Decay Mode Map (Z vs N)", font=dict(size=15, color=ACCENT)),
        xaxis=axis_style(title="N"), yaxis=axis_style(title="Z"),
        height=600,
    )

    # ── 7. Separation energy Sn vs N (isotopic chains) ───────────────
    fig7 = go.Figure()
    chain_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    z_chains = {}
    for r in results:
        z = r["config"].Z
        if z not in z_chains: z_chains[z] = []
        z_chains[z].append(r)
    # Pick representative chains with enough data points
    good_chains = sorted(
        [(z, rs) for z, rs in z_chains.items() if len(rs) >= 3],
        key=lambda x: -len(x[1]))[:6]
    for i, (z, rs) in enumerate(good_chains):
        rs_sorted = sorted(rs, key=lambda r: r["config"].N)
        fig7.add_trace(go.Scatter(
            x=[r["config"].N for r in rs_sorted],
            y=[r.get("Sn", 0) for r in rs_sorted],
            mode="lines+markers", name=f"Z={z}",
            line=dict(color=chain_colors[i % len(chain_colors)], width=2),
            marker=dict(size=5),
            text=[hover_text(r) for r in rs_sorted],
            hovertemplate="%{text}<extra></extra>",
        ))
    for m in MAGIC_NUMBERS:
        fig7.add_vline(x=m, line=dict(color="rgba(255, 255, 255, 0.1)", width=1, dash="dot"))
    fig7.add_hline(y=0, line=dict(color="#e74c3c", width=1, dash="dash"))
    fig7.update_layout(
        **layout_base,
        title=dict(text="Neutron Separation Energy Sn vs N (Isotopic Chains)", font=dict(size=15, color=ACCENT)),
        xaxis=axis_style(title="Neutron number N"),
        yaxis=axis_style(title="Sn (MeV)"),
        height=450,
    )

    # ── 8. Discrepancy table (dynamics vs isobar) ────────────────────
    disc_results = [r for r in results if r.get("discrepancy")]
    fig8 = None
    if disc_results:
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(
            x=[r["config"].N for r in disc_results],
            y=[r["config"].Z for r in disc_results],
            mode="markers+text",
            marker=dict(color="#ff6b6b", size=14, symbol="diamond",
                        line=dict(width=2, color="#ffffff")),
            text=[r["config"].symbol[:15] for r in disc_results],
            textposition="top center",
            textfont=dict(size=9, color="#ff6b6b"),
            hovertext=[hover_text(r) for r in disc_results],
            hovertemplate="%{hovertext}<extra></extra>",
            name="Discrepancy",
        ))
        # Add valley line for context
        z_line3 = list(range(1, 127))
        fig8.add_trace(go.Scatter(
            x=[valley_centerline_N(z) for z in z_line3], y=z_line3,
            mode="lines", line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
            name="Valley", hoverinfo="skip",
        ))
        fig8.update_layout(
            **layout_base,
            title=dict(text="Discrepancies — JANIS Dynamics vs Isobar Competition",
                       font=dict(size=15, color="#ff6b6b")),
            xaxis=axis_style(title="N"), yaxis=axis_style(title="Z"),
            height=400,
        )

    # ── Combine into single HTML ──────────────────────────────────────────
    import plotly.io as pio

    def fig_to_div(fig, first=False):
        return pio.to_html(fig, full_html=first, include_plotlyjs="cdn" if first else False,
                           config={"displayModeBar": True, "scrollZoom": True})

    header = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>JANIS² Nuclear — Survey</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: {BG}; color: {TEXT}; font-family: monospace; padding: 24px; }}
  h1 {{ color: {ACCENT}; font-size: 1.3em; margin-bottom: 4px; letter-spacing: 0.05em; }}
  .meta {{ color: #666; font-size: 0.8em; margin-bottom: 24px; }}
  .section {{ margin-bottom: 32px; }}
  .section-title {{ color: #8b949e; font-size: 0.75em; letter-spacing: 0.12em;
                    text-transform: uppercase; margin-bottom: 8px; padding-bottom: 4px;
                    border-bottom: 1px solid {GRID}; }}
</style>
</head>
<body>
<h1>JANIS² Nuclear — v4.1</h1>
<div class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {len(results)} isotopes | v4.1 +Sn/Sp/Q-values +isobar +shell-gaps +half-life +deformation</div>
<div class="section"><div class="section-title">01 — Segré Nuclear Chart</div>
"""

    sections = [
        ("02 — Binding Energy Curve",           fig2),
        ("03 — Gradient Anomaly Ranking",        fig3),
        ("04 — Shell Proximity Landscape",       fig4),
        ("05 — p-n Alignment vs BE Deviation",   fig5),
        ("06 — Decay Mode Map",                  fig6),
        ("07 — Neutron Separation Energy Chains", fig7),
    ]
    if fig8 is not None:
        sections.append(("08 — Discrepancies (Dynamics vs Isobar)", fig8))

    html = header + fig_to_div(fig1, first=True) + "</div>\n"
    for title, fig in sections:
        html += f'<div class="section"><div class="section-title">{title}</div>\n'
        html += fig_to_div(fig, first=False) + "</div>\n"
    html += "</body></html>"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  → {output_path}  ({os.path.getsize(output_path)//1024} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# CSV logging
# ─────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "timestamp", "group",
    "symbol", "Z", "N", "A", "nz_ratio", "pairing", "spin_parity",
    "classification", "classification_label", "halflife_tier",
    "binding_energy_mev", "be_confidence",
    "predicted_decay",
    "Sn", "Sp", "S2n", "S2p",
    "alpha_q", "Q_beta_minus", "Q_beta_plus", "Q_2beta",
    "drip_line", "deformation_beta2", "r_process", "fissility",
    "nuclear_radius_fm", "neutron_skin_fm",
    "isobar_stable", "isobar_delta_minus", "isobar_delta_plus",
    "shell_gap_n", "shell_gap_p", "pairing_gap_n", "pairing_gap_p",
    "halflife_log10s",
    "valley_displacement", "shell_proximity",
    "shell_proximity_Z", "shell_proximity_N",
    "binding_deviation_mev", "deviation_score", "gradient_anomaly",
    "shell_closure", "doubly_magic",
    "active_proton_shells", "active_neutron_shells",
    "discrepancy", "discrepancy_note",
    "_var", "_sync", "_fc", "_emissions", "_decay_rate",
    "_converged", "_conv_at",
]

def _csv_row(r: Dict, ts: str, group: str) -> list:
    cfg = r["config"]
    hl = r.get("halflife_log10s")
    return [
        ts, group,
        cfg.symbol, cfg.Z, cfg.N, cfg.A, r["nz_ratio"], r["pairing"],
        r.get("spin_parity", ""),
        r["classification"], r["classification_label"], r["halflife_tier"],
        r["binding_energy_mev"], r["be_confidence"],
        r["predicted_decay"],
        r.get("Sn", ""), r.get("Sp", ""), r.get("S2n", ""), r.get("S2p", ""),
        r.get("alpha_q", ""), r.get("Q_beta_minus", ""),
        r.get("Q_beta_plus", ""), r.get("Q_2beta", ""),
        r.get("drip_line", ""), r.get("deformation_beta2", ""),
        int(r.get("r_process", False)), r.get("fissility", ""),
        r.get("nuclear_radius_fm", ""), r.get("neutron_skin_fm", ""),
        int(r.get("isobar_stable", True)),
        r.get("isobar_delta_minus", ""), r.get("isobar_delta_plus", ""),
        r.get("shell_gap_n", ""), r.get("shell_gap_p", ""),
        r.get("pairing_gap_n", ""), r.get("pairing_gap_p", ""),
        hl if hl is not None else "",
        r["valley_displacement"], r["shell_proximity"],
        r["shell_proximity_Z"], r["shell_proximity_N"],
        r["binding_deviation_mev"], r["deviation_score"], r["gradient_anomaly"],
        r["shell_closure"], int(cfg.is_doubly_magic),
        "|".join(r["active_proton_shells"]),
        "|".join(r["active_neutron_shells"]),
        int(r.get("discrepancy", False)), r.get("discrepancy_note", ""),
        round(r["_var"], 6), round(r["_sync"], 5),
        round(r["_fc"],  5), r["_emissions"],
        round(r["_decay_rate"], 6),
        int(r["_converged"]), r["_conv_at"] if r["_conv_at"] is not None else "",
    ]


def results_to_csv(results: List[Dict], groups: List[str], path: str):
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for r, g in zip(results, groups):
            w.writerow(_csv_row(r, ts, g))
    print(f"  → {path}  ({len(results)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Autopilot — default mode
# ─────────────────────────────────────────────────────────────────────────────

def run_autopilot(n_steps: int = 400, skip_discovery: bool = False):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    W    = 148

    print("\n" + "=" * W)
    print("  JANIS² Nuclear — Joint Accelerated Nuclear Intelligence System  v4.1")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
          f"n_steps={n_steps}  |  +Sn/Sp/S2n/S2p +Qα +drip +β2 +r-proc +fissility")
    if skip_discovery:
        print("  MODE: Reference sets only (no full discovery sweep)")
    print("=" * W)

    header = (
        f"  {'Cls':<4} {'Isotope':<40}  {'Label':<22}  "
        f"{'BE':>6}  {'p-n align':<15}  {'Δvalley':>8}  {'shell':>5}  "
        f"{'N/Z':>6}  {'Decay':<20}  {'Notes'}"
    )
    rule = "  " + "-" * (W - 2)

    sets = [
        ("Known Stable",       REFERENCE_STABLE),
        ("Known Radioactive",  REFERENCE_RADIOACTIVE),
        ("Neutron-Rich",       NEUTRON_RICH),
        ("Superheavy / SHE",   SUPERHEAVY),
    ]

    all_results: List[Dict] = []
    all_groups:  List[str]  = []
    cls_counts = {k: 0 for k in CLASSIFICATION_LABELS}

    for group_name, configs in sets:
        print(f"\n-- {group_name} " + "-" * (W - len(group_name) - 5))
        print(header)
        print(rule)
        for cfg in configs:
            r = solve(cfg, n_steps=n_steps)
            cls = r["classification"]
            sc  = f"[{r['shell_closure']}]" if r["shell_closure"] else ""
            anom = r["gradient_anomaly"]
            anom_s = f"(A){anom:+.3f}" if anom > 0.05 else ""
            sync_b = _bar(r["_sync"], 8)
            print(
                f"  {_col(cls)}[{cls}]{RESET} {cfg.symbol:<40}  "
                f"{r['classification_label']:<22}  "
                f"{r['binding_energy_mev']:>5.3f}  "
                f"[{sync_b}]{r['_sync']:.3f}  "
                f"{r['valley_displacement']:>+8.1f}  "
                f"{r['shell_proximity']:>5.2f}  "
                f"{r['nz_ratio']:>6.3f}  "
                f"{r['predicted_decay']:<20}  "
                f"{sc} {anom_s}"
            )
            cls_counts[cls] += 1
            all_results.append(r)
            all_groups.append(group_name)

    # -- Massive Discovery Run (optional) ─────────────────────────────────────
    if not skip_discovery:
        print(f"\n-- MASSIVE ISOTOPE DISCOVERY RUN (Z=1-120, N=1-184) " + "-" * (W - 55))
        discovery_csv = f"nuclear_discovery_{ts}.csv"
        discovery_results = run_sweep(range(1, 121), range(1, 185), n_steps=n_steps,
                                      output_file=discovery_csv, label="discovery")

        for r in discovery_results:
            all_results.append(r)
            all_groups.append("Discovery Sweep")
            cls_counts[r["classification"]] += 1
    else:
        discovery_csv = None

    # -- Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'-' * W}")
    print(f"\n  Classification distribution:")
    for k, v in CLASSIFICATION_LABELS.items():
        bar = "#" * min(cls_counts[k], 80)
        pct = cls_counts[k] / max(sum(cls_counts.values()), 1) * 100
        print(f"    {_col(k)}[{k}]{RESET}  {v:<26}  {cls_counts[k]:5d}  ({pct:5.1f}%)  {bar}")

    # Gradient anomalies
    anomalies = sorted(
        [r for r in all_results if r["gradient_anomaly"] > 0.05],
        key=lambda x: -x["gradient_anomaly"])
    if anomalies:
        print(f"\n  ★ GRADIENT ANOMALIES (dynamics exceed smooth prediction by >0.05):")
        print(f"  {'Cls':<4} {'Isotope':<42} {'anomaly':>9} {'BE dev':>8} {'shell':>6} {'Dv':>8} {'N/Z':>6}")
        print("  " + "─" * 85)
        for r in anomalies[:30]:  # Top 30
            cfg = r["config"]
            print(f"  {_col(r['classification'])}[{r['classification']}]{RESET}  "
                  f"{cfg.symbol:<40}  {r['gradient_anomaly']:>+9.3f}  "
                  f"{r['binding_deviation_mev']:>+7.3f}  "
                  f"{r['shell_proximity']:>6.2f}  "
                  f"{r['valley_displacement']:>+8.1f}  "
                  f"{r['nz_ratio']:>6.4f}")
        if len(anomalies) > 30:
            print(f"  ... and {len(anomalies)-30} more")

    # Island candidates (high anomaly + high shell proximity + near N=184)
    island_candidates = [
        r for r in all_results 
        if r["gradient_anomaly"] > 0.03 
        and r["shell_proximity"] > 0.7
        and abs(r["config"].N - 184) < 20
        and r["config"].Z > 100
    ]
    if island_candidates:
        print(f"\n  🏝️ ISLAND OF STABILITY CANDIDATES:")
        print(f"  {'Cls':<4} {'Isotope':<25} {'Z':<4} {'N':<4} {'anomaly':>8} {'shell prox':>10} {'decay':<18} {'Sn':>8}")
        print("  " + "─" * 85)
        for r in sorted(island_candidates, key=lambda x: -x["gradient_anomaly"])[:20]:
            cfg = r["config"]
            print(f"  {_col(r['classification'])}[{r['classification']}]{RESET}  "
                  f"{cfg.symbol:<23}  {cfg.Z:<4} {cfg.N:<4}  "
                  f"{r['gradient_anomaly']:>+8.3f}  "
                  f"{r['shell_proximity']:>10.3f}  "
                  f"{r['predicted_decay']:<18}  "
                  f"{r.get('Sn',0):>+8.3f}")

    # Discrepancies (dynamics vs isobar competition)
    discrepancies = [r for r in all_results if r.get("discrepancy")]
    if discrepancies:
        print(f"\n  ⚠️ DISCREPANCIES (JANIS dynamics vs isobar competition):")
        print(f"  {'Cls':<4} {'Isotope':<30} {'d- (MeV)':>10} {'d+ (MeV)':>10} {'Note':<40}")
        print("  " + "─" * 85)
        for r in discrepancies[:15]:
            cfg = r["config"]
            print(f"  {_col(r['classification'])}[{r['classification']}]{RESET}  "
                  f"{cfg.symbol:<28}  "
                  f"{r.get('isobar_delta_minus',0):>+10.3f}  "
                  f"{r.get('isobar_delta_plus',0):>+10.3f}  "
                  f"{r['discrepancy_note'][:40]}")

    # Doubly magic check
    doubly_magic = [r for r in all_results if r["config"].is_doubly_magic]
    if doubly_magic:
        print(f"\n  ✨ DOUBLY MAGIC NUCLEI:")
        for r in sorted(doubly_magic, key=lambda x: x["config"].A):
            cfg = r["config"]
            print(f"    {cfg.symbol:<15}  Z={cfg.Z:3d} N={cfg.N:3d}  "
                  f"Class: {r['classification']}  BE={r['binding_energy_mev']:.3f}  "
                  f"Decay: {r['predicted_decay']}")

    # ── Outputs ──────────────────────────────────────────────────────────────
    print(f"\n  Output files:")
    csv_path  = f"nuclear_survey_{ts}.csv"
    html_path = f"nuclear_survey_{ts}.html"
    print(f"    Full survey CSV:  {csv_path}")
    if not skip_discovery:
        print(f"    Discovery CSV:    {discovery_csv}")
    print(f"    Plotly Dashboard: {html_path}")
    results_to_csv(all_results, all_groups, csv_path)
    build_plotly_dashboard(all_results, html_path)
    print(f"\n{'=' * W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Sweep runners
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(z_range, n_range, n_steps=3000, output_file="landscape.csv", label="sweep"):
    total = sum(1 for Z in z_range for N in n_range)
    done  = 0; t0 = time.time(); results = []
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for Z in z_range:
            for N in n_range:
                try:
                    r = solve(NuclearConfig(Z, N), n_steps=n_steps)
                    results.append(r)
                    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                    w.writerow(_csv_row(r, ts, label))
                except Exception as e:
                    print(f"  Error Z={Z} N={N}: {e}")
                done += 1
                if done % 100 == 0:
                    rate = done / (time.time() - t0)
                    print(f"  {label}: {done}/{total} | Z={Z} N={N} | "
                          f"{rate:.1f}/s | ETA {(total-done)/max(rate,1e-8):.0f}s")
    print(f"\n{label} complete — {output_file}")
    return results


def run_gradient_report(n_steps: int = 800):
    print("\n" + "=" * 120)
    print("  NUCLEAR SHELL DYNAMICS — Gradient Anomaly Report")
    print("=" * 120)
    results = [solve(cfg, n_steps=n_steps) for cfg in ALL_CONFIGS]
    results.sort(key=lambda x: -x["gradient_anomaly"])
    
    # Split into tiers for cleaner display
    high_anomaly = [r for r in results if r["gradient_anomaly"] > 0.05]
    mid_anomaly = [r for r in results if 0.01 < r["gradient_anomaly"] <= 0.05]
    low_anomaly = [r for r in results if r["gradient_anomaly"] <= 0.01]
    
    print(f"\n  {'★ HIGH ANOMALY (dynamics > smooth by >0.05)':<80} n={len(high_anomaly)}")
    if high_anomaly:
        print(f"  {'Cls':<4} {'Isotope':<45} {'anomaly':>9} {'BE dev':>8} "
              f"{'shell':>6} {'Δvalley':>8} {'N/Z':>6} {'pairing':>10}")
        print("  " + "─" * 110)
        for r in high_anomaly:
            cfg = r["config"]
            print(f"  {_col(r['classification'])}[{r['classification']}]{RESET}  "
                  f"{cfg.symbol:<43}  {r['gradient_anomaly']:>+9.3f}  "
                  f"{r['binding_deviation_mev']:>+7.3f}  "
                  f"{r['shell_proximity']:>6.2f}  "
                  f"{r['valley_displacement']:>+8.1f}  "
                  f"{r['nz_ratio']:>6.4f}  "
                  f"{r['pairing'][:10]:>10}")
    
    print(f"\n  {'● MID ANOMALY (0.01 to 0.05)':<80} n={len(mid_anomaly)}")
    if mid_anomaly:
        print(f"  {'Cls':<4} {'Isotope':<45} {'anomaly':>9} {'BE dev':>8} "
              f"{'shell':>6} {'Δvalley':>8} {'N/Z':>6}")
        print("  " + "─" * 100)
        for r in mid_anomaly[:20]:  # Top 20 mid anomalies
            cfg = r["config"]
            print(f"  {_col(r['classification'])}[{r['classification']}]{RESET}  "
                  f"{cfg.symbol:<43}  {r['gradient_anomaly']:>+9.3f}  "
                  f"{r['binding_deviation_mev']:>+7.3f}  "
                  f"{r['shell_proximity']:>6.2f}  "
                  f"{r['valley_displacement']:>+8.1f}  "
                  f"{r['nz_ratio']:>6.4f}")
        if len(mid_anomaly) > 20:
            print(f"  ... and {len(mid_anomaly)-20} more")
    
    print(f"\n  {'○ LOW ANOMALY (<=0.01)':<80} n={len(low_anomaly)}")
    
    # Summary statistics
    print("\n" + "=" * 120)
    print("  SUMMARY STATISTICS")
    print("=" * 120)
    print(f"  Total isotopes analyzed:     {len(results)}")
    print(f"  High anomaly (>0.05):        {len(high_anomaly)} ({len(high_anomaly)/len(results)*100:.1f}%)")
    print(f"  Mid anomaly (0.01-0.05):     {len(mid_anomaly)} ({len(mid_anomaly)/len(results)*100:.1f}%)")
    print(f"  Low anomaly (≤0.01):         {len(low_anomaly)} ({len(low_anomaly)/len(results)*100:.1f}%)")
    
    # Classification breakdown within anomalies
    if high_anomaly:
        print("\n  High anomaly by classification:")
        cls_count = {}
        for r in high_anomaly:
            cls = r["classification"]
            cls_count[cls] = cls_count.get(cls, 0) + 1
        for cls, cnt in sorted(cls_count.items()):
            print(f"    {_col(cls)}[{cls}]{RESET}  {CLASSIFICATION_LABELS.get(cls, '?'):<26}  {cnt:3d}")
    
    # Island candidates (high anomaly + high shell proximity + near N=184)
    island_candidates = [
        r for r in results 
        if r["gradient_anomaly"] > 0.03 
        and r["shell_proximity"] > 0.7
        and abs(r["config"].N - 184) < 15
        and r["config"].Z > 100
    ]
    if island_candidates:
        print("\n  🏝️ ISLAND OF STABILITY CANDIDATES (high anomaly + shell proximity + near N=184):")
        print(f"  {'Cls':<4} {'Isotope':<25} {'Z':<4} {'N':<4} {'anomaly':>8} {'shell prox':>10} {'decay':<18}")
        print("  " + "─" * 85)
        for r in sorted(island_candidates, key=lambda x: -x["gradient_anomaly"])[:15]:
            cfg = r["config"]
            print(f"  {_col(r['classification'])}[{r['classification']}]{RESET}  "
                  f"{cfg.symbol:<23}  {cfg.Z:<4} {cfg.N:<4}  "
                  f"{r['gradient_anomaly']:>+8.3f}  "
                  f"{r['shell_proximity']:>10.3f}  "
                  f"{r['predicted_decay']:<18}")
    
    print("=" * 120 + "\n")


def analyze_csv(path: str, top_n: int = 40):
    from collections import Counter
    with open(path, encoding="utf-8") as f: rows = list(csv.DictReader(f))
    print(f"\nAnalysis: {path}  ({len(rows)} configurations)")
    cc = Counter(r.get("classification", "?") for r in rows)
    print("\n  Classification breakdown:")
    for k in sorted(cc):
        print(f"    {_col(k)}[{k}]{RESET}  {CLASSIFICATION_LABELS.get(k,'?'):<26}  {cc[k]:5d}")
    try:
        rows.sort(key=lambda r: -float(r.get("gradient_anomaly", 0)))
        anomalies = [r for r in rows if float(r.get("gradient_anomaly", 0)) > 0.05]
        print(f"\n  Top gradient anomalies:")
        print(f"  {'Cls':>3} {'Z':>4} {'N':>4} {'A':>5} {'anomaly':>9} {'BE dev':>8} {'shell':>6} {'N/Z':>6}")
        for r in anomalies[:top_n]:
            print(f"  {_col(r.get('classification','?'))}[{r.get('classification','?'):>1}]{RESET} "
                  f"{r['Z']:>4} {r['N']:>4} {int(r['Z'])+int(r['N']):>5} "
                  f"{float(r.get('gradient_anomaly',0)):>+9.3f}  "
                  f"{float(r.get('binding_deviation_mev',0)):>+7.3f}  "
                  f"{float(r.get('shell_proximity',0)):>6.2f}  "
                  f"{float(r.get('nz_ratio',0)):>6.4f}")
    except Exception as e:
        print(f"  (analysis failed: {e})")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        run_autopilot(n_steps=400)

    elif args[0] == "single":
        Z = int(args[1]); N = int(args[2])
        label = args[3] if len(args) > 3 else f"Z={Z} N={N} A={Z+N}"
        r = solve(NuclearConfig(Z, N, label), n_steps=800, verbose=True)
        print(); print_result(r, verbose=True); print()

    elif args[0] == "gradient":
        run_gradient_report(n_steps=600)

    elif args[0] == "sweep":
        run_sweep(range(1, 83), range(1, 131), n_steps=400,
                  output_file="stability_valley.csv", label="valley")
        analyze_csv("stability_valley.csv")

    elif args[0] == "island":
        run_sweep(range(100, 127), range(150, 196), n_steps=500,
                  output_file="island_sweep.csv", label="island")
        analyze_csv("island_sweep.csv")

    elif args[0] == "island_highres":
        Z_range = list(range(110, 127))
        N_range = list(range(170, 196))
        print(f"\n  HIGH-RES ISLAND SCAN: Z={Z_range[0]}-{Z_range[-1]} ({len(Z_range)}), "
              f"N={N_range[0]}-{N_range[-1]} ({len(N_range)}) = {len(Z_range)*len(N_range)} isotopes")
        run_sweep(Z_range, N_range, n_steps=800,
                  output_file="island_highres.csv", label="island_highres")
        analyze_csv("island_highres.csv", top_n=100)

    elif args[0] == "full_sweep":
        print("\n  FULL NUCLEAR LANDSCAPE SWEEP: Z=1-120, N=1-184")
        print("  This will take days. Consider running with --dry-run first.")
        run_sweep(range(1, 121), range(1, 185), n_steps=400,
                  output_file="full_nuclear_landscape.csv", label="full")
        analyze_csv("full_nuclear_landscape.csv")

    elif args[0] == "dryrun":
        # Estimate runtime without solving
        z_range = range(1, 121)
        n_range = range(1, 185)
        total = sum(1 for Z in z_range for N in n_range)
        est_per_isotope = 0.5  # seconds per isotope at n_steps=400
        est_hours = (total * est_per_isotope) / 3600
        print(f"\n  DRY RUN: {total} isotopes")
        print(f"  Estimated at {est_per_isotope}s/isotope: {est_hours:.1f} hours")
        print(f"  Estimated at 1.0s/isotope: {total/3600:.1f} hours")
        print(f"  Estimated at 2.0s/isotope: {total/1800:.1f} hours")
        print("\n  Tip: Use --quick for faster but less accurate runs")
        
    elif args[0] == "quick":
        # Fast but less accurate for exploration
        if len(args) > 1 and args[1] == "island":
            run_sweep(range(100, 127), range(150, 196), n_steps=200,
                      output_file="island_quick.csv", label="island_quick")
            analyze_csv("island_quick.csv")
        else:
            run_sweep(range(1, 83), range(1, 131), n_steps=200,
                      output_file="stability_quick.csv", label="quick")
            analyze_csv("stability_quick.csv")

    elif args[0] == "analyze":
        path = args[1] if len(args) > 1 else "stability_valley.csv"
        top = int(args[2]) if len(args) > 2 else 50
        analyze_csv(path, top_n=top)

    elif args[0] == "compare":
        # Compare two CSV outputs
        path1 = args[1]
        path2 = args[2]
        print(f"\n  COMPARING: {path1} vs {path2}")
        import pandas as pd
        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)
        merged = pd.merge(df1, df2, on=['Z', 'N'], suffixes=('_1', '_2'))
        diff = merged[merged['classification_1'] != merged['classification_2']]
        print(f"  Disagreements: {len(diff)} / {len(merged)} ({len(diff)/len(merged)*100:.1f}%)")
        if len(diff) > 0:
            print("\n  Top disagreements by anomaly difference:")
            diff['anomaly_diff'] = diff['gradient_anomaly_1'] - diff['gradient_anomaly_2']
            diff_sorted = diff.sort_values('anomaly_diff', key=abs, ascending=False).head(20)
            for _, row in diff_sorted.iterrows():
                print(f"    Z={int(row['Z'])} N={int(row['N'])}: {row['classification_1']} vs {row['classification_2']} "
                      f"(Δ={row['anomaly_diff']:+.3f})")

    else:
        print(__doc__)