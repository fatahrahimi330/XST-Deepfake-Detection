import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from einops import rearrange, repeat
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — exact values from Table 2 and Experiment 6 of the paper
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = "processed_ffpp"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"
# Table 2
IMG_SIZE     = 224
PATCH_SIZE   = 7
NUM_CLASSES  = 2
CHANNELS     = 512      # FL output channels
DIM          = 1024     # patch embedding dimension
HEADS        = 8
MLP_DIM      = 2048
DEPTH        = 6
WEIGHT_DECAY = 0.0000001   # 1e-7
BATCH_SIZE   = 32
# Paper: "Augmentation: No"
AUGMENT      = False

# Experiment 6 (MTCNN + FaceForensics++, best result = 85% test acc)
EPOCHS       = 100
LR           = 1e-7
PATIENCE     = 10       # early stopping patience

NUM_WORKERS  = 4
SAVE_PATH    = "model_c_cvit.pth"

# Normalization — paper uses ImageNet stats (standard for VGG-based FL)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATASET
#     Paper: "Augmentation: No" for Model C
#     Paper: face images at 224×224 (already done by your MTCNN pipeline)
# ══════════════════════════════════════════════════════════════════════════════

class FaceDataset(Dataset):

    EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

    TRANSFORM = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])

    def __init__(self, root: str, split: str, percent: float = 1.0):

        split_dir = os.path.join(root, split)

        real_samples = []
        fake_samples = []

        # scan dataset
        for cls in ["fake", "real"]:   # order does NOT matter anymore
            cls_dir = os.path.join(split_dir, cls)

            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Directory not found: {cls_dir}")

            files = []
            for ext in self.EXTS:
                files.extend(
                    glob.glob(os.path.join(cls_dir, "**", ext), recursive=True)
                )

            if cls == "real":
                real_samples.extend([(p, 0) for p in files])
            else:
                fake_samples.extend([(p, 1) for p in files])

        # -------- apply percentage sampling --------
        if percent < 1.0:
            import random
            random.seed(42)

            real_n = max(1, int(len(real_samples) * percent))
            fake_n = max(1, int(len(fake_samples) * percent))

            real_samples = random.sample(real_samples, real_n)
            fake_samples = random.sample(fake_samples, fake_n)

        self.samples = real_samples + fake_samples

        # shuffle dataset
        import random
        random.shuffle(self.samples)

        n_real = len(real_samples)
        n_fake = len(fake_samples)

        print(f"[{split:5s}] {len(self.samples):7,} frames "
              f"({n_real:,} real / {n_fake:,} fake)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.TRANSFORM(img), label



# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE LEARNING (FL) COMPONENT
#
#     Section 4.2.3: "The FL component follows the structure of VGG architecture.
#     It has 17 convolutional layers, with a kernel of 3×3. All convolutional
#     layers have a stride and padding of 1. Batch normalization and ReLU are
#     applied in all layers. Five max-pooling of 2×2 with stride 2. After each
#     max-pooling, the width (channel) is doubled, with the first layer having
#     32 channels and the last layer 512."
#
#     Block layout (3 conv per block except last 2 which have 4):
#       Block1: 3 conv,  32ch  → MaxPool → 112×112
#       Block2: 3 conv,  64ch  → MaxPool →  56×56
#       Block3: 3 conv, 128ch  → MaxPool →  28×28
#       Block4: 4 conv, 256ch  → MaxPool →  14×14
#       Block5: 4 conv, 512ch  → MaxPool →   7×7
#     Total conv layers: 3+3+3+4+4 = 17 ✓
# ══════════════════════════════════════════════════════════════════════════════

def conv_bn_relu(in_ch, out_ch, kernel=3, stride=1, padding=1):
    """Single Conv → BN → ReLU unit (Table 2: kernel (3,3), padding 1, stride 1)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class FeatureLearning(nn.Module):
    """
    VGG-style CNN without FC layers.
    Input : (B, 3, 224, 224)
    Output: (B, 512, 7, 7)   — 10.8M parameters as stated in the paper
    """

    def __init__(self):
        super().__init__()

        # Block 1: 3 conv, 3→32, MaxPool → 112×112
        self.block1 = nn.Sequential(
            conv_bn_relu(3,   32),
            conv_bn_relu(32,  32),
            conv_bn_relu(32,  32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 2: 3 conv, 32→64, MaxPool → 56×56
        self.block2 = nn.Sequential(
            conv_bn_relu(32,  64),
            conv_bn_relu(64,  64),
            conv_bn_relu(64,  64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 3: 3 conv, 64→128, MaxPool → 28×28
        self.block3 = nn.Sequential(
            conv_bn_relu(64,  128),
            conv_bn_relu(128, 128),
            conv_bn_relu(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 4: 4 conv, 128→256, MaxPool → 14×14
        self.block4 = nn.Sequential(
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
            conv_bn_relu(256, 256),
            conv_bn_relu(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 5: 4 conv, 256→512, MaxPool → 7×7
        self.block5 = nn.Sequential(
            conv_bn_relu(256, 512),
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x   # (B, 512, 7, 7)



class FeedForward(nn.Module):
    """MLP block inside Transformer encoder."""
    def __init__(self, dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.ReLU(inplace=True),      # paper uses ReLU in MLP
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        self.heads   = heads
        self.scale   = (dim // heads) ** -0.5
        self.norm    = nn.LayerNorm(dim)
        self.to_qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.to_out  = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x   = self.norm(x)
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    """Stacked encoder blocks (depth=6 per Table 2)."""
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads=heads, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
            ])
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return self.norm(x)


class ViTComponent(nn.Module):
    """
    ViT component as described in Section 4.2.3.

    FL output 512×7×7 → split into 7 row-patches of size 512×7=3584
    → Linear projection to dim=1024
    → CLS token prepended → 8 tokens total → pos_embed (8×1024)
    → Transformer encoder (depth=6, heads=8, mlp_dim=2048)
    → MLP head: Linear(1024→2048) → ReLU → Linear(2048→2)
    """

    def __init__(self, num_patches=7, patch_dim=512*7,
                 dim=1024, depth=6, heads=8,
                 mlp_dim=2048, num_classes=2, dropout=0.0):
        super().__init__()
        # Patch embedding: 3584 → 1024
        self.patch_embed = nn.Linear(patch_dim, dim)

        # CLS token + positional embedding
        # "position embedding has a 2×1024 dimension" in Wodajo 2021;
        # with 7 patches + 1 CLS = 8 positions total → 8×1024
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)

        self.transformer = TransformerEncoder(
            dim=dim, depth=depth, heads=heads,
            mlp_dim=mlp_dim, dropout=dropout,
        )

        # MLP head: "first layer 2048, last layer 2" (Section 3 of Wodajo 2021)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, feature_map):
        # feature_map: (B, 512, 7, 7)
        B = feature_map.size(0)

        # Split 7×7 spatial grid into 7 row-patches, each of size 512*7=3584
        # (B, 512, 7, 7) → (B, 7, 512*7)
        x = feature_map.permute(0, 2, 1, 3)    # (B, 7, 512, 7)
        x = x.reshape(B, 7, -1)                # (B, 7, 3584)

        # Patch embedding → (B, 7, 1024)
        x = self.patch_embed(x)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)       # (B, 8, 1024)
        x   = x + self.pos_embed
        x   = self.dropout(x)

        # Transformer encoder
        x = self.transformer(x)

        # Classification from CLS token
        return self.mlp_head(x[:, 0])


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FULL MODEL C  —  FL + ViT
# ══════════════════════════════════════════════════════════════════════════════

class CViT(nn.Module):
    """
    Convolutional Vision Transformer — Model C of Soudy et al. (2024).
    Input : (B, 3, 224, 224)
    Output: (B, 2) logits
    """

    def __init__(self):
        super().__init__()
        self.fl  = FeatureLearning()
        self.vit = ViTComponent(
            num_patches = 7,
            patch_dim   = CHANNELS * PATCH_SIZE,  # 512 * 7 = 3584
            dim         = DIM,                    # 1024
            depth       = DEPTH,                  # 6
            heads       = HEADS,                  # 8
            mlp_dim     = MLP_DIM,                # 2048
            num_classes = NUM_CLASSES,            # 2
        )

    def forward(self, x):
        features = self.fl(x)       # (B, 512, 7, 7)
        return self.vit(features)   # (B, 2)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """Stops training when val loss does not improve for `patience` epochs."""

    def __init__(self, patience=10, save_path="best.pth"):
        self.patience   = patience
        self.save_path  = save_path
        self.best_loss  = float("inf")
        self.counter    = 0
        self.stopped    = False

    def step(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.save_path)
            return True          # improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
            return False


def run_epoch(model, loader, criterion, optimizer=None, device="cuda"):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with (torch.enable_grad() if training else torch.no_grad()):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total * 100


def train(model, train_loader, val_loader):

    model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    early_stop = EarlyStopping(patience=PATIENCE, save_path=SAVE_PATH)

    # --------- store history ---------
    train_losses = []
    val_losses   = []
    train_accs   = []
    val_accs     = []

    # --------- setup live plot ---------
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

    print(f"\n{'Epoch':>6} | {'Train loss':>10} {'Train acc':>10} "
          f"| {'Val loss':>9} {'Val acc':>9} | {'Status':>10}")
    print("-" * 72)

    for epoch in range(1, EPOCHS + 1):

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, DEVICE)

        va_loss, va_acc = run_epoch(model, val_loader, criterion,
                                    None, DEVICE)

        # --------- save metrics ---------
        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        improved = early_stop.step(va_loss, model)
        status   = "✓ saved" if improved else f"no impv ({early_stop.counter}/{PATIENCE})"

        print(f"{epoch:6d} | {tr_loss:10.4f} {tr_acc:9.2f}% "
              f"| {va_loss:9.4f} {va_acc:8.2f}% | {status}")

        # --------- update live plot ---------
        ax1.clear()
        ax2.clear()

        ax1.plot(train_losses, label="Train Loss")
        ax1.plot(val_losses, label="Val Loss")
        ax1.set_title("Loss Curve")
        ax1.set_xlabel("Epoch")
        ax1.legend()

        ax2.plot(train_accs, label="Train Acc")
        ax2.plot(val_accs, label="Val Acc")
        ax2.set_title("Accuracy Curve")
        ax2.set_xlabel("Epoch")
        ax2.legend()

        plt.pause(0.01)

        if early_stop.stopped:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    plt.ioff()
    plt.show()

    print(f"\nBest val loss : {early_stop.best_loss:.4f}")
    print(f"Weights saved : {SAVE_PATH}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, test_loader):
    model.eval()
    model.to(DEVICE)
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            logits = model(imgs.to(DEVICE))
            probs  = F.softmax(logits, dim=1)[:, 1]    # P(fake)
            preds  = logits.argmax(1)
            all_preds  += preds.cpu().tolist()
            all_labels += labels.tolist()
            all_probs  += probs.cpu().tolist()

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) * 100
    auc = roc_auc_score(all_labels, all_probs)

    print(f"\n── Test Results ──────────────────────────────────────")
    print(f"  Accuracy : {acc:.2f}%")
    print(f"  AUC      : {auc:.4f}")
    print()
    print(classification_report(
        all_labels, all_preds,
        target_names=["real", "fake"], digits=4
    ))
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"               pred_real  pred_fake")
    print(f"  actual_real   {cm[0,0]:8d}   {cm[0,1]:8d}")
    print(f"  actual_fake   {cm[1,0]:8d}   {cm[1,1]:8d}")
    return acc, auc


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Model C — CViT (Soudy et al. 2024, exact architecture)")
    print("=" * 60)
    print(f"Device      : {DEVICE}")
    print(f"IMG size    : {IMG_SIZE}  |  Patch size: {PATCH_SIZE}")
    print(f"Dim={DIM}  Heads={HEADS}  MLP dim={MLP_DIM}  Depth={DEPTH}")
    print(f"LR={LR}  WD={WEIGHT_DECAY}  Batch={BATCH_SIZE}  Epochs={EPOCHS}")
    print(f"Augmentation: {AUGMENT}  (paper: No)")
    print()

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("Scanning dataset …")
    train_ds = FaceDataset(DATASET_ROOT, "train", percent=0.01)
    val_ds   = FaceDataset(DATASET_ROOT, "val", percent=0.01)
    test_ds  = FaceDataset(DATASET_ROOT, "test", percent=0.01)


    loader_kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                     pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    # ── Model ─────────────────────────────────────────────────────────────────
    model    = CViT()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {n_params:,}")
    # Paper states ~38.6M (FL ≈10.8M + ViT ≈27.8M)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training …")
    print("=" * 60)
    train(model, train_loader, val_loader)

    # ── Test ──────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading best weights for test evaluation …")
    print("=" * 60)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    evaluate(model, test_loader)
