"""
MobileFaceLoRA: Parameter-Efficient Face Recognition on Mobile Devices
via Low-Rank Adaptation of Foundation Models

Core module — contains all model definitions, datasets, training,
evaluation, ablation, and export utilities.
"""

import os, sys, math, time, copy, json, random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
import sklearn.preprocessing

CLIP_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]
FACE_IMAGE_MEAN = [0.5, 0.5, 0.5]
FACE_IMAGE_STD = [0.5, 0.5, 0.5]

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

DEFAULT_CONFIG = dict(
    # Foundation model
    clip_model='openai/clip-vit-base-patch16',
    hidden_dim=768,
    embed_dim=512,
    input_size=224,
    preprocess_mean=CLIP_IMAGE_MEAN,
    preprocess_std=CLIP_IMAGE_STD,

    # LoRA
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.10,
    lora_targets=['q_proj', 'v_proj'],

    # Identity-balanced training
    use_pk_sampler=False,
    sampler_identities_per_batch=16,
    sampler_images_per_identity=4,

    # ArcFace
    arcface_s=64.0,
    arcface_m=0.5,

    # Optional face-teacher distillation
    use_teacher_distillation=False,
    teacher_type='edgeface',
    teacher_model_name='edgeface_s_gamma_05',
    distill_weight=0.25,
    distill_loss='cosine',

    # Training
    batch_size=64,
    epochs=7,
    lr=5e-5,            # changed from 1e-4 to 5e-5
    lora_lr=None,
    head_lr=None,
    weight_decay=5e-4,
    seed=42,
    deterministic=False,
    arcface_label_smoothing=0.0,
    max_grad_norm=1.0,
    num_workers=4,
    # Set to an integer to stop after that many non-improving benchmark
    # evaluations. None means run the full configured epoch count.
    early_stopping_patience=3,
    early_stopping_min_delta=0.0,

    # Paths
    train_dir='data/aligned_train_224',
    val_dir='data/val',
    output_dir='outputs',
    edgeface_dir='third_party/edgeface',

    # Benchmarks available
    benchmarks=dict(
        LFW=('lfw_ann.txt', 'lfw_112x112'),
        AgeDB_30=('agedb_30_ann.txt', 'agedb_30_112x112'),
        CALFW=('calfw_ann.txt', 'calfw_112x112'),
        CPLFW=('cplfw_ann.txt', 'cplfw_112x112'),
    ),
)


EXPERIMENT_PRESETS = {
    # Reproduce the old notebook baseline before comparing any fix.
    'old_preprocess_r8': dict(
        preprocess_mean=FACE_IMAGE_MEAN,
        preprocess_std=FACE_IMAGE_STD,
        lora_rank=8,
        lora_alpha=16,
        lora_targets=['q_proj', 'v_proj'],
        arcface_label_smoothing=0.1,
        max_grad_norm=1.0,
    ),
    # Main next run: same rank and targets, but with CLIP-native inputs.
    'clip_preprocess_r8': dict(
        preprocess_mean=CLIP_IMAGE_MEAN,
        preprocess_std=CLIP_IMAGE_STD,
        lora_rank=8,
        lora_alpha=16,
        lora_targets=['q_proj', 'v_proj'],
        arcface_label_smoothing=0.0,
        max_grad_norm=1.0,
    ),
    # Same corrected preprocessing, but gentler updates for verification.
    'clip_preprocess_r8_low_lr': dict(
        preprocess_mean=CLIP_IMAGE_MEAN,
        preprocess_std=CLIP_IMAGE_STD,
        lora_rank=8,
        lora_alpha=16,
        lora_targets=['q_proj', 'v_proj'],
        lr=2e-5,
        arcface_label_smoothing=0.0,
        max_grad_norm=1.0,
        early_stopping_patience=3,
    ),
    # Follow-up only if CLIP preprocessing improves average benchmark accuracy.
    'clip_qkvo_r8': dict(
        preprocess_mean=CLIP_IMAGE_MEAN,
        preprocess_std=CLIP_IMAGE_STD,
        lora_rank=8,
        lora_alpha=16,
        lora_targets=['q_proj', 'k_proj', 'v_proj', 'out_proj'],
        arcface_label_smoothing=0.0,
        max_grad_norm=1.0,
    ),
    # Stronger next experiment: adapter capacity + identity-balanced
    # batches + face-teacher guidance. This tests whether the missing
    # signal is biometric geometry rather than LoRA rank alone.
    'hybrid_teacher_qkvo_r8': dict(
        preprocess_mean=CLIP_IMAGE_MEAN,
        preprocess_std=CLIP_IMAGE_STD,
        lora_rank=8,
        lora_alpha=16,
        lora_targets=['q_proj', 'k_proj', 'v_proj', 'out_proj'],
        lr=2e-5,
        lora_lr=2e-5,
        head_lr=1e-4,
        epochs=7,
        arcface_label_smoothing=0.0,
        max_grad_norm=1.0,
        early_stopping_patience=3,
        use_pk_sampler=True,
        sampler_identities_per_batch=16,
        sampler_images_per_identity=4,
        use_teacher_distillation=True,
        teacher_type='edgeface',
        teacher_model_name='edgeface_s_gamma_05',
        distill_weight=0.25,
    ),
}


def make_experiment_config(preset='clip_preprocess_r8', **overrides):
    """Create a full config for a named experiment preset."""
    if preset not in EXPERIMENT_PRESETS:
        available = ', '.join(sorted(EXPERIMENT_PRESETS))
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(EXPERIMENT_PRESETS[preset])
    cfg.update(overrides)
    return cfg


def set_reproducibility(seed=42, deterministic=False):
    """Seed common RNGs so repeated experiments are easier to compare."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ══════════════════════════════════════════════════════════════
#  PHASE 3 — TRAINING DATASET  (VGGFace2 folder structure)
# ══════════════════════════════════════════════════════════════

class VGGFace2Dataset(Dataset):
    """Loads face images from identity sub-folders (VGGFace2 layout)."""

    def __init__(self, root_dir, transform=None, teacher_transform=None):
        self.transform = transform
        self.teacher_transform = teacher_transform
        self.samples = []
        self.samples_by_class = defaultdict(list)
        self.class_to_idx = {}
        identities = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        for idx, identity in enumerate(identities):
            self.class_to_idx[identity] = idx
            id_dir = os.path.join(root_dir, identity)
            for fname in os.listdir(id_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    sample_idx = len(self.samples)
                    self.samples.append((os.path.join(id_dir, fname), idx))
                    self.samples_by_class[idx].append(sample_idx)
        print(f"[VGGFace2] {len(self.samples)} images, "
              f"{len(identities)} identities")

    @property
    def num_classes(self):
        return len(self.class_to_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        teacher_img = None
        if self.teacher_transform:
            teacher_img = self.teacher_transform(img)
        if self.transform:
            img = self.transform(img)
        if teacher_img is not None:
            return img, teacher_img, label
        return img, label


class PKIdentityBatchSampler(Sampler):
    """Yield P x K batches with several images per identity.

    ArcFace optimizes class separation, but face verification also needs
    stable positive structure inside the batch. P x K sampling increases
    same-identity coverage without changing the dataset.
    """

    def __init__(self, samples_by_class, identities_per_batch=16,
                 images_per_identity=4, batches_per_epoch=None, seed=42):
        self.samples_by_class = {
            cls: list(indices) for cls, indices in samples_by_class.items()
            if len(indices) > 0
        }
        self.classes = sorted(self.samples_by_class)
        self.identities_per_batch = int(identities_per_batch)
        self.images_per_identity = int(images_per_identity)
        self.batch_size = self.identities_per_batch * self.images_per_identity
        self.seed = seed
        self.epoch = 0
        total_samples = sum(len(v) for v in self.samples_by_class.values())
        self.batches_per_epoch = (
            batches_per_epoch or max(1, total_samples // self.batch_size)
        )
        if self.identities_per_batch < 1 or self.images_per_identity < 1:
            raise ValueError("P x K sampler values must be positive")
        if not self.classes:
            raise ValueError("P x K sampler received no identities")

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            if len(self.classes) >= self.identities_per_batch:
                chosen_classes = rng.sample(
                    self.classes, self.identities_per_batch)
            else:
                chosen_classes = [
                    rng.choice(self.classes)
                    for _ in range(self.identities_per_batch)
                ]

            batch = []
            for cls in chosen_classes:
                candidates = self.samples_by_class[cls]
                if len(candidates) >= self.images_per_identity:
                    batch.extend(rng.sample(
                        candidates, self.images_per_identity))
                else:
                    batch.extend([
                        rng.choice(candidates)
                        for _ in range(self.images_per_identity)
                    ])
            rng.shuffle(batch)
            yield batch
        self.epoch += 1

    def __len__(self):
        return self.batches_per_epoch


def _normalization_stats(mean=None, std=None):
    return (
        CLIP_IMAGE_MEAN if mean is None else mean,
        CLIP_IMAGE_STD if std is None else std,
    )


def get_train_transforms(input_size=224, mean=None, std=None,
                         interpolation=transforms.InterpolationMode.BICUBIC):
    mean, std = _normalization_stats(mean, std)
    return transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.1),
        transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1),
    ])


def get_val_transforms(input_size=224, mean=None, std=None,
                       interpolation=transforms.InterpolationMode.BICUBIC):
    mean, std = _normalization_stats(mean, std)
    return transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=interpolation),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_edgeface_teacher_transforms():
    """Deterministic preprocessing for EdgeFace teacher embeddings."""
    return get_val_transforms(
        112,
        FACE_IMAGE_MEAN,
        FACE_IMAGE_STD,
        interpolation=transforms.InterpolationMode.BILINEAR,
    )


def build_training_data(cfg):
    """Create the training dataset and loader for standard or hybrid runs."""
    train_transform = get_train_transforms(
        cfg['input_size'],
        cfg.get('preprocess_mean'),
        cfg.get('preprocess_std'),
    )
    teacher_transform = None
    if cfg.get('use_teacher_distillation'):
        teacher_transform = get_edgeface_teacher_transforms()

    train_ds = VGGFace2Dataset(
        cfg['train_dir'],
        transform=train_transform,
        teacher_transform=teacher_transform,
    )

    loader_kwargs = dict(
        num_workers=cfg['num_workers'],
        pin_memory=True,
        persistent_workers=(cfg['num_workers'] > 0),
    )
    if cfg.get('use_pk_sampler'):
        p = cfg.get('sampler_identities_per_batch', 16)
        k = cfg.get('sampler_images_per_identity', 4)
        batch_sampler = PKIdentityBatchSampler(
            train_ds.samples_by_class,
            identities_per_batch=p,
            images_per_identity=k,
            seed=cfg.get('seed', 42),
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )
        print(f"[Sampler] P x K identity batches enabled: P={p}, K={k}, "
              f"batch_size={p*k}")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg['batch_size'],
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
    return train_ds, train_loader


# ══════════════════════════════════════════════════════════════
#  PHASE 4 — MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════

class ArcFaceHead(nn.Module):
    """Additive Angular Margin (ArcFace) classification head.
    Used ONLY during training; discarded at inference.

    Uses the trigonometric identity cos(θ+m) = cosθ·cos(m) - sinθ·sin(m)
    to avoid numerically unstable acos. Includes piecewise fallback for
    hard samples where θ+m > π (Deng et al., 2019)."""

    def __init__(self, in_features, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precalculate margin constants for speed and stability
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        norm_e = F.normalize(embeddings)
        norm_w = F.normalize(self.weight)
        cos = F.linear(norm_e, norm_w).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Trigonometric identity for cos(theta + m)
        sin = torch.sqrt(1.0 - torch.pow(cos, 2).clamp(0, 1))
        phi = cos * self.cos_m - sin * self.sin_m

        # Safe margin: if theta + m > pi, use monotonic fallback
        phi = torch.where(cos > self.th, phi, cos - self.mm)

        # Apply margin only to ground truth classes
        one_hot = F.one_hot(labels, self.weight.size(0)).float()
        logits = (one_hot * phi) + ((1.0 - one_hot) * cos)
        logits *= self.s

        return logits


class MobileFaceLoRA(nn.Module):
    """CLIP-ViT backbone + LoRA adapters + projection + ArcFace head."""

    def __init__(self, peft_backbone, hidden_dim, embed_dim, num_classes,
                 s=64.0, m=0.5):
        super().__init__()
        self.backbone = peft_backbone
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        self.arcface = ArcFaceHead(embed_dim, num_classes, s, m)

    def get_embedding(self, x):
        out = self.backbone(pixel_values=x)
        features = out.pooler_output          # (B, hidden_dim)
        embedding = self.projection(features)  # (B, embed_dim)
        return embedding

    def forward(self, x, labels=None):
        emb = self.get_embedding(x)
        if labels is not None:
            logits = self.arcface(emb, labels)
            return logits, emb
        return emb


def build_model(cfg, num_classes, device):
    """Build the full MobileFaceLoRA model."""
    from transformers import CLIPVisionModel
    from peft import LoraConfig, get_peft_model

    if cfg.get('seed') is not None:
        set_reproducibility(
            cfg.get('seed', 42),
            cfg.get('deterministic', False),
        )

    # 1. Load CLIP-ViT backbone
    backbone = CLIPVisionModel.from_pretrained(cfg['clip_model'])

    # 2. Freeze all backbone weights
    for p in backbone.parameters():
        p.requires_grad = False

    # 3. Inject LoRA adapters
    lora_cfg = LoraConfig(
        r=cfg['lora_rank'],
        lora_alpha=cfg['lora_alpha'],
        target_modules=cfg['lora_targets'],
        lora_dropout=cfg['lora_dropout'],
        bias='none',
    )
    peft_backbone = get_peft_model(backbone, lora_cfg)
    peft_backbone.print_trainable_parameters()

    # 4. Assemble full model
    model = MobileFaceLoRA(
        peft_backbone,
        hidden_dim=cfg['hidden_dim'],
        embed_dim=cfg['embed_dim'],
        num_classes=num_classes,
        s=cfg['arcface_s'],
        m=cfg['arcface_m'],
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total: {total/1e6:.2f}M | "
          f"Trainable: {trainable/1e6:.2f}M "
          f"({100*trainable/total:.2f}%)")
    return model


# ══════════════════════════════════════════════════════════════
#  PHASE 6 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def build_optimizer(model, cfg):
    """Build optimizer groups so adapters and head can use different LRs."""
    lora_params, head_params, other_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora_' in name:
            lora_params.append(param)
        elif name.startswith('projection') or name.startswith('arcface'):
            head_params.append(param)
        else:
            other_params.append(param)

    groups = []
    if lora_params:
        groups.append({
            'params': lora_params,
            'lr': cfg.get('lora_lr') or cfg['lr'],
        })
    if head_params:
        groups.append({
            'params': head_params,
            'lr': cfg.get('head_lr') or cfg['lr'],
        })
    if other_params:
        groups.append({'params': other_params, 'lr': cfg['lr']})

    return torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])


def _unpack_training_batch(batch):
    if len(batch) == 3:
        images, teacher_images, labels = batch
        return images, teacher_images, labels
    images, labels = batch
    return images, None, labels


def _distillation_loss(student_emb, teacher_emb, loss_type='cosine'):
    student = F.normalize(student_emb, dim=1)
    teacher = F.normalize(teacher_emb, dim=1)
    if loss_type == 'mse':
        return F.mse_loss(student, teacher)
    return (1.0 - F.cosine_similarity(student, teacher, dim=1)).mean()


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, cfg=None,
                    teacher_model=None):
    model.train()
    cfg = cfg or {}
    label_smoothing = cfg.get('arcface_label_smoothing', 0.0)
    max_grad_norm = cfg.get('max_grad_norm', None)
    distill_weight = cfg.get('distill_weight', 0.0)
    distill_loss_type = cfg.get('distill_loss', 'cosine')
    total_loss, correct, total = 0.0, 0, 0
    total_cls_loss, total_distill_loss = 0.0, 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    if teacher_model is not None:
        teacher_model.eval()
    for batch in pbar:
        images, teacher_images, labels = _unpack_training_batch(batch)
        images, labels = images.to(device), labels.to(device)
        if teacher_images is not None:
            teacher_images = teacher_images.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, student_emb = model(images, labels)
            cls_loss = F.cross_entropy(
                logits, labels, label_smoothing=label_smoothing)
            loss = cls_loss
            distill_loss = torch.zeros((), device=device)
            if (teacher_model is not None and teacher_images is not None
                    and distill_weight > 0):
                with torch.no_grad():
                    teacher_emb = teacher_model.get_embedding(teacher_images)
                distill_loss = _distillation_loss(
                    student_emb, teacher_emb, distill_loss_type)
                loss = loss + distill_weight * distill_loss
        scaler.scale(loss).backward()
        if max_grad_norm:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        total_cls_loss += cls_loss.item() * images.size(0)
        total_distill_loss += distill_loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
        postfix = {
            'loss': f"{loss.item():.4f}",
            'acc': f"{100*correct/total:.1f}%",
        }
        if teacher_model is not None and teacher_images is not None:
            postfix['distill'] = f"{distill_loss.item():.4f}"
        pbar.set_postfix(**postfix)
    return {
        'loss': total_loss / total,
        'cls_loss': total_cls_loss / total,
        'distill_loss': total_distill_loss / total,
        'acc': correct / total,
    }


def train(model, train_loader, cfg, device, eval_fn=None, teacher_model=None):
    """Full training loop with cosine annealing LR and best-checkpoint
    selection based on average benchmark accuracy."""
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs']
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    history = {
        'loss': [],
        'cls_loss': [],
        'distill_loss': [],
        'acc': [],
        'lr': [],
        'bench': [],
    }

    if cfg.get('use_teacher_distillation') and teacher_model is None:
        teacher_model = build_teacher_model(cfg, device)
    if cfg.get('use_teacher_distillation'):
        dataset = getattr(train_loader, 'dataset', None)
        if not getattr(dataset, 'teacher_transform', None):
            print("[Warning] Teacher distillation is enabled, but the "
                  "loader does not return teacher images. Use "
                  "build_training_data(cfg) for hybrid runs.")

    # Best-checkpoint tracking (by benchmark accuracy)
    best_benchmark_acc = 0.0
    best_epoch = 0
    best_state = None
    no_improve_epochs = 0
    patience = cfg.get('early_stopping_patience', None)
    min_delta = cfg.get('early_stopping_min_delta', 0.0)

    for epoch in range(1, cfg['epochs'] + 1):
        t0 = time.time()
        metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, cfg,
            teacher_model=teacher_model)
        loss = metrics['loss']
        acc = metrics['acc']
        elapsed = time.time() - t0
        scheduler.step()
        lr_values = [group['lr'] for group in optimizer.param_groups]
        lr = lr_values[0]
        history['loss'].append(loss)
        history['cls_loss'].append(metrics['cls_loss'])
        history['distill_loss'].append(metrics['distill_loss'])
        history['acc'].append(acc)
        history['lr'].append(lr)
        lr_text = ','.join(f"{value:.6f}" for value in lr_values)
        print(f"Epoch {epoch}/{cfg['epochs']} — "
              f"Loss: {loss:.4f} | Acc: {100*acc:.2f}% | LR: {lr_text} | "
              f"Time: {elapsed/60:.1f}min")

        # Benchmark evaluation every epoch
        if eval_fn:
            should_stop = False
            # Free GPU memory occupied by training before evaluation
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            try:
                bench_results = eval_fn(model)
                history['bench'].append(bench_results)

                # Track best checkpoint by average benchmark accuracy
                if bench_results:
                    avg_acc = np.mean(
                        [v['accuracy'] for v in bench_results.values()])
                    print(f"  Avg benchmark: {100*avg_acc:.2f}%")
                    if avg_acc > best_benchmark_acc + min_delta:
                        best_benchmark_acc = avg_acc
                        best_epoch = epoch
                        best_state = copy.deepcopy(model.state_dict())
                        no_improve_epochs = 0
                        print(f"  ★ New best model "
                              f"(avg benchmark: {100*avg_acc:.2f}%)")
                    else:
                        no_improve_epochs += 1
                        if patience and no_improve_epochs >= patience:
                            should_stop = True
                            print(f"  Early stopping: no benchmark "
                                  f"improvement for {no_improve_epochs} "
                                  f"epochs")
            except Exception as e:
                print(f"  [Warning] Eval failed: {e}")
                history['bench'].append(None)
            finally:
                # Always restore training mode after eval
                model.train()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            if should_stop:
                break

    # Restore best model if we tracked benchmarks
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n★ Restored best model from epoch {best_epoch} "
              f"(avg benchmark: {100*best_benchmark_acc:.2f}%)")

    return history


# ══════════════════════════════════════════════════════════════
#  PHASE 7 — BENCHMARK EVALUATION
# ══════════════════════════════════════════════════════════════

class BenchmarkPairDataset(Dataset):
    """Loads image pairs from annotation file for FR benchmarks."""

    def __init__(self, val_dir, ann_file, img_subdir, transform=None):
        self.val_dir = val_dir
        self.transform = transform
        self.images = []    # ordered list of image paths
        self.labels = []    # per-pair label (1=same, 0=diff)
        ann_path = os.path.join(val_dir, ann_file)
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split()
                lbl = int(parts[0])
                img1 = os.path.join(val_dir, parts[1])
                img2 = os.path.join(val_dir, parts[2])
                self.images.extend([img1, img2])
                self.labels.append(lbl)
        print(f"  [{img_subdir}] {len(self.labels)} pairs, "
              f"{len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


def _calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, ~actual_issame))
    tn = np.sum(np.logical_and(~predict_issame, ~actual_issame))
    fn = np.sum(np.logical_and(~predict_issame, actual_issame))
    tpr = 0 if (tp+fn)==0 else float(tp)/float(tp+fn)
    fpr = 0 if (fp+tn)==0 else float(fp)/float(fp+tn)
    acc = float(tp+tn) / dist.size
    return tpr, fpr, acc


def evaluate_benchmark(embeddings, issame, nfolds=10):
    """10-fold cross-validation on pair embeddings (standard FR protocol)."""
    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]
    diff = np.subtract(emb1, emb2)
    dist = np.sum(np.square(diff), axis=1)
    issame = np.asarray(issame, dtype=bool)
    thresholds = np.arange(0, 4, 0.01)

    kfold = KFold(n_splits=nfolds, shuffle=False)
    accuracies = np.zeros(nfolds)
    best_thresholds = np.zeros(nfolds)
    indices = np.arange(len(issame))

    for fold, (train_idx, test_idx) in enumerate(kfold.split(indices)):
        acc_train = np.zeros(len(thresholds))
        for ti, t in enumerate(thresholds):
            _, _, acc_train[ti] = _calculate_accuracy(
                t, dist[train_idx], issame[train_idx])
        best_t = thresholds[np.argmax(acc_train)]
        _, _, accuracies[fold] = _calculate_accuracy(
            best_t, dist[test_idx], issame[test_idx])
        best_thresholds[fold] = best_t

    return np.mean(accuracies), np.std(accuracies), np.mean(best_thresholds)


@torch.no_grad()
def extract_embeddings(model, dataset, device, batch_size=32):
    """Extract L2-normalised embeddings for all images in dataset."""
    model.eval()
    # Free cached GPU memory before large eval pass
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=False)
    all_emb = []
    for imgs in tqdm(loader, desc="  Extracting", leave=False):
        imgs = imgs.to(device)
        emb = model.get_embedding(imgs)
        all_emb.append(emb.detach().cpu().numpy())
        del imgs, emb  # Explicitly free GPU tensors
    embeddings = np.concatenate(all_emb, axis=0)
    embeddings = sklearn.preprocessing.normalize(embeddings)
    return embeddings


def evaluate_all_benchmarks(model, cfg, device):
    """Run evaluation on all configured benchmarks. Returns dict of results."""
    transform = get_val_transforms(
        cfg['input_size'],
        cfg.get('preprocess_mean'),
        cfg.get('preprocess_std'),
    )
    results = {}
    for name, (ann, img_dir) in cfg['benchmarks'].items():
        ds = BenchmarkPairDataset(cfg['val_dir'], ann, img_dir, transform)
        embs = extract_embeddings(model, ds, device)
        acc, std, thresh = evaluate_benchmark(embs, ds.labels)
        results[name] = {'accuracy': acc, 'std': std, 'threshold': thresh}
        print(f"  {name}: {100*acc:.2f}% ± {100*std:.2f}% "
              f"(threshold={thresh:.3f})")
    return results


# ══════════════════════════════════════════════════════════════
#  PHASE 8 — LoRA RANK ABLATION STUDY
# ══════════════════════════════════════════════════════════════

def ablation_study(ranks, cfg, device):
    """Train & evaluate across multiple LoRA ranks."""
    # Prepare data once
    train_ds, train_loader = build_training_data(cfg)

    all_results = []
    for rank in ranks:
        print(f"\n{'='*60}")
        print(f"  ABLATION: LoRA rank = {rank}")
        print(f"{'='*60}")
        rank_cfg = {**cfg, 'lora_rank': rank, 'lora_alpha': rank * 2}
        model = build_model(rank_cfg, train_ds.num_classes, device)
        history = train(model, train_loader, rank_cfg, device)
        print("\nBenchmark Evaluation:")
        bench = evaluate_all_benchmarks(model, rank_cfg, device)

        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        result = {
            'rank': rank,
            'trainable_params_M': trainable / 1e6,
            'total_params_M': total / 1e6,
            'param_ratio': trainable / total,
            'final_loss': history['loss'][-1],
            'final_train_acc': history['acc'][-1],
        }
        for bname, bres in bench.items():
            result[f'{bname}_acc'] = bres['accuracy']
        all_results.append(result)

        # Save checkpoint
        ckpt_path = os.path.join(
            cfg['output_dir'], f'mobilefacelora_r{rank}.pth')
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved: {ckpt_path}")

    return all_results


# ══════════════════════════════════════════════════════════════
#  PHASE 9 — MOBILE EXPORT PIPELINE
# ══════════════════════════════════════════════════════════════

def export_for_mobile(model, cfg, device):
    """Merge LoRA → ONNX → (optionally TFLite)."""
    from peft import PeftModel
    model.eval()

    # 1. Merge LoRA weights into base model
    if isinstance(model.backbone, PeftModel):
        model.backbone = model.backbone.merge_and_unload()
        print("[Export] LoRA weights merged into backbone.")

    # 2. Export to ONNX
    dummy = torch.randn(1, 3, cfg['input_size'], cfg['input_size']).to(device)
    onnx_path = os.path.join(cfg['output_dir'], 'mobilefacelora.onnx')

    # Wrap for ONNX export (only embedding, no ArcFace head)
    class EmbeddingWrapper(nn.Module):
        def __init__(self, full_model):
            super().__init__()
            self.backbone = full_model.backbone
            self.projection = full_model.projection
        def forward(self, x):
            out = self.backbone(pixel_values=x)
            return self.projection(out.pooler_output)

    wrapper = EmbeddingWrapper(model).to(device).eval()
    torch.onnx.export(
        wrapper, dummy, onnx_path, opset_version=17,
        input_names=['face_image'], output_names=['embedding'],
        dynamic_axes={'face_image': {0: 'batch'}, 'embedding': {0: 'batch'}},
    )
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"[Export] ONNX saved: {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path, size_mb


# ══════════════════════════════════════════════════════════════
#  PHASE 11 — VISUALIZATION & METRICS
# ══════════════════════════════════════════════════════════════

def plot_training_history(history, save_dir='outputs'):
    """Plot training loss and accuracy curves."""
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['loss'])+1)

    ax1.plot(epochs, history['loss'], 'b-o', markersize=4)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss'); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a*100 for a in history['acc']], 'r-o', markersize=4)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy'); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.show()


def plot_ablation_results(results, save_dir='outputs'):
    """Generate ablation study figures for the paper."""
    import matplotlib.pyplot as plt
    ranks = [r['rank'] for r in results]
    lfw = [r.get('LFW_acc', 0)*100 for r in results]
    params = [r['trainable_params_M'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Fig 1: Accuracy vs Rank
    axes[0].plot(ranks, lfw, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('LoRA Rank'); axes[0].set_ylabel('LFW Accuracy (%)')
    axes[0].set_title('Fig 3: LFW Accuracy vs LoRA Rank')
    axes[0].grid(True, alpha=0.3)

    # Fig 2: Params vs Rank
    axes[1].bar(ranks, params, color='steelblue', alpha=0.8)
    axes[1].set_xlabel('LoRA Rank')
    axes[1].set_ylabel('Trainable Params (M)')
    axes[1].set_title('Fig 4: Trainable Parameters vs LoRA Rank')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Fig 3: Pareto — Accuracy vs Params
    axes[2].scatter(params, lfw, s=100, c='red', zorder=5)
    axes[2].plot(params, lfw, 'r--', alpha=0.5)
    for r, p, a in zip(ranks, params, lfw):
        axes[2].annotate(f'r={r}', (p, a), textcoords='offset points',
                         xytext=(5, 5), fontsize=9)
    axes[2].set_xlabel('Trainable Params (M)')
    axes[2].set_ylabel('LFW Accuracy (%)')
    axes[2].set_title('Fig 5: Pareto — Accuracy vs Efficiency')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_figures.png'), dpi=150)
    plt.show()


def print_results_table(results):
    """Print thesis-ready comparison table."""
    header = (f"{'Model':<25} {'LFW':>8} {'AgeDB':>8} "
              f"{'CALFW':>8} {'CPLFW':>8} {'Params(M)':>10}")
    print("\n" + "="*75)
    print("  TABLE 1 — MobileFaceLoRA Results")
    print("="*75)
    print(header)
    print("-"*75)
    for r in results:
        name = f"MobileFaceLoRA r={r['rank']}"
        lfw = r.get('LFW_acc', 0)*100
        age = r.get('AgeDB_30_acc', 0)*100
        cal = r.get('CALFW_acc', 0)*100
        cpl = r.get('CPLFW_acc', 0)*100
        params = r['trainable_params_M']
        print(f"{name:<25} {lfw:>7.2f}% {age:>7.2f}% "
              f"{cal:>7.2f}% {cpl:>7.2f}% {params:>9.3f}")
    print("="*75)


# ══════════════════════════════════════════════════════════════
#  EDGEFACE BASELINE (for comparison)
# ══════════════════════════════════════════════════════════════

class EdgeFaceEmbeddingWrapper:
    """Small adapter so EdgeFace can act as evaluator or teacher."""

    def __init__(self, model):
        self.model = model

    def eval(self):
        self.model.eval()
        return self

    def get_embedding(self, x):
        return self.model(x)


def load_edgeface_wrapper(cfg, device, model_name=None,
                          require_checkpoint=False):
    """Load an EdgeFace checkpoint and expose get_embedding()."""
    model_name = model_name or cfg.get(
        'teacher_model_name', 'edgeface_s_gamma_05')
    edgeface_dir = os.path.abspath(cfg.get(
        'edgeface_dir', os.path.join('third_party', 'edgeface')))
    if not os.path.isdir(edgeface_dir):
        raise FileNotFoundError(
            f"EdgeFace source directory not found: {edgeface_dir}. "
            "Run 'git submodule update --init --recursive'.")
    if edgeface_dir not in sys.path:
        sys.path.insert(0, edgeface_dir)
    from backbones import get_model as get_ef_model

    model = get_ef_model(model_name)
    ckpt = os.path.join(edgeface_dir, 'checkpoints', f'{model_name}.pt')
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        print(f"[EdgeFace] Loaded checkpoint: {ckpt}")
    else:
        message = f"[EdgeFace] Checkpoint not found: {ckpt}"
        if require_checkpoint:
            raise FileNotFoundError(message)
        print(f"{message}. Using randomly initialized weights.")
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return EdgeFaceEmbeddingWrapper(model)


def build_teacher_model(cfg, device):
    """Build the optional face-recognition teacher for distillation."""
    teacher_type = cfg.get('teacher_type', 'edgeface')
    if teacher_type != 'edgeface':
        raise ValueError(f"Unsupported teacher_type: {teacher_type}")
    print(f"[Teacher] Loading {teacher_type}: "
          f"{cfg.get('teacher_model_name', 'edgeface_s_gamma_05')}")
    return load_edgeface_wrapper(cfg, device, require_checkpoint=True)


def evaluate_edgeface_baseline(cfg, device, model_name='edgeface_s_gamma_05'):
    """Load pretrained EdgeFace and evaluate on benchmarks."""
    sys.path.insert(0, os.path.join(os.getcwd(), 'edgeface-master'))
    from backbones import get_model as get_ef_model

    model = get_ef_model(model_name)
    ckpt = os.path.join('edgeface-master', 'checkpoints',
                        f'{model_name}.pt')
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        print(f"[EdgeFace] Loaded checkpoint: {ckpt}")
    model = model.to(device).eval()

    # Wrap to match our interface
    class EdgeFaceWrapper:
        def __init__(self, m): self.m = m
        def eval(self): self.m.eval(); return self
        def get_embedding(self, x):
            # EdgeFace expects 112x112, normalize to [-1,1]
            return self.m(x)

    wrapper = EdgeFaceWrapper(model)
    # Use 112x112 transforms for EdgeFace
    transform = get_val_transforms(
        112,
        FACE_IMAGE_MEAN,
        FACE_IMAGE_STD,
        interpolation=transforms.InterpolationMode.BILINEAR,
    )
    results = {}
    for name, (ann, img_dir) in cfg['benchmarks'].items():
        ds = BenchmarkPairDataset(cfg['val_dir'], ann, img_dir, transform)
        embs = extract_embeddings(wrapper, ds, device)
        acc, std, thresh = evaluate_benchmark(embs, ds.labels)
        results[name] = {'accuracy': acc, 'std': std}
        print(f"  [EdgeFace] {name}: {100*acc:.2f}% ± {100*std:.2f}%")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [EdgeFace] Total params: {params:.2f}M")
    return results, params
