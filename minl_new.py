from __future__ import annotations
import argparse
from pathlib import Path
import os
import platform
from typing import Callable, List, Optional, Tuple
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from networks.mlp import SimpleMLP
from networks.cnn import SmallCNN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F  # <-- ADICIONE

try:
    # AIMET (quantização) - carregado de forma opcional
    from aimet_torch.quantsim import QuantizationSimModel
    from aimet_torch.defs import QuantScheme
    _AIMET_AVAILABLE = True
except Exception:  # ImportError ou outros
    _AIMET_AVAILABLE = False


def normalize_path(p: str) -> str:
    """Converte caminhos WSL (/mnt/c/...) para Windows (C:\\...).
    Se o caminho já for Windows retorna sem alterações.
    """
    def is_wsl() -> bool:
        # detecta WSL por variável de ambiente ou pelo release do kernel
        if os.environ.get("WSL_DISTRO_NAME"):
            return True
        try:
            release = platform.uname().release.lower()
            if "microsoft" in release or "wsl" in release:
                return True
        except Exception:
            pass
        return False

    # Se já for caminho WSL e estamos no WSL, não converte
    if p.startswith("/mnt/"):
        if is_wsl():
            return p
        parts = p.split("/")
        drive = parts[2].upper() + ":"
        rest = "/".join(parts[3:])
        return str(Path(drive) / Path(rest))
    return p


def load_image(path: str, resize: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Carrega uma imagem e retorna tensor float (C,H,W) normalizado em [0,1].
    """
    orig_path = path
    path = normalize_path(path)
    try:
        img = Image.open(path).convert("RGB")
    except FileNotFoundError:
        # fallback: se estivermos em WSL o caminho /mnt/... pode ser válido
        try:
            img = Image.open(orig_path).convert("RGB")
        except FileNotFoundError:
            # tentar versão WSL se a origem era Windows (C:\...)
            if isinstance(orig_path, str) and ":\\" in orig_path:
                # C:\Users\... -> /mnt/c/Users/...
                drive = orig_path[0].lower()
                rest = orig_path[2:].replace('\\', '/')
                wsl_path = f"/mnt/{drive}/{rest}"
                img = Image.open(wsl_path).convert("RGB")
            else:
                raise
    if resize is not None:
        img = img.resize(resize, Image.BILINEAR)
    to_tensor = transforms.ToTensor()
    return to_tensor(img)


class ImageDataset(Dataset):
    """Dataset que itera sobre imagens de um diretório.

    Se a pasta tiver subpastas, cada subpasta é tratada como uma classe.
    Caso contrário, todas as imagens recebem label 0.
    """

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(normalize_path(root))
        self.transform = transform
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        items: List[Tuple[Path, int]] = []
        class_map = {}
        subdirs = [d for d in self.root.iterdir() if d.is_dir()]
        if subdirs:
            for idx, sd in enumerate(sorted(subdirs)):
                class_map[sd.name] = idx
                for f in sd.rglob("*"):
                    if f.suffix.lower() in exts:
                        items.append((f, idx))
        else:
            for f in self.root.rglob("*"):
                if f.is_file() and f.suffix.lower() in exts:
                    items.append((f, 0))

        self.items = items
        self.class_map = class_map or {"_": 0}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label


def positional_encoding(coords: torch.Tensor, L: int = 6, include_input: bool = False) -> torch.Tensor:
    """Aplica Position Encoding tipo Fourier às coordenadas.

    Args:
        coords: tensor de forma (..., 2) com coordenadas normalizadas em [0,1].
        L: número de bandas/frequências (k = 0..L-1, frequência = 2^k * pi).
        include_input: se True, concatena também as coordenadas originais.

    Retorna:
        tensor de forma (..., D) onde D = 4*L (+2 se include_input=True) para coordenadas 2D.
    """
    if not torch.is_tensor(coords):
        coords = torch.tensor(coords)
    dtype = coords.dtype
    device = coords.device
    # freqs: (L,)
    freqs = (2 ** torch.arange(L, dtype=dtype, device=device)) * torch.pi
    # coords: (..., 2) -> angles: (..., L, 2)
    angles = coords[..., None, :] * freqs[None, :, None]
    s = torch.sin(angles)
    c = torch.cos(angles)
    # concat sin and cos along last dim -> (..., L, 4)
    combined = torch.cat([s, c], dim=-1)
    # flatten L and feature dims -> (..., 4*L)
    out = combined.view(*coords.shape[:-1], -1)
    if include_input:
        out = torch.cat([coords, out], dim=-1)
    return out


def reshape_lenslet(img_chw: torch.Tensor, micro: int = 11) -> torch.Tensor:
    """(C, Htot, Wtot) -> (C, Hm, Wm, U, V), com U=V=micro; recorta bordas para múltiplos de micro."""
    C, Htot, Wtot = img_chw.shape
    Hm, Wm = Htot // micro, Wtot // micro
    if Hm == 0 or Wm == 0:
        raise ValueError(f"Imagem muito pequena para micro={micro}: {(C, Htot, Wtot)}")
    Hc, Wc = Hm * micro, Wm * micro
    img_crop = img_chw[:, :Hc, :Wc].contiguous()
    lf = img_crop.view(C, Hm, micro, Wm, micro).permute(0, 1, 3, 2, 4).contiguous()
    return lf  # (C, Hm, Wm, U, V)


class LensletMicroDataset(Dataset):
    """Amostra MI-wise: (PE(x,y), micro-imagem GT) do LF lenslet (C,Hm,Wm,U,V)."""
    def __init__(self, lf_chwuv: torch.Tensor, L: int, include_input: bool):
        self.lf = lf_chwuv
        self.C, self.Hm, self.Wm, self.U, self.V = lf_chwuv.shape
        self.L = L
        self.include_input = include_input
        xs = torch.linspace(0.0, 1.0, self.Wm, dtype=lf_chwuv.dtype)
        ys = torch.linspace(0.0, 1.0, self.Hm, dtype=lf_chwuv.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.coords = torch.stack([gx, gy], dim=-1)  # (Hm, Wm, 2)
        pe0 = positional_encoding(self.coords[0, 0], L=L, include_input=include_input)
        self.in_dim = int(pe0.numel())

    def __len__(self) -> int:
        return self.Hm * self.Wm

    def __getitem__(self, idx: int):
        i = idx // self.Wm  # y
        j = idx % self.Wm   # x
        pe = positional_encoding(self.coords[i, j], L=self.L, include_input=self.include_input).reshape(-1)  # (in_dim,)
        micro = self.lf[:, i, j, :, :]  # (C, U, V)
        return pe, micro


def assemble_mosaic(pred_mis: torch.Tensor, Hm: int, Wm: int) -> torch.Tensor:
    """pred_mis: (Hm*Wm, C, U, V) -> mosaic (C, Hm*U, Wm*V)"""
    N, C, U, V = pred_mis.shape
    mosaic = torch.zeros(C, Hm * U, Wm * V, dtype=pred_mis.dtype)
    idx = 0
    for i in range(Hm):
        for j in range(Wm):
            if idx >= N: break
            y0, x0 = i * U, j * V
            mosaic[:, y0:y0+U, x0:x0+V] = pred_mis[idx]
            idx += 1
    return mosaic


def psnr_torch(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """PSNR entre imagens (C,H,W) em [0,1]. Retorna escalar tensor."""
    mse = F.mse_loss(pred, target, reduction='mean')
    return 10.0 * torch.log10((max_val * max_val) / (mse + 1e-12))

def _gaussian_kernel2d(ksize: int = 11, sigma: float = 1.5, device=None, dtype=None) -> torch.Tensor:
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    g1 = torch.exp(-0.5 * (ax / sigma) ** 2)
    g1 = g1 / g1.sum()
    g2 = g1[:, None] * g1[None, :]
    g2 = g2 / g2.sum()
    return g2

def ssim_torch(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0, ksize: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM entre (C,H,W) ou (B,C,H,W) em [0,1]. Retorna escalar tensor (média no batch e canais)."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if y.dim() == 3:
        y = y.unsqueeze(0)
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    kernel = _gaussian_kernel2d(ksize, sigma, device=device, dtype=dtype)
    kernel = kernel.expand(C, 1, ksize, ksize).contiguous()  # (C,1,K,K)

    pad = ksize // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    y_pad = F.pad(y, (pad, pad, pad, pad), mode='reflect')

    mu_x = F.conv2d(x_pad, kernel, groups=C)
    mu_y = F.conv2d(y_pad, kernel, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x = F.conv2d(x_pad * x_pad, kernel, groups=C) - mu_x2
    sigma_y = F.conv2d(y_pad * y_pad, kernel, groups=C) - mu_y2
    sigma_xy = F.conv2d(x_pad * y_pad, kernel, groups=C) - mu_xy

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sigma_x + sigma_y + C2)
    ssim_map = numerator / (denominator + 1e-12)
    return ssim_map.mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default="/home/seidy/lucas/MINL/Dataset/Original_11x11_center/Ankylosaurus.png",
        help="Caminho para a imagem (aceita /mnt/ WSL ou caminho Windows)",
    )
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W","H"), default=None,
                        help="redimensionar para (W H); omita para usar resolução original")
    parser.add_argument("--L", type=int, default=10, help="número de bandas para position encoding")
    parser.add_argument("--include-input", action="store_true", help="concatena coordenadas originais ao encoding")
    parser.add_argument("--alpha", type=float, default=1e-4, help="coeficiente alpha para regularização L1")
    parser.add_argument("--epochs", type=int, default=250, help="número de épocas de treino")
    parser.add_argument("--batch-size", type=int, default=5000, help="tamanho do minibatch (patches)")
    parser.add_argument("--lr-start", type=float, default=1e-2, help="learning rate inicial para Adam")
    parser.add_argument("--lr-end", type=float, default=1e-4, help="learning rate final ao fim do treino")
    parser.add_argument("--save-recon", action="store_true", help="salvar reconstruções (centro das micro-imagens) a cada --save-interval épocas")
    parser.add_argument("--save-interval", type=int, default=10, help="intervalo (épocas) entre salvamentos quando --save-recon ativado")
    parser.add_argument("--save-dir", type=str, default="output/Ankylosaurus_PTQ/", help="diretório para salvar imagens reconstruídas (padrão: mesmo diretório da imagem)")
    parser.add_argument("--num-workers", type=int, default=0, help="workers do DataLoader (Windows: 0 ou 2)")
    parser.add_argument("--micro-size", type=int, default=11, help="tamanho da micro-imagem (U=V)")
    # ---- Quantização PTQ AIMET ----
    parser.add_argument("--ptq-enable", action="store_true", help="Ativa quantização PTQ (AIMET) após o treino base")
    parser.add_argument("--ptq-calib-batches", type=int, default=10, help="Número de batches usados na calibração compute_encodings")
    parser.add_argument("--ptq-act-bits", type=int, default=8, help="Bits para ativações")
    parser.add_argument("--ptq-wt-bits", type=int, default=8, help="Bits para pesos")
    parser.add_argument("--ptq-quant-scheme", type=str, default="tf", choices=["tf","tf_enhanced","percentile"], help="QuantScheme AIMET")
    parser.add_argument("--ptq-percentile", type=float, default=99.9, help="Percentil (se scheme=percentile)")
    parser.add_argument("--ptq-output-dir", type=str, default="quant_export", help="Diretório para exportar modelo/encodings quantizados")
    parser.add_argument("--ptq-eval-mosaic", action="store_true", help="Gera mosaico usando modelo quantizado pós-calibração")
    args = parser.parse_args()

    # enforce single save_interval int used everywhere
    save_interval = int(getattr(args, "save_interval", 10))

    img_path = args.image
    tensor = load_image(img_path, resize=tuple(args.resize) if args.resize else None)
    print(f"Imagem carregada: {img_path}")
    print(f"Tensor shape: {tuple(tensor.shape)}  dtype={tensor.dtype}")

    # Diretório da imagem para fallback de salvamento
    folder = str(Path(normalize_path(img_path)).parent)

    # Reorganiza lenslet -> LF (C,Hm,Wm,U,V)
    lf = reshape_lenslet(tensor, micro=int(args.micro_size))
    C, Hm, Wm, U, V = lf.shape
    print(f"Light field lenslet -> (C,Hm,Wm,U,V) = {(C,Hm,Wm,U,V)}")

    # Mosaico ground-truth (para métricas PSNR/SSIM)
    gt_mis = lf.permute(1, 2, 0, 3, 4).contiguous().view(Hm * Wm, C, U, V)  # (Hm*Wm, C, U, V)
    mosaic_gt = assemble_mosaic(gt_mis, Hm=Hm, Wm=Wm).clamp(0.0, 1.0)       # (C, Hm*U, Wm*V)

    # Dataset MI-wise
    lenslet_ds = LensletMicroDataset(lf, L=args.L, include_input=args.include_input)
    print(f"Dataset MI-wise: N={len(lenslet_ds)} (Hm={Hm}, Wm={Wm})  in_dim={lenslet_ds.in_dim}  micro=(C={C},{U},{V})")

    # modelos
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_dim = lenslet_ds.in_dim
    out_patch_dim = C * U * V
    mlp = SimpleMLP(in_dim=in_dim, hidden=512, out_dim=out_patch_dim).to(device)
    cnn = SmallCNN(in_channels=C, hidden=64).to(device)

    # Wrapper para AIMET combinar mlp -> reshape -> cnn
    class CombinedModel(nn.Module):
        def __init__(self, mlp: nn.Module, cnn: nn.Module, C: int, U: int, V: int):
            super().__init__()
            self.mlp = mlp
            self.cnn = cnn
            self.C = C; self.U = U; self.V = V
        def forward(self, x: torch.Tensor):
            out = self.mlp(x).view(-1, self.C, self.U, self.V)
            out = self.cnn(out)
            return out
    combined = CombinedModel(mlp, cnn, C, U, V).to(device)

    # DataLoader
    from torch.utils.data import DataLoader
    pin_memory = (device.type == 'cuda')
    loader = DataLoader(
        lenslet_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # Print trainable parameter counts (MLP, CNN, total)
    def _count_params(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    mlp_params = _count_params(mlp)
    cnn_params = _count_params(cnn)
    total_params = mlp_params + cnn_params
    print(f"Trainable parameters: mlp={mlp_params:,}  cnn={cnn_params:,}  total={total_params:,}")

    # otimizador e scheduler linear do lr_start -> lr_end em 'epochs'
    lr_start = float(args.lr_start)
    lr_end = float(args.lr_end)
    optimizer = torch.optim.Adam(list(mlp.parameters()) + list(cnn.parameters()), lr=lr_start)
    epochs = int(args.epochs)
    def lr_lambda(epoch_idx: int):
        if epochs <= 1:
            return lr_end / lr_start
        return 1.0 - (1.0 - (lr_end / lr_start)) * (epoch_idx / max(1, epochs - 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    alpha = float(args.alpha)

    print(f"Iniciando treino: samples={len(lenslet_ds)} epochs={epochs} batch_size={args.batch_size} lr_start={lr_start} lr_end={lr_end} alpha={alpha}")
    loss_history: list[float] = []
    entropy_history: list[float] = []
    psnr_history: list[float] = []   # <-- ADICIONE
    ssim_history: list[float] = []   # <-- ADICIONE

    # Salvar mosaico da época 0 (antes do treinamento)
    try:
        raw_save_dir = args.save_dir or folder
        save_dir = normalize_path(raw_save_dir) if isinstance(raw_save_dir, str) else folder
        os.makedirs(save_dir, exist_ok=True)
        recon_loader = DataLoader(lenslet_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=pin_memory)
        preds = []
        mlp.eval(); cnn.eval()
        with torch.no_grad():
            for xb_batch, _ in recon_loader:
                xb_batch = xb_batch.to(device, non_blocking=pin_memory)
                pred_batch = cnn(mlp(xb_batch).view(-1, C, U, V))
                preds.append(pred_batch.cpu())
        mlp.train(); cnn.train()
        preds = torch.cat(preds, dim=0)  # (Hm*Wm, C, U, V)
        mosaic = assemble_mosaic(preds, Hm=Hm, Wm=Wm)
        pil = transforms.ToPILImage()(mosaic)
        out_path = os.path.join(save_dir, "mosaic_ep0.png")
        pil.save(out_path)
        print(f"Salvo mosaico inicial (época 0) em: {out_path}")

        # Métricas iniciais
        try:
            psnr0 = psnr_torch(mosaic.clamp(0,1), mosaic_gt).item()
            ssim0 = ssim_torch(mosaic.clamp(0,1), mosaic_gt).item()
            print(f"Ep 0: PSNR={psnr0:.3f} dB | SSIM={ssim0:.4f}")
        except Exception as e:
            print(f"Falha ao calcular PSNR/SSIM na época 0: {e}")
    except Exception as e:
        print(f"Falha ao salvar mosaico da época 0 em '{save_dir}': {e}")
        import traceback; traceback.print_exc()

    for ep in range(epochs):
        running_loss = 0.0
        it = 0
        total_entropy = 0.0
        total_samples_entropy = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=pin_memory)
            yb = yb.to(device, non_blocking=pin_memory)  # (B, C, U, V)
            optimizer.zero_grad()
            mlp_out = mlp(xb)                      # (B, C*U*V)
            mlp_micro = mlp_out.view(-1, C, U, V)  # (B, C, U, V)
            pred = cnn(mlp_micro)                  # (B, C, U, V)

            diff = pred - yb
            loss_l2 = diff.view(diff.shape[0], -1).pow(2).sum(dim=1).mean()

            params = list(mlp.parameters()) + list(cnn.parameters())
            l1_reg = sum(p.abs().sum() for p in params)
            loss = loss_l2 + alpha * l1_reg

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            it += 1

            with torch.no_grad():
                # Cross-entropy entre GT (como distribuição) e predição (softmax)
                # Flatten para (B, N), N = C*U*V
                pred_flat = pred.detach().view(pred.shape[0], -1)   # logits
                tgt_flat  = yb.detach().view(yb.shape[0],  -1)      # intensidades GT

                # q: distribuição predita (softmax); p: GT normalizado (não-negativo)
                q = torch.softmax(pred_flat, dim=1)                 # (B, N)
                p_pos = torch.relu(tgt_flat) + 1e-12
                p = p_pos / p_pos.sum(dim=1, keepdim=True)

                # H(p, q) em bits
                batch_xent = - (p * q.clamp_min(1e-12).log2()).sum(dim=1).mean().item()
                total_entropy += batch_xent * pred.shape[0]
                total_samples_entropy += pred.shape[0]

        scheduler.step()
        avg_loss = running_loss / max(1, it)
        avg_entropy = total_entropy / total_samples_entropy if total_samples_entropy > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {ep+1}/{epochs}  avg_loss={avg_loss:.6e}  loss_l2={loss_l2.item():.6e}  l1_reg={l1_reg.item():.6e}  lr={current_lr:.6e}")
        
        # record histories
        loss_history.append(avg_loss)
        entropy_history.append(avg_entropy)

        # salvar reconstrução a cada intervalo: mosaico completo Hm*U x Wm*V
        current_psnr, current_ssim = float('nan'), float('nan')  # valores desta época
        if ((ep+1) % save_interval == 0):
            raw_save_dir = args.save_dir or folder
            save_dir = normalize_path(raw_save_dir) if isinstance(raw_save_dir, str) else folder
            print(f"Salvando mosaico (época {ep+1}) em {save_dir} ...")
            try:
                os.makedirs(save_dir, exist_ok=True)
                recon_loader = DataLoader(lenslet_ds, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers, pin_memory=pin_memory)
                preds = []
                mlp.eval(); cnn.eval()
                with torch.no_grad():
                    for xb_batch, _ in recon_loader:
                        xb_batch = xb_batch.to(device, non_blocking=pin_memory)
                        pred_batch = cnn(mlp(xb_batch).view(-1, C, U, V))
                        preds.append(pred_batch.cpu())
                mlp.train(); cnn.train()
                preds = torch.cat(preds, dim=0)  # (Hm*Wm, C, U, V)
                mosaic_pred = assemble_mosaic(preds, Hm=Hm, Wm=Wm).clamp(0.0, 1.0)  # (C, Hm*U, Wm*V)

                # métricas desta época
                current_psnr = psnr_torch(mosaic_pred, mosaic_gt).item()
                current_ssim = ssim_torch(mosaic_pred, mosaic_gt).item()
                print(f"Ep {ep+1}: PSNR={current_psnr:.3f} dB | SSIM={current_ssim:.4f}")

                pil = transforms.ToPILImage()(mosaic_pred)
                out_path = os.path.join(save_dir, f"mosaic_ep{ep+1}.png")
                pil.save(out_path)
            except Exception as e:
                print(f"Erro ao salvar mosaico na época {ep+1} em '{save_dir}': {e}")
                import traceback
                traceback.print_exc()

        # atualiza históricos PSNR/SSIM (NaN quando não salvo nesta época)
        psnr_history.append(current_psnr)
        ssim_history.append(current_ssim)

        # save metric plots every epoch (agora com PSNR/SSIM)
        raw_save_dir = args.save_dir or folder
        save_dir = normalize_path(raw_save_dir) if isinstance(raw_save_dir, str) else folder
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            print(f"Falha ao criar diretório de salvamento '{save_dir}': {e}")
            import traceback; traceback.print_exc()

        try:
            fig, axes = plt.subplots(4, 1, figsize=(7, 12))
            epochs_axis = range(1, len(loss_history) + 1)

            axes[0].plot(epochs_axis, loss_history, '-o')
            axes[0].set_title('Training loss per epoch')
            axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')

            axes[1].plot(epochs_axis, entropy_history, '-o', color='tab:orange')
            axes[1].set_title('Cross-entropy (GT vs pred) per epoch')
            axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Cross-entropy (bits)')

            axes[2].plot(epochs_axis, psnr_history, '-o', color='tab:green')
            axes[2].set_title('PSNR per epoch (mosaic)'); axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('PSNR (dB)')

            axes[3].plot(epochs_axis, ssim_history, '-o', color='tab:red')
            axes[3].set_title('SSIM per epoch (mosaic)'); axes[3].set_xlabel('Epoch'); axes[3].set_ylabel('SSIM')

            plt.tight_layout()
            metrics_path = os.path.join(save_dir, 'training_metrics.png')
            fig.savefig(metrics_path)
            plt.close(fig)
        except Exception as e:
            print(f"Falha ao salvar plots de métricas em '{save_dir}': {e}")
            import traceback; traceback.print_exc()

    print("Treino concluído")

    # ================= PTQ (AIMET) =================
    if args.ptq_enable:
        if not _AIMET_AVAILABLE:
            print("[PTQ] AIMET não está instalado. Pule ou instale: pip install aimet-torch (ver doc oficial).")
        else:
            print("[PTQ] Iniciando fluxo de quantização pós-treino (PTQ) AIMET... - Bits pesos: {}  ativações: {}  scheme: {}".format(args.ptq_wt_bits, args.ptq_act_bits, args.ptq_quant_scheme))
            quant_scheme_map = {
                'tf': QuantScheme.post_training_tf,
                'tf_enhanced': QuantScheme.post_training_tf_enhanced,
                'percentile': QuantScheme.post_training_percentile
            }
            qscheme = quant_scheme_map[args.ptq_quant_scheme]
            sim = QuantizationSimModel(
                model=combined,
                quant_scheme=qscheme,
                default_output_bw=args.ptq_act_bits,
                default_param_bw=args.ptq_wt_bits,
                dummy_input=torch.randn(2, lenslet_ds.in_dim, device=device)
            )
            if args.ptq_quant_scheme == 'percentile':
                # Configurar percentil (ativações) conforme documentação AIMET
                for qc in sim.quant_wrappers():
                    for quantizer in qc.output_quantizers:
                        quantizer.is_percentile_enabled = True
                        quantizer.percentile_value = args.ptq_percentile
            # Função de calibração
            from torch.utils.data import DataLoader
            calib_loader = DataLoader(lenslet_ds, batch_size=min(args.batch_size, 1024), shuffle=False,
                                      num_workers=args.num_workers, pin_memory=pin_memory)
            def forward_pass(model, _: Optional[None] = None):
                model.eval()
                batches = 0
                with torch.no_grad():
                    for xb_calib, _ in calib_loader:
                        xb_calib = xb_calib.to(device, non_blocking=pin_memory)
                        _ = model(xb_calib)
                        batches += 1
                        if batches >= args.ptq_calib_batches:
                            break
            print(f"[PTQ] Calibrando encodings usando até {args.ptq_calib_batches} batches...")
            sim.compute_encodings(forward_pass, None)
            print("[PTQ] Encodings computados.")

            # Avaliação opcional do modelo quantizado
            if args.ptq_eval_mosaic:
                print("[PTQ] Gerando mosaico quantizado para métricas...")
                try:
                    recon_loader = DataLoader(lenslet_ds, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=pin_memory)
                    preds_q = []
                    sim.model.eval()
                    with torch.no_grad():
                        for xb_batch, _ in recon_loader:
                            xb_batch = xb_batch.to(device, non_blocking=pin_memory)
                            pred_batch = sim.model(xb_batch).cpu()
                            preds_q.append(pred_batch)
                    preds_q = torch.cat(preds_q, dim=0)
                    mosaic_q = assemble_mosaic(preds_q.view(Hm*Wm, C, U, V), Hm=Hm, Wm=Wm).clamp(0,1)
                    try:
                        psnr_q = psnr_torch(mosaic_q, mosaic_gt).item()
                        ssim_q = ssim_torch(mosaic_q, mosaic_gt).item()
                        print(f"[PTQ] Quantizado: PSNR={psnr_q:.3f} dB | SSIM={ssim_q:.4f}")
                    except Exception as e:
                        print(f"[PTQ] Falha métricas quantizadas: {e}")
                    q_dir = normalize_path(args.ptq_output_dir)
                    os.makedirs(q_dir, exist_ok=True)
                    out_q_img = os.path.join(q_dir, "mosaic_quant.png")
                    transforms.ToPILImage()(mosaic_q).save(out_q_img)
                    print(f"[PTQ] Mosaico quantizado salvo em: {out_q_img}")
                    # Salvar métricas em JSON + gráfico comparativo
                    try:
                        import json
                        metrics_json = {
                            "psnr_history_float": psnr_history,
                            "ssim_history_float": ssim_history,
                            "psnr_quant": psnr_q if 'psnr_q' in locals() else None,
                            "ssim_quant": ssim_q if 'ssim_q' in locals() else None,
                            "epochs": len(psnr_history)
                        }
                        with open(os.path.join(q_dir, 'ptq_metrics.json'), 'w') as f_json:
                            json.dump(metrics_json, f_json, indent=2)
                        # Gráfico comparativo PSNR / SSIM
                        try:
                            fig2, axes2 = plt.subplots(2, 1, figsize=(6, 8))
                            epochs_axis = list(range(1, len(psnr_history)+1))
                            axes2[0].plot(epochs_axis, psnr_history, '-o', label='Float (treino)')
                            if 'psnr_q' in locals():
                                axes2[0].axhline(psnr_q, color='r', linestyle='--', label=f'Quant ({psnr_q:.2f} dB)')
                            axes2[0].set_title('PSNR Float vs Quantizado')
                            axes2[0].set_xlabel('Epoch'); axes2[0].set_ylabel('PSNR (dB)'); axes2[0].legend()
                            axes2[1].plot(epochs_axis, ssim_history, '-o', label='Float (treino)', color='tab:orange')
                            if 'ssim_q' in locals():
                                axes2[1].axhline(ssim_q, color='r', linestyle='--', label=f'Quant ({ssim_q:.4f})')
                            axes2[1].set_title('SSIM Float vs Quantizado')
                            axes2[1].set_xlabel('Epoch'); axes2[1].set_ylabel('SSIM'); axes2[1].legend()
                            plt.tight_layout()
                            comp_path = os.path.join(q_dir, 'ptq_comparison.png')
                            fig2.savefig(comp_path)
                            plt.close(fig2)
                            print(f"[PTQ] Gráfico comparativo salvo em: {comp_path}")
                        except Exception as e:
                            print(f"[PTQ] Falha ao gerar gráfico comparativo: {e}")
                    except Exception as e:
                        print(f"[PTQ] Falha ao salvar JSON de métricas PTQ: {e}")
                except Exception as e:
                    print(f"[PTQ] Erro ao gerar mosaico quantizado: {e}")

            # Export
            export_dir = normalize_path(args.ptq_output_dir)
            os.makedirs(export_dir, exist_ok=True)
            try:
                sim.export(path=export_dir, filename_prefix="minl_ptq", dummy_input=torch.randn(1, lenslet_ds.in_dim, device=device))
                print(f"[PTQ] Artefatos exportados para: {export_dir}")
            except Exception as e:
                print(f"[PTQ] Falha ao exportar artefatos: {e}")

    # If user asked to save reconstructions but the final epoch wasn't on a save interval,
    # save one final reconstruction now so the user always gets the last prediction.
    if args.save_recon:
        # if the last epoch was already saved (e.g. epochs % save_interval == 0) we skip
        try:
            save_interval = int(args.save_interval)
        except Exception:
            save_interval = 1
        saved_on_last = (epochs % save_interval) == 0
        if not saved_on_last:
            raw_save_dir = args.save_dir or folder
            save_dir = normalize_path(raw_save_dir) if isinstance(raw_save_dir, str) else folder
            print(f"Salvando reconstrução final (época {epochs}) em {save_dir} ...")
            try:
                os.makedirs(save_dir, exist_ok=True)
                recon_loader = DataLoader(lenslet_ds, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers, pin_memory=pin_memory)
                preds = []
                mlp.eval(); cnn.eval()
                with torch.no_grad():
                    for xb_batch, _ in recon_loader:
                        xb_batch = xb_batch.to(device, non_blocking=pin_memory)
                        pred_batch = cnn(mlp(xb_batch).view(-1, C, U, V))
                        preds.append(pred_batch.cpu())
                mlp.train(); cnn.train()
                preds = torch.cat(preds, dim=0)
                mosaic = assemble_mosaic(preds, Hm=Hm, Wm=Wm).clamp(0.0, 1.0)
                # métricas finais
                try:
                    psnrF = psnr_torch(mosaic, mosaic_gt).item()
                    ssimF = ssim_torch(mosaic, mosaic_gt).item()
                    print(f"Final: PSNR={psnrF:.3f} dB | SSIM={ssimF:.4f}")
                except Exception as e:
                    print(f"Falha ao calcular PSNR/SSIM (final): {e}")
                # salvar imagem
                pil = transforms.ToPILImage()(mosaic)
                out_path = os.path.join(save_dir, f"mosaic_final_ep{epochs}.png")
                pil.save(out_path)
                if os.path.isfile(out_path):
                    sz = os.path.getsize(out_path)
                    print(f"Reconstrução final salva: {out_path} (size={sz} bytes)")
                else:
                    print(f"Aviso: arquivo de reconstrução final não encontrado após salvar em '{out_path}'")
            except Exception as e:
                print(f"Erro ao salvar reconstrução final em '{save_dir}': {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()