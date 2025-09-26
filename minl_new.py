from __future__ import annotations
import argparse
from pathlib import Path
import os
import platform
import sys
import subprocess
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
    from aimet_common.defs import QuantScheme
    from aimet_torch.nn import QuantizationMixin
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
    """Aplica Position Encoding tipo Fourier às coordenadas, robusto a overflow.

    - Para L alto (ex.: 250), 2**k em float32 pode estourar (Inf) e gerar NaN em sin/cos.
      Para evitar isso, realizamos o cálculo das frequências e dos ângulos em float64 e
      depois convertemos de volta para o dtype original.

    Args:
        coords: tensor de forma (..., 2) com coordenadas normalizadas em [0,1].
        L: número de bandas/frequências (k = 0..L-1, frequência = 2^k * pi).
        include_input: se True, concatena também as coordenadas originais.

    Retorna:
        tensor de forma (..., D) onde D = 4*L (+2 se include_input=True) para coordenadas 2D.
    """
    if not torch.is_tensor(coords):
        coords = torch.tensor(coords)
    orig_dtype = coords.dtype
    device = coords.device

    # Cálculo em float64 para evitar overflow de 2**k em float32
    coords64 = coords.to(torch.float64)
    try:
        freqs64 = (2.0 ** torch.arange(L, dtype=torch.float64, device=device)) * torch.pi.to(torch.float64)
    except Exception:
        # Fallback simples caso torch.pi não suporte to(float64) em alguma build
        import math as _math
        freqs64 = (2.0 ** torch.arange(L, dtype=torch.float64, device=device)) * _math.pi

    # coords: (..., 2) -> angles: (..., L, 2) em float64
    angles64 = coords64[..., None, :] * freqs64[None, :, None]
    s64 = torch.sin(angles64)
    c64 = torch.cos(angles64)

    # Concatena e volta ao dtype original
    combined64 = torch.cat([s64, c64], dim=-1)  # (..., L, 4)
    combined = combined64.to(orig_dtype)

    # Flatten L e features -> (..., 4*L)
    out = combined.view(*coords.shape[:-1], -1)
    if include_input:
        out = torch.cat([coords, out], dim=-1)
    # Aviso opcional para L muito alto (apenas uma vez por processo)
    if L >= 128:
        # Use um guard de módulo para avisar apenas uma vez
        g = globals()
        if '_PE_HIGH_L_WARNED' not in g:
            g['_PE_HIGH_L_WARNED'] = False
        if not g['_PE_HIGH_L_WARNED']:
            print(f"[PE][WARN] L={L} é bastante alto; o encoding é calculado em float64 para evitar overflow, o que pode aumentar custo.")
            g['_PE_HIGH_L_WARNED'] = True
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

# Modelo combinado em escopo de módulo (pickleável) para AIMET/PTQ
class CombinedModel(nn.Module):
    def __init__(self, mlp: nn.Module, cnn: nn.Module, C: int, U: int, V: int):
        super().__init__()
        self.mlp = mlp
        self.cnn = cnn
        self.C = C
        self.U = U
        self.V = V

    def forward(self, x: torch.Tensor):
        out = self.mlp(x).view(-1, self.C, self.U, self.V)
        out = self.cnn(out)
        return out

# Implementações quantizadas para AIMET
if _AIMET_AVAILABLE:
    from networks.mlp import Sine as SineMLP
    from networks.cnn import Sine as SineCNN
    
    @QuantizationMixin.implements(SineMLP)
    class QuantizedSineMLP(QuantizationMixin, SineMLP):
        def __quant_init__(self):
            super().__quant_init__()
            # Declare the number of input/output quantizers
            self.input_quantizers = torch.nn.ModuleList([None])
            self.output_quantizers = torch.nn.ModuleList([None])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Quantize input tensors
            if self.input_quantizers[0]:
                x = self.input_quantizers[0](x)

            # Run forward with quantized inputs and parameters
            with self._patch_quantized_parameters():
                ret = super().forward(x)

            # Quantize output tensors
            if self.output_quantizers[0]:
                ret = self.output_quantizers[0](ret)

            return ret

    @QuantizationMixin.implements(SineCNN)
    class QuantizedSineCNN(QuantizationMixin, SineCNN):
        def __quant_init__(self):
            super().__quant_init__()
            # Declare the number of input/output quantizers
            self.input_quantizers = torch.nn.ModuleList([None])
            self.output_quantizers = torch.nn.ModuleList([None])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Quantize input tensors
            if self.input_quantizers[0]:
                x = self.input_quantizers[0](x)

            # Run forward with quantized inputs and parameters
            with self._patch_quantized_parameters():
                ret = super().forward(x)

            # Quantize output tensors
            if self.output_quantizers[0]:
                ret = self.output_quantizers[0](ret)

            return ret

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default="/home/seidy/lucas/MINL/Dataset/Original_11x11_center/Black_Fence.png",
        help="Caminho para a imagem (aceita /mnt/ WSL ou caminho Windows)",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Executa o pipeline para TODAS as imagens da pasta (png/jpg/jpeg/bmp/tif/tiff). Cada imagem terá diretórios de saída próprios.",
    )
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W","H"), default=None,
                        help="redimensionar para (W H); omita para usar resolução original")
    parser.add_argument("--L", type=int, default=250, help="número de bandas para position encoding")
    parser.add_argument("--include-input", action="store_true", help="concatena coordenadas originais ao encoding")
    parser.add_argument("--alpha", type=float, default=1e-4, help="coeficiente alpha para regularização L1")
    parser.add_argument("--epochs", type=int, default= 250, help="número de épocas de treino")
    parser.add_argument("--batch-size", type=int, default=5000, help="tamanho do minibatch (patches)")
    parser.add_argument("--lr-start", type=float, default=1e-2, help="learning rate inicial para Adam")
    parser.add_argument("--lr-end", type=float, default=1e-4, help="learning rate final ao fim do treino")
    parser.add_argument("--save-recon", action="store_true", help="salvar reconstruções (centro das micro-imagens) a cada --save-interval épocas")
    parser.add_argument("--save-interval", type=int, default=10, help="intervalo (épocas) entre salvamentos quando --save-recon ativado")
    parser.add_argument("--save-dir", type=str, default="output/teste/", help="diretório para salvar imagens reconstruídas (padrão: mesmo diretório da imagem)")
    parser.add_argument("--num-workers", type=int, default=0, help="workers do DataLoader (Windows: 0 ou 2)")
    parser.add_argument("--micro-size", type=int, default=11, help="tamanho da micro-imagem (U=V)")
    # Arquitetura MLP/CNN
    parser.add_argument("--mlp-hidden", type=int, default=128, help="tamanho do hidden do MLP")
    parser.add_argument("--mlp-layers", type=int, default=1, help="número de camadas ocultas do MLP")
    parser.add_argument("--cnn-hidden", type=int, default=64, help="número de canais ocultos da CNN")
    parser.add_argument("--cnn-blocks", type=int, default=2, help="número de blocos conv (3x3) na CNN")
    parser.add_argument("--cnn-out-activation", action="store_true", help="aplica ativação Sine na saída da CNN")
    # Reprodutibilidade e checkpoints
    parser.add_argument("--seed", type=int, default=13, help="Seed global para inicialização (torch, numpy, random)")
    parser.add_argument("--save-ckpt", action="store_true", help="Salvar checkpoints por época (pesos e otimizador)")
    parser.add_argument("--ckpt-interval", type=int, default=10, help="Intervalo de épocas para checkpoint")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/teste/", help="Diretório para salvar checkpoints")
    parser.add_argument("--log-file", type=str, default="log_test.txt", help="Arquivo de log para salvar métricas e eventos")
    # Estabilidade numérica
    parser.add_argument("--nan-guard", action="store_true", help="Detecta e corrige NaNs/Infs durante o treino e na avaliação de métricas")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Se > 0, aplica clipping de gradiente pelo norm (ex: 1.0)")
    parser.add_argument("--debug-nan", action="store_true", help="Ativa detecção de anomalias do autograd e logs quando aparecerem NaN/Inf")
    parser.add_argument("--resume-latest", action="store_true", help="Se habilitado, tenta retomar do último checkpoint em --ckpt-dir")
    parser.add_argument("--resume-from", type=str, default=None, help="Caminho para um checkpoint específico (.pth) para retomar")
    # ---- Quantização PTQ AIMET ----
    parser.add_argument("--ptq-enable", action="store_true", help="Ativa quantização PTQ (AIMET) após o treino base")
    parser.add_argument("--ptq-calib-batches", type=int, default=10, help="Número de batches usados na calibração compute_encodings")
    parser.add_argument("--ptq-act-bits", type=int, default=8, help="Bits para ativações")
    parser.add_argument("--ptq-wt-bits", type=int, default=8, help="Bits para pesos")
    parser.add_argument("--ptq-quant-scheme", type=str, default="tf", choices=["tf","tf_enhanced","percentile"], help="QuantScheme AIMET")
    parser.add_argument("--ptq-percentile", type=float, default=99.9, help="Percentil (se scheme=percentile)")
    parser.add_argument("--ptq-output-dir", type=str, default="output/teste_PTQ/", help="Diretório para exportar modelo/encodings quantizados")
    parser.add_argument("--ptq-eval-mosaic", action="store_true", help="Gera mosaico usando modelo quantizado pós-calibração")
    parser.add_argument("--ptq-eval-per-epoch", action="store_true", help="(Opcional) Avalia PSNR/SSIM com PTQ durante o treino nas épocas em que salvamos mosaicos. Por padrão PTQ roda só após o treino.")
    parser.add_argument("--ptq-sweep-checkpoints", action="store_true", help="(Opcional) Após o treino, varre checkpoints salvos e calcula PSNR/SSIM PTQ por checkpoint para gerar uma curva PTQ por época.")
    args = parser.parse_args()

    # Deprecated flag handling: PTQ is strictly post-training now; this flag is a no-op.
    if getattr(args, 'ptq_eval_per_epoch', False):
        print("[WARN] --ptq-eval-per-epoch está depreciado e não tem efeito. O PTQ ocorre apenas após o treino. Use --ptq-sweep-checkpoints para analisar por época.")

    # Execução em lote: processa todas as imagens de um diretório chamando este script recursivamente
    if args.image_dir:
        img_dir = normalize_path(args.image_dir.rstrip('}'))  # tolera '}' no final
        pdir = Path(img_dir)
        if not pdir.exists() or not pdir.is_dir():
            print(f"[MULTI-RUN] Diretório inválido: {img_dir}")
            return
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = sorted([str(f) for f in pdir.rglob("*") if f.is_file() and f.suffix.lower() in exts])
        if not files:
            print(f"[MULTI-RUN] Nenhuma imagem encontrada em: {img_dir}")
            return
        print(f"[MULTI-RUN] Encontradas {len(files)} imagens em '{img_dir}'. Iniciando processamento...\n")

        for idx, fpath in enumerate(files, 1):
            name = Path(fpath).stem
            save_dir = os.path.join("output", name)
            ckpt_dir = os.path.join("checkpoints", name)
            ptq_dir = os.path.join("output", f"{name}_PTQ")

            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--image", fpath,
                "--L", str(args.L),
                "--alpha", str(args.alpha),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--lr-start", str(args.lr_start),
                "--lr-end", str(args.lr_end),
                "--save-interval", str(int(getattr(args, "save_interval", 10))),
                "--num-workers", str(args.num_workers),
                "--micro-size", str(args.micro_size),
                "--mlp-hidden", str(args.mlp_hidden),
                "--mlp-layers", str(args.mlp_layers),
                "--cnn-hidden", str(args.cnn_hidden),
                "--cnn-blocks", str(args.cnn_blocks),
                *( ["--cnn-out-activation"] if args.cnn_out_activation else [] ),
                "--save-dir", save_dir,
                "--ckpt-dir", ckpt_dir,
            ]
            if args.resize:
                w, h = args.resize
                cmd += ["--resize", str(w), str(h)]
            if args.include_input:
                cmd += ["--include-input"]
            if args.save_recon:
                cmd += ["--save-recon"]
            if args.seed is not None:
                cmd += ["--seed", str(args.seed)]
            if args.save_ckpt:
                cmd += ["--save-ckpt", "--ckpt-interval", str(args.ckpt_interval)]
            if args.ptq_enable:
                cmd += [
                    "--ptq-enable",
                    "--ptq-calib-batches", str(args.ptq_calib_batches),
                    "--ptq-act-bits", str(args.ptq_act_bits),
                    "--ptq-wt-bits", str(args.ptq_wt_bits),
                    "--ptq-quant-scheme", str(args.ptq_quant_scheme),
                    "--ptq-percentile", str(args.ptq_percentile),
                    "--ptq-output-dir", ptq_dir,
                ]
                if args.ptq_eval_mosaic:
                    cmd += ["--ptq-eval-mosaic"]
                if args.ptq_sweep_checkpoints:
                    cmd += ["--ptq-sweep-checkpoints"]

            print(f"[MULTI-RUN] ({idx}/{len(files)}) Executando para: {fpath}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[MULTI-RUN] Falha ao processar '{fpath}': {e}")
        print("[MULTI-RUN] Finalizado processamento de todas as imagens.")
        return

    # enforce single save_interval int used everywhere
    save_interval = int(getattr(args, "save_interval", 10))

    img_path = args.image
    # Seed global
    if args.seed is not None:
        import random, numpy as np
        print(f"[SEED] Configurando seed global: {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    tensor = load_image(img_path, resize=tuple(args.resize) if args.resize else None)
    # Debug/anomalias
    if args.debug_nan:
        try:
            torch.autograd.set_detect_anomaly(True)
            print("[DEBUG-NAN] Autograd anomaly detection ativado.")
        except Exception as e:
            print(f"[DEBUG-NAN] Falha ao ativar anomaly detection: {e}")

    def _chk(name: str, t: torch.Tensor):
        if args.debug_nan:
            if not torch.isfinite(t).all():
                num_bad = (~torch.isfinite(t)).sum().item()
                print(f"[DEBUG-NAN] {name} possui {num_bad} valores não finitos.")
    print(f"Imagem carregada: {img_path}")
    print(f"Tensor shape: {tuple(tensor.shape)}  dtype={tensor.dtype}")

    # Função simples para anexar mensagens em um log.txt
    def _append_log(msg: str):
        try:
            log_dir = normalize_path(getattr(args, 'save_dir', '.'))
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, args.log_file)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(str(msg).rstrip('\n') + '\n')
        except Exception:
            pass

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
    mlp = SimpleMLP(in_dim=in_dim, hidden=int(args.mlp_hidden), out_dim=out_patch_dim, num_layers=int(args.mlp_layers)).to(device)
    cnn = SmallCNN(in_channels=C, hidden=int(args.cnn_hidden), num_blocks=int(args.cnn_blocks), out_activation=bool(args.cnn_out_activation)).to(device)

    # Wrapper para AIMET combinar mlp -> reshape -> cnn
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
    print(f"\nTrainable parameters: mlp={mlp_params:,}  cnn={cnn_params:,}  Total={total_params:,}")

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

    # ====== RESUME (opcional) ======
    start_epoch = 0
    def _load_checkpoint(path: str):
        nonlocal start_epoch, loss_history, entropy_history, psnr_history, ssim_history
        print(f"[RESUME] Carregando checkpoint: {path}")
        ckpt = torch.load(path, map_location=device)
        mlp.load_state_dict(ckpt['mlp_state'])
        cnn.load_state_dict(ckpt['cnn_state'])
        try:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        except Exception as e:
            print(f"[RESUME] Aviso: não foi possível restaurar optimizer: {e}")
        try:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        except Exception as e:
            print(f"[RESUME] Aviso: não foi possível restaurar scheduler: {e}")
        # epoch salvo é 1-indexed (ep+1); retomamos do mesmo índice como zero-based loop
        start_epoch = int(ckpt.get('epoch', 0))
        # restaurar históricos (se existirem)
        loss_history = list(ckpt.get('loss_history', []))
        entropy_history = list(ckpt.get('entropy_history', []))
        psnr_history = list(ckpt.get('psnr_history', []))
        ssim_history = list(ckpt.get('ssim_history', []))
        print(f"[RESUME] Retomando a partir da época {start_epoch+1} até {epochs}")
        try:
            _append_log(f"RESUME FROM {path} START_EPOCH={start_epoch+1}")
        except Exception:
            pass

    if args.resume_from:
        _load_checkpoint(normalize_path(args.resume_from))
    elif args.resume_latest:
        try:
            import glob
            ckpt_dir = normalize_path(args.ckpt_dir)
            latest = sorted(glob.glob(os.path.join(ckpt_dir, 'minl_ep*.pth')))
            if latest:
                _load_checkpoint(latest[-1])
            else:
                print(f"[RESUME] Nenhum checkpoint encontrado em {ckpt_dir}; iniciando do zero.")
        except Exception as e:
            print(f"[RESUME] Falha ao localizar checkpoint: {e}")

    print(f"\n[Iniciando treino]\n samples={len(lenslet_ds)} epochs={epochs} batch_size={args.batch_size} lr_start={lr_start} lr_end={lr_end} alpha={alpha}")
    loss_history: list[float] = []
    entropy_history: list[float] = []
    psnr_history: list[float] = []   # <-- ADICIONE
    ssim_history: list[float] = []   # <-- ADICIONE
    # Históricos de métricas quantizadas (preenchidos nas épocas salvas)
    ptq_psnr_history: list[float] = []
    ptq_ssim_history: list[float] = []

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
                if args.nan_guard and (not torch.isfinite(pred_batch).all()):
                    print("[NAN-GUARD] Predições iniciais contêm valores não finitos; aplicando nan_to_num.")
                    pred_batch = torch.nan_to_num(pred_batch, nan=0.0, posinf=1.0, neginf=0.0)
                preds.append(pred_batch.cpu())
        mlp.train(); cnn.train()
        preds = torch.cat(preds, dim=0)  # (Hm*Wm, C, U, V)
        # sanitize and clamp to avoid NaNs/Inf in epoch-0 mosaic
        try:
            preds = torch.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
        except Exception:
            pass
        mosaic = assemble_mosaic(preds, Hm=Hm, Wm=Wm)
        try:
            mosaic = torch.nan_to_num(mosaic, nan=0.0, posinf=1.0, neginf=0.0)
        except Exception:
            pass
        mosaic = mosaic.clamp(0.0, 1.0)
        pil = transforms.ToPILImage()(mosaic)
        out_path = os.path.join(save_dir, "mosaic_ep0.png")
        pil.save(out_path)
        print(f"Salvo mosaico inicial (época 0) em: {out_path}")
        _append_log(f"EP0_SAVE {out_path}")

        # Métricas iniciais
        try:
            psnr0 = psnr_torch(mosaic.clamp(0,1), mosaic_gt).item()
            ssim0 = ssim_torch(mosaic.clamp(0,1), mosaic_gt).item()
            print(f"Ep 0: PSNR={psnr0:.3f} dB | SSIM={ssim0:.4f}")
            _append_log(f"EP0_METRIC PSNR={psnr0:.6f} SSIM={ssim0:.6f}")
        except Exception as e:
            print(f"Falha ao calcular PSNR/SSIM na época 0: {e}")
    except Exception as e:
        print(f"Falha ao salvar mosaico da época 0 em '{save_dir}': {e}")
        import traceback; traceback.print_exc()

    for ep in range(start_epoch, epochs):
        running_loss = 0.0
        it = 0
        total_entropy = 0.0
        total_samples_entropy = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=pin_memory)
            yb = yb.to(device, non_blocking=pin_memory)  # (B, C, U, V)
            optimizer.zero_grad()
            mlp_out = mlp(xb)                      # (B, C*U*V)
            _chk('mlp_out', mlp_out)
            mlp_micro = mlp_out.view(-1, C, U, V)  # (B, C, U, V)
            pred = cnn(mlp_micro)                  # (B, C, U, V)
            _chk('pred', pred)
            if args.nan_guard and (not torch.isfinite(pred).all()):
                print("[NAN-GUARD] Predições contêm NaN/Inf; aplicando nan_to_num.")
                pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
            if args.nan_guard and (not torch.isfinite(yb).all()):
                print("[NAN-GUARD] Alvos contêm NaN/Inf; aplicando nan_to_num.")
                yb = torch.nan_to_num(yb, nan=0.0, posinf=1.0, neginf=0.0)

            diff = pred - yb
            loss_l2 = diff.view(diff.shape[0], -1).pow(2).sum(dim=1).mean()

            params = list(mlp.parameters()) + list(cnn.parameters())
            l1_reg = sum(p.abs().sum() for p in params)
            loss = loss_l2 + alpha * l1_reg

            _chk('loss_before_backward', loss.detach())
            loss.backward()
            if float(args.grad_clip) and args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(list(mlp.parameters()) + list(cnn.parameters()), max_norm=float(args.grad_clip))
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
        _append_log(f"EPOCH {ep+1} avg_loss={avg_loss:.6e} loss_l2={loss_l2.item():.6e} l1_reg={l1_reg.item():.6e} lr={current_lr:.6e} xent={avg_entropy:.6f}")
        
        # record histories
        loss_history.append(avg_loss)
        entropy_history.append(avg_entropy)

        # salvar reconstrução a cada intervalo: mosaico completo Hm*U x Wm*V
        current_psnr, current_ssim = float('nan'), float('nan')  # valores desta época (float)
        current_psnr_q, current_ssim_q = float('nan'), float('nan')  # valores desta época (quantizado)
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
                        if args.nan_guard and (not torch.isfinite(pred_batch).all()):
                            print("[NAN-GUARD] Predições (época) contêm valores não finitos; aplicando nan_to_num.")
                            pred_batch = torch.nan_to_num(pred_batch, nan=0.0, posinf=1.0, neginf=0.0)
                        preds.append(pred_batch.cpu())
                mlp.train(); cnn.train()
                preds = torch.cat(preds, dim=0)  # (Hm*Wm, C, U, V)
                if args.nan_guard:
                    preds = torch.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
                mosaic_pred = assemble_mosaic(preds, Hm=Hm, Wm=Wm).clamp(0.0, 1.0)  # (C, Hm*U, Wm*V)

                # métricas desta época
                current_psnr = psnr_torch(mosaic_pred, mosaic_gt).item()
                current_ssim = ssim_torch(mosaic_pred, mosaic_gt).item()
                print(f"Ep {ep+1}: PSNR={current_psnr:.3f} dB | SSIM={current_ssim:.4f}")
                _append_log(f"EPOCH {ep+1} PSNR={current_psnr:.6f} SSIM={current_ssim:.6f}")

                pil = transforms.ToPILImage()(mosaic_pred)
                out_path = os.path.join(save_dir, f"mosaic_ep{ep+1}.png")
                pil.save(out_path)

                # (Removido) PTQ por época não é calculado aqui. PTQ é executado apenas após o treino.
            except Exception as e:
                print(f"Erro ao salvar mosaico na época {ep+1} em '{save_dir}': {e}")
                import traceback
                traceback.print_exc()

        # atualiza históricos PSNR/SSIM (NaN quando não salvo nesta época)
        psnr_history.append(current_psnr)
        ssim_history.append(current_ssim)
        ptq_psnr_history.append(current_psnr_q)
        ptq_ssim_history.append(current_ssim_q)

        # save metric plots every epoch (agora com PSNR/SSIM)
        raw_save_dir = args.save_dir or folder
        save_dir = normalize_path(raw_save_dir) if isinstance(raw_save_dir, str) else folder
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            print(f"Falha ao criar diretório de salvamento '{save_dir}': {e}")
            import traceback; traceback.print_exc()

        # Checkpoint por época
        if args.save_ckpt and ((ep+1) % int(args.ckpt_interval) == 0):
            try:
                ckpt_dir = normalize_path(args.ckpt_dir)
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt = {
                    'epoch': ep+1,
                    'mlp_state': mlp.state_dict(),
                    'cnn_state': cnn.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'args': vars(args),
                    'psnr_history': psnr_history,
                    'ssim_history': ssim_history,
                    'loss_history': loss_history,
                    'entropy_history': entropy_history,
                }
                ckpt_path = os.path.join(ckpt_dir, f"minl_ep{ep+1:04d}.pth")
                torch.save(ckpt, ckpt_path)
                print(f"[CKPT] Checkpoint salvo: {ckpt_path}")
                _append_log(f"CKPT {ep+1} {ckpt_path}")
            except Exception as e:
                print(f"[CKPT] Falha ao salvar checkpoint: {e}")
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

            axes[2].plot(epochs_axis, psnr_history, '-o', color='tab:green', label='Float')
            # série PTQ: plota somente se houver algum valor finito
            try:
                import numpy as _np
                _has_ptq_psnr = any(_np.isfinite(v) for v in ptq_psnr_history)
            except Exception:
                _has_ptq_psnr = False
            if _has_ptq_psnr:
                axes[2].plot(epochs_axis, ptq_psnr_history, 'x--', color='tab:olive', label='PTQ')
            axes[2].set_title('PSNR per epoch (mosaic)'); axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('PSNR (dB)')
            axes[2].legend(loc='best')

            axes[3].plot(epochs_axis, ssim_history, '-o', color='tab:red', label='Float')
            try:
                import numpy as _np
                _has_ptq_ssim = any(_np.isfinite(v) for v in ptq_ssim_history)
            except Exception:
                _has_ptq_ssim = False
            if _has_ptq_ssim:
                axes[3].plot(epochs_axis, ptq_ssim_history, 'x--', color='tab:pink', label='PTQ')
            axes[3].set_title('SSIM per epoch (mosaic)'); axes[3].set_xlabel('Epoch'); axes[3].set_ylabel('SSIM')
            axes[3].legend(loc='best')

            plt.tight_layout()
            metrics_path = os.path.join(save_dir, 'training_metrics.png')
            fig.savefig(metrics_path)
            _append_log(f"EPOCH {ep+1} METRICS_PLOT {metrics_path}")
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
            try:
                _append_log("PTQ ENCODINGS_DONE")
            except Exception:
                pass

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
                            if args.nan_guard and (not torch.isfinite(pred_batch).all()):
                                print("[NAN-GUARD] Predições quantizadas contêm valores não finitos; aplicando nan_to_num.")
                                pred_batch = torch.nan_to_num(pred_batch, nan=0.0, posinf=1.0, neginf=0.0)
                            preds_q.append(pred_batch)
                    preds_q = torch.cat(preds_q, dim=0)
                    if args.nan_guard:
                        preds_q = torch.nan_to_num(preds_q, nan=0.0, posinf=1.0, neginf=0.0)
                    mosaic_q = assemble_mosaic(preds_q.view(Hm*Wm, C, U, V), Hm=Hm, Wm=Wm).clamp(0,1)
                    try:
                        psnr_q = psnr_torch(mosaic_q, mosaic_gt).item()
                        ssim_q = ssim_torch(mosaic_q, mosaic_gt).item()
                        print(f"[PTQ] Quantizado: PSNR={psnr_q:.3f} dB | SSIM={ssim_q:.4f}")
                        try:
                            _append_log(f"PTQ_METRIC PSNR={psnr_q:.6f} SSIM={ssim_q:.6f}")
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"[PTQ] Falha métricas quantizadas: {e}")
                    q_dir = normalize_path(args.ptq_output_dir)
                    os.makedirs(q_dir, exist_ok=True)
                    out_q_img = os.path.join(q_dir, "mosaic_quant.png")
                    transforms.ToPILImage()(mosaic_q).save(out_q_img)
                    print(f"[PTQ] Mosaico quantizado salvo em: {out_q_img}")
                    try:
                        _append_log(f"PTQ_SAVE {out_q_img}")
                    except Exception:
                        pass
                    # Salvar métricas em JSON + gráfico comparativo
                    try:
                        import json
                        metrics_json = {
                            "psnr_history_float": psnr_history,
                            "ssim_history_float": ssim_history,
                            "psnr_history_ptq": ptq_psnr_history if 'ptq_psnr_history' in locals() else None,
                            "ssim_history_ptq": ptq_ssim_history if 'ptq_ssim_history' in locals() else None,
                            "psnr_quant_final": psnr_q if 'psnr_q' in locals() else None,
                            "ssim_quant_final": ssim_q if 'ssim_q' in locals() else None,
                            "epochs": len(psnr_history)
                        }
                        with open(os.path.join(q_dir, 'ptq_metrics.json'), 'w') as f_json:
                            json.dump(metrics_json, f_json, indent=2)
                    except Exception as e:
                        print(f"[PTQ] Falha ao salvar JSON de métricas PTQ: {e}")
                except Exception as e:
                    print(f"[PTQ] Erro ao gerar mosaico quantizado: {e}")

            # Export (garantir dispositivo consistente: exportar em CPU)
            export_dir = normalize_path(args.ptq_output_dir)
            os.makedirs(export_dir, exist_ok=True)
            try:
                prev_device = next(combined.parameters()).device
                sim.model.to('cpu')
                sim.export(path=export_dir, filename_prefix="minl_ptq", dummy_input=torch.randn(1, lenslet_ds.in_dim, device='cpu'))
                print(f"[PTQ] Artefatos exportados para: {export_dir}")
                # restaurar
                sim.model.to(prev_device)
                try:
                    _append_log(f"PTQ_EXPORT {export_dir}")
                except Exception:
                    pass
            except Exception as e:
                print(f"[PTQ] Falha ao exportar artefatos: {e}")

            # Varredura de checkpoints para curva PTQ por época (pós-treino)
            if args.ptq_sweep_checkpoints:
                try:
                    import glob, re, copy, json
                    ckpt_dir = normalize_path(args.ckpt_dir)
                    pattern = os.path.join(ckpt_dir, "minl_ep*.pth")
                    ckpt_files = sorted(glob.glob(pattern))
                    if not ckpt_files:
                        print(f"[PTQ][SWEEP] Nenhum checkpoint encontrado em '{ckpt_dir}'. Ative --save-ckpt ou ajuste --ckpt-dir.")
                    else:
                        print(f"[PTQ][SWEEP] Encontrados {len(ckpt_files)} checkpoints. Avaliando PTQ por checkpoint...")
                        # Guardar estado atual para restaurar ao final
                        state_mlp = copy.deepcopy(mlp.state_dict())
                        state_cnn = copy.deepcopy(cnn.state_dict())

                        # Preparar loaders reutilizados
                        from torch.utils.data import DataLoader
                        recon_loader = DataLoader(lenslet_ds, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=pin_memory)

                        sweep_epochs: list[int] = []
                        sweep_psnr: list[float] = []
                        sweep_ssim: list[float] = []

                        quant_scheme_map = {
                            'tf': QuantScheme.post_training_tf,
                            'tf_enhanced': QuantScheme.post_training_tf_enhanced,
                            'percentile': QuantScheme.post_training_percentile
                        }
                        qscheme_sweep = quant_scheme_map[args.ptq_quant_scheme]

                        # Regex para extrair número da época do nome do arquivo
                        rx = re.compile(r"minl_ep(\d+)\.pth$")

                        for ckpt_path in ckpt_files:
                            try:
                                m = rx.search(os.path.basename(ckpt_path))
                                ep_ckpt = int(m.group(1)) if m else None
                                print(f"[PTQ][SWEEP] Checkpoint: {ckpt_path} (epoch={ep_ckpt})")
                                data = torch.load(ckpt_path, map_location=device)
                                mlp.load_state_dict(data['mlp_state'])
                                cnn.load_state_dict(data['cnn_state'])

                                # Criar sim para este checkpoint
                                sim_ck = QuantizationSimModel(
                                    model=combined,
                                    quant_scheme=qscheme_sweep,
                                    default_output_bw=args.ptq_act_bits,
                                    default_param_bw=args.ptq_wt_bits,
                                    dummy_input=torch.randn(2, lenslet_ds.in_dim, device=device)
                                )
                                if args.ptq_quant_scheme == 'percentile':
                                    for qc in sim_ck.quant_wrappers():
                                        for quantizer in qc.output_quantizers:
                                            quantizer.is_percentile_enabled = True
                                            quantizer.percentile_value = args.ptq_percentile

                                # Calibração rápida
                                def _fp_ck(model, _=None):
                                    model.eval()
                                    batches = 0
                                    with torch.no_grad():
                                        for xb_calib, _ in recon_loader:
                                            xb_calib = xb_calib.to(device, non_blocking=pin_memory)
                                            _ = model(xb_calib)
                                            batches += 1
                                            if batches >= args.ptq_calib_batches:
                                                break
                                sim_ck.compute_encodings(_fp_ck, None)

                                # Métricas deste checkpoint (quantizado)
                                preds_q = []
                                sim_ck.model.eval()
                                with torch.no_grad():
                                    for xb_batch, _ in recon_loader:
                                        xb_batch = xb_batch.to(device, non_blocking=pin_memory)
                                        pred_batch = sim_ck.model(xb_batch).cpu()
                                        if args.nan_guard and (not torch.isfinite(pred_batch).all()):
                                            pred_batch = torch.nan_to_num(pred_batch, nan=0.0, posinf=1.0, neginf=0.0)
                                        preds_q.append(pred_batch)
                                preds_q = torch.cat(preds_q, dim=0)
                                mosaic_q = assemble_mosaic(preds_q.view(Hm*Wm, C, U, V), Hm=Hm, Wm=Wm).clamp(0,1)
                                psnr_q_ck = psnr_torch(mosaic_q, mosaic_gt).item()
                                ssim_q_ck = ssim_torch(mosaic_q, mosaic_gt).item()

                                sweep_epochs.append(ep_ckpt if ep_ckpt is not None else len(sweep_epochs)+1)
                                sweep_psnr.append(psnr_q_ck)
                                sweep_ssim.append(ssim_q_ck)

                                print(f"[PTQ][SWEEP] ep={ep_ckpt}: PSNR={psnr_q_ck:.3f} dB | SSIM={ssim_q_ck:.4f}")
                                del sim_ck
                            except Exception as e:
                                print(f"[PTQ][SWEEP] Falha no checkpoint '{ckpt_path}': {e}")

                        # Restaurar modelo original treinado
                        mlp.load_state_dict(state_mlp)
                        cnn.load_state_dict(state_cnn)

                        # Persistir resultados
                        q_dir = normalize_path(args.ptq_output_dir)
                        os.makedirs(q_dir, exist_ok=True)
                        try:
                            sweep_json = {
                                'sweep_epochs': sweep_epochs,
                                'sweep_psnr_ptq': sweep_psnr,
                                'sweep_ssim_ptq': sweep_ssim,
                            }
                            with open(os.path.join(q_dir, 'ptq_sweep.json'), 'w') as f:
                                json.dump(sweep_json, f, indent=2)
                            print(f"[PTQ][SWEEP] JSON salvo em {os.path.join(q_dir, 'ptq_sweep.json')}")
                        except Exception as e:
                            print(f"[PTQ][SWEEP] Falha ao salvar JSON: {e}")

                        # Plotar curvas: float vs PTQ (sweep)
                        try:
                            fig3, axes3 = plt.subplots(2, 1, figsize=(6, 8))
                            epochs_axis = list(range(1, len(psnr_history)+1))
                            axes3[0].plot(epochs_axis, psnr_history, '-o', label='Float (treino)')
                            axes3[0].plot(sweep_epochs, sweep_psnr, 'x--', label='PTQ (checkpoints)')
                            axes3[0].set_title('PSNR Float vs PTQ')
                            axes3[0].set_xlabel('Epoch'); axes3[0].set_ylabel('PSNR (dB)'); axes3[0].legend()
                            axes3[1].plot(epochs_axis, ssim_history, '-o', color='tab:blue', label='Float (treino)')
                            axes3[1].plot(sweep_epochs, sweep_ssim, '-o', color='tab:orange', label='PTQ')
                            axes3[1].set_title('SSIM Float vs PTQ (checkpoints)')
                            axes3[1].set_xlabel('Epoch'); axes3[1].set_ylabel('SSIM'); axes3[1].legend()
                            plt.tight_layout()
                            sweep_plot = os.path.join(q_dir, 'ptq_sweep.png')
                            fig3.savefig(sweep_plot)
                            plt.close(fig3)
                            print(f"[PTQ][SWEEP] Gráfico salvo em: {sweep_plot}")
                        except Exception as e:
                            print(f"[PTQ][SWEEP] Falha ao gerar gráfico de sweep: {e}")
                except Exception as e:
                    print(f"[PTQ][SWEEP] Erro geral na varredura de checkpoints: {e}")

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
                    try:
                        _append_log(f"FINAL_METRIC PSNR={psnrF:.6f} SSIM={ssimF:.6f}")
                    except Exception:
                        pass
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