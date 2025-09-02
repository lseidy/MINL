import argparse
import imageio
import numpy as np
import sys
from pathlib import Path

try:
    import pylightfield as plf
except ModuleNotFoundError:
    print("Error: 'pylightfield' not found in this Python environment.")
    print("If you're running in WSL, install it there (e.g. apt install python3-pip; pip install <pylightfield-src>).\n"
          "pylightfield is not available on PyPI; you may need to install it from its source repository or wheel.\n"
          "Options:\n  - Run this script inside WSL after installing the package.\n  - If you have the pylightfield repo, install with: pip install /path/to/pylightfield\n"
          "If you prefer, provide a folder with already-extracted LF views and I can adapt the script to use that instead.")
    sys.exit(1)


def normalize_path(p: str) -> str:
    # If given a WSL /mnt path but running under Windows, convert to Windows path
    if p.startswith('/mnt/') and sys.platform.startswith('win'):
        parts = p.split('/')
        drive = parts[2].upper()
        rest = parts[3:]
        return str(Path(f"{drive}:/" + "/".join(rest)))
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lfr', required=True, help='Path to .LFR file (WSL path accepted)')
    parser.add_argument('--out', required=True, help='Output PNG path')
    args = parser.parse_args()

    arquivo_lfr = args.lfr
    arquivo_png = args.out

    arquivo_lfr = normalize_path(arquivo_lfr)
    arquivo_png = normalize_path(arquivo_png)

    print(f"Tentando converter com pylightfield: {arquivo_lfr}")

    try:
        lightfield = plf.LightField.from_lfr(arquivo_lfr)
        v_center = lightfield.v // 2
        u_center = lightfield.u // 2
        imagem_central = lightfield.get_view(v_center, u_center)

        if imagem_central.dtype != np.uint8:
            imagem_central = (imagem_central * 255).astype(np.uint8)

        imageio.imwrite(arquivo_png, imagem_central)
        print(f"Sucesso! Arquivo salvo como: {arquivo_png}")

    except Exception as e:
        print(f"Ocorreu um erro ao converter: {e}")


if __name__ == '__main__':
    main()