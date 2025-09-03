import os
import sys
import time
import shlex
import subprocess
from pathlib import Path
import re

from PIL import Image
import numpy as np
import wandb


class Params:
    # minimal set of fields for wandb configuration and defaults
    wandb_active = True
    wandb_entity = 'minl'
    config_name = 'run_minl_new'
    model = 'MINL'
    dataset_name = 'Ankylosaurus'
    dataset_path = '/mnt/c/Users/lucas/Documents/Mestrado/MINL/Dataset/Original_11x11_center/Ankylosaurus.png'
    epochs = 250
    batch_size = 5000
    lr_start = 1e-2
    lr_end = 1e-4
    alpha = 1e-4



def compute_image_entropy(path: str) -> float:
    """Compute grayscale Shannon entropy of an image file (bits).
    Returns NaN if image can't be opened.
    """
    try:
        im = Image.open(path).convert('L')
    except Exception:
        return float('nan')
    a = np.array(im).ravel()
    if a.size == 0:
        return float('nan')
    hist, _ = np.histogram(a, bins=256, range=(0, 255), density=True)
    probs = hist[hist > 0]
    entropy = -(probs * np.log2(probs)).sum()
    return float(entropy)


def build_cmd(args_dict: dict) -> list:
    cmd = [sys.executable, str(Path(__file__).parent / 'minl_new.py')]
    for k, v in args_dict.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f'--{k}')
        elif isinstance(v, (list, tuple)):
            cmd.append(f'--{k}')
            cmd.extend([str(x) for x in v])
        else:
            cmd.append(f'--{k}')
            cmd.append(str(v))
    return cmd


def main():
    # Configure the CLI attributes you want passed to minl_new.py
    params = {
        'image': '/mnt/c/Users/lucas/Documents/Mestrado/MINL/Dataset/Original_11x11_center/Ankylosaurus.png',
        'resize': [64, 64],
        'L': 6,
        'include-input': False,
        'alpha': 1e-4,
        'epochs': 250,
        'batch-size': 5000,
        'lr-start': 1e-2,
        'lr-end': 1e-4,
        'save-recon': True,
        'save-interval': 10,
        # save-dir will default to image folder if omitted; you can set it here
        'save-dir': str(Path(__file__).parent / 'output_minl'),
    }

    # Initialize wandb
    wandb_entity = os.environ.get('WANDB_ENTITY')
    # Read the API key from the standard environment variable (do NOT hard-code keys)
    wandb_api = os.environ.get('WANDB_API_KEY')
    try:
        if wandb_api:
            wandb.login(key=wandb_api, force=True)
        else:
            wandb.login()
    except Exception:
        print('wandb login failed or skipped; continuing without login')

    # Choose a run name for wandb (separate from CLI params passed to minl_new)
    run_name = 'run_minl_ankylosaurus'
    run = wandb.init(
        project='MINL',
        name=run_name,
        entity=wandb_entity,
        config=params,
    )

    cmd = build_cmd(params)
    print('Running:', ' '.join(shlex.quote(x) for x in cmd))

    # Regex to parse epoch output lines from minl_new.py
    epoch_re = re.compile(r'Epoch\s+(\d+)/(\d+)\s+avg_loss=([0-9.eE+-]+)\s+loss_l2=([0-9.eE+-]+)\s+l1_reg=([0-9.eE+-]+)\s+lr=([0-9.eE+-]+)')

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    try:
        for line in proc.stdout:
            line = line.rstrip('\n')
            print(line)
            m = epoch_re.search(line)
            if m:
                epoch = int(m.group(1))
                total_epochs = int(m.group(2))
                avg_loss = float(m.group(3))
                loss_l2 = float(m.group(4))
                l1_reg = float(m.group(5))
                lr = float(m.group(6))

                entropy = float('nan')
                # try to locate saved image (minl_new saves recon_ep{epoch}.png into save-dir)
                save_dir = Path(params.get('save-dir') or Path(params['image']).parent)
                img_path = save_dir / f'recon_ep{epoch}.png'
                # wait briefly for file to appear (minl_new prints epoch then saves)
                for _ in range(10):
                    if img_path.exists():
                        entropy = compute_image_entropy(str(img_path))
                        break
                    time.sleep(0.25)

                wandb.log({'avg_loss': avg_loss, 'loss_l2': loss_l2, 'l1_reg': l1_reg, 'lr': lr, 'entropy': entropy}, step=epoch)

    finally:
        proc.wait()
        wandb.finish()


if __name__ == '__main__':
    main()
