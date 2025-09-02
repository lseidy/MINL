import numpy as np
from PIL import Image
import os

input_png = "/mnt/c/Users/lucas/Documents/Mestrado/MINL/Dataset/Original_11x11_center/Ankylosaurus.png"
output_dir = "multiviews_separadas"
mi_h, mi_w = 11, 11  # tamanho da microimagem

# Carrega imagem
img = Image.open(input_png)
arr = np.array(img)
if arr.ndim == 2:
    arr = arr[..., None]
H, W = arr.shape[:2]
C = arr.shape[2]

n_blocks_y = H // mi_h
n_blocks_x = W // mi_w

# Cria pasta de saída
os.makedirs(output_dir, exist_ok=True)

# Para cada posição angular (i, j), extrai a microimagem correspondente
for i in range(mi_h):
    for j in range(mi_w):
        # Cria imagem vazia para essa microimagem
        if C == 1:
            microimg = np.zeros((n_blocks_y, n_blocks_x), dtype=arr.dtype)
        else:
            microimg = np.zeros((n_blocks_y, n_blocks_x, C), dtype=arr.dtype)
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                y0 = by * mi_h
                x0 = bx * mi_w
                if C == 1:
                    microimg[by, bx] = arr[y0 + i, x0 + j, 0]
                else:
                    microimg[by, bx, :] = arr[y0 + i, x0 + j, :]
        # Salva a microimagem
        if C == 1:
            out_img = Image.fromarray(microimg)
        else:
            out_img = Image.fromarray(microimg)
        out_path = os.path.join(output_dir, f"microimg_ang_{i:02d}_{j:02d}.png")
        out_img.save(out_path)
print(f"Salvo {mi_h * mi_w} microimagens em {output_dir}")