import os
from PIL import Image
import numpy as np

# Parâmetros
input_path = '/mnt/c/Users/lucas/Documents/Mestrado/MINL/Dataset/Original_11x11_center/Ankylosaurus.png'
output_dir = 'inputs_ankylossaurus'
mi_h, mi_w = 11, 11  # tamanho da microimagem

# Cria pasta de saída se não existir
os.makedirs(output_dir, exist_ok=True)

# Carrega imagem
img = Image.open(input_path)
arr = np.array(img)
if arr.ndim == 2:
    arr = arr[..., None]
H, W = arr.shape[:2]
C = arr.shape[2]

n_blocks_y = H // mi_h
n_blocks_x = W // mi_w

count = 0
for by in range(n_blocks_y):
    for bx in range(n_blocks_x):
        y0 = by * mi_h
        x0 = bx * mi_w
        block = arr[y0:y0+mi_h, x0:x0+mi_w]
        if C == 1:
            block = block[..., 0]
        out_path = os.path.join(output_dir, f"mi_y{by:02d}_x{bx:02d}.png")
        Image.fromarray(block).save(out_path)
        count += 1

print(f"Salvo {count} microimagens em {output_dir}")