import os
import argparse
from PIL import Image
import numpy as np

def extract_and_reconstruct(img, block_size=(16,16), center_size=(11,11)):
    """
    Separa a imagem em blocos 16x16, extrai o centro 11x11 de cada bloco,
    remonta a imagem final e retorna o array resultante.
    """
    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[..., None]
    H, W = arr.shape[:2]
    C = arr.shape[2]
    bh, bw = block_size
    ch, cw = center_size
    n_blocks_y = H // bh
    n_blocks_x = W // bw
    out_H = n_blocks_y * ch
    out_W = n_blocks_x * cw
    if C == 1:
        out_img = np.zeros((out_H, out_W), dtype=arr.dtype)
    else:
        out_img = np.zeros((out_H, out_W, C), dtype=arr.dtype)
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y0 = by * bh
            x0 = bx * bw
            block = arr[y0:y0+bh, x0:x0+bw]
            start_y = (bh - ch) // 2
            start_x = (bw - cw) // 2
            center_block = block[start_y:start_y+ch, start_x:start_x+cw]
            oy = by * ch
            ox = bx * cw
            if C == 1:
                out_img[oy:oy+ch, ox:ox+cw] = center_block[...,0]
            else:
                out_img[oy:oy+ch, ox:ox+cw, :] = center_block
    return out_img

def main():
    parser = argparse.ArgumentParser(description="Separa a imagem em blocos 16x16, extrai centro 11x11 de cada bloco e salva a imagem final.")
    parser.add_argument('--input', required=True, help='Caminho da imagem de entrada')
    parser.add_argument('--output', required=True, help='Caminho da imagem de saída')
    parser.add_argument('--block', type=int, nargs=2, default=[16,16], help='Tamanho do bloco original (ex: 16 16)')
    parser.add_argument('--center', type=int, nargs=2, default=[11,11], help='Tamanho do centro a extrair (ex: 11 11)')
    args = parser.parse_args()
    img = Image.open(args.input)
    out_img = extract_and_reconstruct(img, tuple(args.block), tuple(args.center))
    Image.fromarray(out_img).save(args.output)
    print(f"Imagem final salva em {args.output} (shape: {out_img.shape})")

if __name__ == '__main__':
    main()
