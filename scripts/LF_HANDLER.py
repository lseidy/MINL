import torch
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
import numpy as np

def extract_epi(lf_tensor, direction='vertical', idx=0):
    
    """
    Extrai uma EPI concatenada.

    Args:
        lf_tensor (Tensor): Tensor com shape (B, U_ang, V_ang, H_spat, W_spat)
                            U_ang, V_ang: dimensões angulares
                            H_spat, W_spat: dimensões espaciais das sub-imagens
        direction (str): 'horizontal' ou 'vertical'
        idx (int): índice da amostra no batch

    Returns:
        Tensor: EPI concatenada com shape (1, U_ang, V_ang, L_concat), 
                onde L_concat depende da direção.
    """
    
    with torch.no_grad():
        epis = []
        H_spat = lf_tensor.shape[3]
        W_spat = lf_tensor.shape[4]

        if direction == 'horizontal':
            # lf_tensor[idx, U_ang_slice, V_ang_slice, H_spat_slice, W_spat_slice]
            for y_s in range(H_spat): 
                # epi slice shape: (U_ang, V_ang, W_spat)
                epi = lf_tensor[idx, :, :, y_s, :]
                #print(epi.shape) 
                epis.append(epi)
                
            # Concatenate H_spat slices of (U_ang, V_ang, W_spat) along W_spat dimension (dim=2)
            # Resulting shape: (U_ang, V_ang, H_spat * W_spat)
            epis = torch.cat(epis, dim=2)
        elif direction == 'vertical':
            for x_s in range(W_spat):
                # epi slice shape: (U_ang, V_ang, H_spat)
                epi = lf_tensor[idx, :, :, :, x_s]
                #print(epi.shape)
                epis.append(epi)
                
            # Concatenate W_spat slices of (U_ang, V_ang, H_spat) along H_spat dimension (dim=2)
            # Resulting shape: (U_ang, V_ang, W_spat * H_spat)
            epis = torch.cat(epis, dim=2)
        else:
            raise ValueError("Direção deve ser 'horizontal' ou 'vertical'")

        epi_tensor = epis.unsqueeze(0) # Shape: (1, U_ang, V_ang, L_concat)
        return epi_tensor

def lenslet_to_lf(lenslet_img, V_angular_views=16, H_angular_views=16):
    """
    Converte uma imagem lenslet em tensor de light field.
    Args:
        lenslet_img (torch.Tensor): Tensor (B, H_full, W_full)
        V_angular_views (int): Número de vistas angulares na vertical (corresponde a U_ang em extract_epi)
        H_angular_views (int): Número de vistas angulares na horizontal (corresponde a V_ang em extract_epi)
    Returns:
        torch.Tensor: LF tensor (B, V_angular_views, H_angular_views, Y_subimg_spatial, X_subimg_spatial)
    """
    B, H_full, W_full = lenslet_img.shape
    Y_subimg_spatial = H_full // V_angular_views
    X_subimg_spatial = W_full // H_angular_views

    # Output LF tensor shape: (B, V_angular_views, H_angular_views, Y_subimg_spatial, X_subimg_spatial)
    # This corresponds to (B, U_ang for extract_epi, V_ang for extract_epi, H_spat for extract_epi, W_spat for extract_epi)
    lf = torch.zeros((B, V_angular_views, H_angular_views, Y_subimg_spatial, X_subimg_spatial), dtype=lenslet_img.dtype, device=lenslet_img.device)

    for v_idx in range(V_angular_views):
        for h_idx in range(H_angular_views):
            lf[:, v_idx, h_idx, :, :] = lenslet_img[:, v_idx::V_angular_views, h_idx::H_angular_views]

    return lf

def lf_to_lenslet(lf_tensor):
   
    """
    Converte um tensor de light field de volta para uma imagem lenslet.
    Args:
        lf_tensor (torch.Tensor): Tensor LF (B, V_ang_views, H_ang_views, Y_subimg, X_subimg)
                                  V_ang_views é o número de vistas angulares na vertical (U_ang anterior)
                                  H_ang_views é o número de vistas angulares na horizontal (V_ang anterior)
    Returns:
        torch.Tensor: Imagem lenslet reconstruída (B, H_full, W_full)
    """
    B, V_angular_views, H_angular_views, Y_subimg_spatial, X_subimg_spatial = lf_tensor.shape
    
    H_full = Y_subimg_spatial * V_angular_views
    W_full = X_subimg_spatial * H_angular_views
    
    reconstructed_lenslet = torch.zeros((B, H_full, W_full), dtype=lf_tensor.dtype, device=lf_tensor.device)
    
    for v_idx in range(V_angular_views):
        for h_idx in range(H_angular_views):
            sub_image = lf_tensor[:, v_idx, h_idx, :, :]
            reconstructed_lenslet[:, v_idx::V_angular_views, h_idx::H_angular_views] = sub_image
            
    return reconstructed_lenslet

# --- Script Principal ---
path = "/mnt/c/Users/lucas/Documents/Mestrado/MINL/Dataset/Original/Black_Fence.png"
# 1. Carregar a imagem lenslet única
try:
    lenslet_image_np_original = imageio.imread(path) # Mantemos a original para comparação
except FileNotFoundError:
    print(f"Erro: Arquivo não encontrado em '{path}'")
    exit()

lenslet_image_processed_np = lenslet_image_np_original.copy()
if lenslet_image_processed_np.ndim == 3:
    if lenslet_image_processed_np.shape[2] == 4:
        lenslet_image_processed_np = lenslet_image_processed_np[:,:,:3]
    lenslet_image_processed_np = np.mean(lenslet_image_processed_np, axis=2)

lenslet_img_tensor = torch.tensor(lenslet_image_processed_np, dtype=torch.float32).unsqueeze(0) / 255.0
print(f"Shape da imagem lenslet carregada (B, H_full, W_full): {lenslet_img_tensor.shape}")

# 2. Converter a imagem lenslet para um tensor Light Field 5D
num_angular_views_U = 16 
num_angular_views_V = 16 

lf_tensor_processed = lenslet_to_lf(lenslet_img_tensor, 
                                    V_angular_views=num_angular_views_U, 
                                    H_angular_views=num_angular_views_V)
print(f"Shape do Light Field tensor processado (B, U_ang, V_ang, H_spat, W_spat): {lf_tensor_processed.shape}")

# 3. Extrair blocos EPI concatenados usando extract_epi
epi_block_horizontal = extract_epi(lf_tensor_processed, idx=0, direction='horizontal')
epi_block_vertical = extract_epi(lf_tensor_processed, idx=0, direction='vertical')

print(f"\n--- Resultados de extract_epi ---")
print(f"Shape do bloco EPI horizontal concatenado (1, U_ang, V_ang, L_concat): {epi_block_horizontal.shape}")
print(f"Shape interno do bloco EPI horizontal (U_ang, V_ang, L_concat): {epi_block_horizontal[0].shape}")
print(f"Shape do bloco EPI vertical concatenado (1, U_ang, V_ang, L_concat): {epi_block_vertical.shape}")
print(f"Shape interno do bloco EPI vertical (U_ang, V_ang, L_concat): {epi_block_vertical[0].shape}")

# 4. Visualizar as EPIs extraídas por extract_epi
plt.figure(figsize=(15, 10))

# Visualizar EPI horizontal
epi_3D_horizontal = epi_block_horizontal[0].cpu().numpy()
central_U_idx = epi_3D_horizontal.shape[0] // 2
epi_2D_to_show_h = epi_3D_horizontal[central_U_idx, :, :]
# Salvar EPIs (horizontal)
out_dir = os.path.join(os.path.dirname(path), "EPIs")
os.makedirs(out_dir, exist_ok=True)

def _save_gray_image(arr, dst_path):
    # arr: 2D numpy float array (expected range roughly 0..1)
    a = np.asarray(arr, dtype=np.float32)
    mn = a.min()
    mx = a.max()
    if mx > mn:
        norm = (a - mn) / (mx - mn)
    else:
        norm = np.clip(a, 0.0, 1.0)
    img_u8 = (norm * 255.0).astype(np.uint8)
    imageio.imwrite(dst_path, img_u8)

central_h_path = os.path.join(out_dir, f"epi_horizontal_U{central_U_idx}.png")
_save_gray_image(epi_2D_to_show_h, central_h_path)
print(f"Saved horizontal central EPI: {central_h_path}")
plt.subplot(2, 1, 1)
plt.imshow(epi_2D_to_show_h, cmap='gray', aspect='auto')
plt.title(f"EPI Horizontal Concatenada (Fatia U_ang={central_U_idx})\nShape: {epi_2D_to_show_h.shape}")
plt.xlabel("Dimensão Espacial Concatenada (H_subimagem * W_subimagem)")
plt.ylabel("Dimensão Angular V (V_ang)")

# Visualizar EPI vertical
epi_3D_vertical = epi_block_vertical[0].cpu().numpy()
epi_2D_to_show_v = epi_3D_vertical[central_U_idx, :, :]
# Salvar EPIs (vertical)
central_v_path = os.path.join(out_dir, f"epi_vertical_U{central_U_idx}.png")
_save_gray_image(epi_2D_to_show_v, central_v_path)
print(f"Saved vertical central EPI: {central_v_path}")

# Salvar todas as fatias U (horizontal e vertical) para inspeção
for u_idx in range(epi_3D_horizontal.shape[0]):
    h_path = os.path.join(out_dir, f"epi_horizontal_u{u_idx}.png")
    v_path = os.path.join(out_dir, f"epi_vertical_u{u_idx}.png")
    _save_gray_image(epi_3D_horizontal[u_idx], h_path)
    _save_gray_image(epi_3D_vertical[u_idx], v_path)

print(f"Saved all U-slice EPIs into: {out_dir}")
plt.subplot(2, 1, 2)
plt.imshow(epi_2D_to_show_v, cmap='gray', aspect='auto')
plt.title(f"EPI Vertical Concatenada (Fatia U_ang={central_U_idx})\nShape: {epi_2D_to_show_v.shape}")
plt.xlabel("Dimensão Espacial Concatenada (W_subimagem * H_subimagem)")
plt.ylabel("Dimensão Angular V (V_ang)")
plt.tight_layout()
plt.show()

# --- 5. Reconstrução e Visualização da Imagem Original ---
print(f"\n--- Reconstrução da Imagem Original ---")

# 5.1 Reconstruir a imagem lenslet
reconstructed_lenslet_tensor = lf_to_lenslet(lf_tensor_processed) # Shape (B, H_full, W_full)
reconstructed_lenslet_img_np = reconstructed_lenslet_tensor[0].cpu().numpy() # Pega o primeiro item do batch
print(f"Shape da imagem lenslet reconstruída: {reconstructed_lenslet_img_np.shape}")

# 5.2 Extrair a vista de sub-abertura central
# lf_tensor_processed tem shape (B, U_ang, V_ang, H_spat, W_spat)
# U_ang = num_angular_views_U, V_ang = num_angular_views_V
central_u_angular_idx = lf_tensor_processed.shape[1] // 2
central_v_angular_idx = lf_tensor_processed.shape[2] // 2

central_sub_aperture_view_tensor = lf_tensor_processed[0, central_u_angular_idx, central_v_angular_idx, :, :]
central_sub_aperture_view_np = central_sub_aperture_view_tensor.cpu().numpy()
print(f"Shape da vista de sub-abertura central: {central_sub_aperture_view_np.shape}")

# 5.3 Visualizar as imagens reconstruídas e a original (lenslet)
plt.figure(figsize=(18, 6))

# Imagem Lenslet Original (processada para grayscale, se necessário)
plt.subplot(1, 3, 1)
# Se a original era colorida, lenslet_image_processed_np é a versão em grayscale
# Se já era grayscale, são a mesma.
plt.imshow(lenslet_image_processed_np, cmap='gray') 
plt.title(f"Imagem Lenslet Original (Processada)\nShape: {lenslet_image_processed_np.shape}")
plt.axis('off')

# Imagem Lenslet Reconstruída
plt.subplot(1, 3, 2)
plt.imshow(reconstructed_lenslet_img_np, cmap='gray')
plt.title(f"Imagem Lenslet Reconstruída\nShape: {reconstructed_lenslet_img_np.shape}")
plt.axis('off')

# Vista de Sub-Abertura Central
plt.subplot(1, 3, 3)
plt.imshow(central_sub_aperture_view_np, cmap='gray')
plt.title(f"Vista de Sub-Abertura Central\n(U_ang={central_u_angular_idx}, V_ang={central_v_angular_idx})\nShape: {central_sub_aperture_view_np.shape}")
plt.axis('off')

plt.tight_layout()
plt.show()