import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import wandb
from PixelCNN import PixelCNN

# --- 1. Codificação de Posição (Position Encoding) ---
# [cite_start]Conforme descrito na seção II-A, item 'a' e Equação (1) do artigo[cite: 86, 87].
class PositionalEncoding(nn.Module):
    def __init__(self, L=10):
        """
        Inicializa o codificador posicional.
        Args:
            L (int): Hiperparâmetro que controla a dimensionalidade do embedding.
        """
        super().__init__()
        self.L = L

    def forward(self, coords):
        """
        Aplica a codificação posicional nas coordenadas de entrada.
        Args:
            coords (torch.Tensor): Tensor de coordenadas com shape (batch_size, 2).
        Returns:
            torch.Tensor: Tensor de coordenadas codificadas com shape (batch_size, 2 * 2 * L).
        """
        if self.L == 0:
            return coords

        pi = math.pi
        # Ensure frequency bands are on the same device and dtype as input coords
        freq_bands = (2. ** torch.arange(self.L, device=coords.device, dtype=coords.dtype)) * pi

        # Multiplica cada coordenada por cada banda de frequência
        # Shape: (batch_size, 2, L)
        scaled_coords = coords.unsqueeze(-1) * freq_bands

        # Aplica seno e cosseno e concatena
        # Shape: (batch_size, 2, 2*L)
        encoded_coords = torch.cat([torch.sin(scaled_coords), torch.cos(scaled_coords)], dim=-1)

        # Achata para o formato final (batch_size, 2 * 2 * L)
        return encoded_coords.flatten(1)

    def get_output_dim(self):
        # A dimensão de saída é 2 (x,y) * 2 (sin, cos) * L (frequências)
        return 2 * 2 * self.L

# --- 2. Arquitetura do Modelo MiNL ---
# [cite_start]Conforme descrito na seção II-A, item 'b' e Figura 2(b) do artigo[cite: 88, 89].

class SineActivation(nn.Module):
    """Sine activation (SIREN-style) with adjustable frequency w0."""
    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


def siren_linear_init(linear: nn.Linear, w0: float = 30.0, is_first: bool = False, in_features: int = None):
    """Initialization suggested by SIREN for linear layers."""
    with torch.no_grad():
        if in_features is None:
            in_features = linear.in_features
        if is_first:
            linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
        else:
            bound = math.sqrt(6.0 / in_features) / w0
            linear.weight.uniform_(-bound, bound)
        if linear.bias is not None:
            linear.bias.zero_()


class PositionalEncoding2D(nn.Module):
    """Fourier features for 2D coordinates as in the reference.
    Returns tensor of shape (B, 4L): [sin_x, cos_x, sin_y, cos_y].
    """
    def __init__(self, L: int = 8):
        super().__init__()
        self.L = L
        # frequencies = [2^0, 2^1, ..., 2^{L-1}] * pi
        freqs = (2.0 ** torch.arange(L)) * math.pi
        self.register_buffer('freqs', freqs, persistent=False)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        # xy: (B,2) with values in [0,1]
        x = xy[:, :1]
        y = xy[:, 1:2]
        xw = x * self.freqs.to(x.device)
        yw = y * self.freqs.to(x.device)
        sinx = torch.sin(xw)
        cosx = torch.cos(xw)
        siny = torch.sin(yw)
        cosy = torch.cos(yw)
        return torch.cat([sinx, cosx, siny, cosy], dim=1)

class MiNL(nn.Module):
    def __init__(self, coord_dim=None, mi_resolution=(11, 11), pe_L=8,
                 mlp_hidden=128, mlp_layers=3, siren_w0=30.0,
                 cnn_channels=64, cnn_blocks=3):
        super().__init__()
        self.mi_h, self.mi_w = mi_resolution
        self.siren_w0 = siren_w0

        # positional encoder (replace previous PositionalEncoding)
        self.pe = PositionalEncoding2D(L=pe_L)

        mlp_in = 4 * pe_L
        # build SIREN MLP
        layers = []
        in_f = mlp_in
        # first layer
        fc = nn.Linear(in_f, mlp_hidden)
        siren_linear_init(fc, w0=self.siren_w0, is_first=True, in_features=in_f)
        layers += [fc, SineActivation(self.siren_w0)]
        for i in range(mlp_layers - 1):
            fc = nn.Linear(mlp_hidden, mlp_hidden)
            siren_linear_init(fc, w0=self.siren_w0, is_first=False, in_features=mlp_hidden)
            layers += [fc, SineActivation(self.siren_w0)]
        self.mlp = nn.Sequential(*layers)

        # project latent to feature map for microimage (produce a 1x1 feature map)
        # PixelCNN expects input shape (B, cnn_channels, 1, 1)
        self.to_feat = nn.Linear(mlp_hidden, cnn_channels)
        siren_linear_init(self.to_feat, w0=self.siren_w0, is_first=False, in_features=mlp_hidden)

        # Use PixelCNN decoder (upsamples from 1x1 -> mi_h x mi_w and refines)
        # PixelCNN applies a sigmoid at the output so no extra activation is needed here.
        self.cnn = PixelCNN(in_channels=cnn_channels, out_channels=3, mi_h=self.mi_h, mi_w=self.mi_w)

        # Alternative convolutional decoder used when the input is a full coordinate map
        # (B, H, W, 2) -> we process each coordinate with the MLP and reshape to (B, C, H, W)
        # then refine with a small conv head to produce (B, out_channels, H, W).
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            SineActivation(self.siren_w0),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            SineActivation(self.siren_w0),
            nn.Conv2d(cnn_channels, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def _apply_siren_init(self, module, w0=30.0):
        """
        Inicialização recomendada para redes SIREN.
        Para a primeira camada (linear no início do MLP), usa-se uma amplitude maior.
        Para camadas subsequentes, a amplitude é escalada por 1/w0.
        """
        for m in module.modules():
            if isinstance(m, nn.Linear):
                in_features = m.in_features
                with torch.no_grad():
                    bound = math.sqrt(6.0 / in_features) / (w0 if in_features > 0 else 1.0)
                    m.weight.uniform_(-bound, bound)
                    if m.bias is not None:
                        m.bias.zero_()

    def forward(self, coords):
        # coords expected normalized [0,1]
        # Two supported input shapes:
        #  - (B, 2): one 2D coordinate per micro-image (original behaviour)
        #  - (B, H, W, 2): full coordinate grid for the micro-image (e.g. H=W=11)
        if coords.dim() == 2:
            # Single coordinate per sample -> original path
            pe = self.pe(coords)
            z = self.mlp(pe)
            feat = self.to_feat(z)
            B = feat.shape[0]
            # reshape to (B, C, 1, 1) for PixelCNN
            feat = feat.view(B, -1, 1, 1)
            out = self.cnn(feat)
            return out

        elif coords.dim() == 4:
            # Full coordinate map per sample: (B, H, W, 2)
            B, H, W, two = coords.shape
            assert two == 2, "coords last dimension must be 2"
            # Flatten spatial positions and run through positional encoder + MLP
            coords_flat = coords.view(B * H * W, 2)
            pe = self.pe(coords_flat)
            z = self.mlp(pe)
            feat = self.to_feat(z)  # (B*H*W, cnn_channels)
            # reshape to (B, C, H, W)
            feat = feat.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            # refine with conv decoder -> (B, 3, H, W)
            out = self.decoder_conv(feat)
            return out

        else:
            raise ValueError(f"Unsupported coords shape: {tuple(coords.shape)}. Expected (B,2) or (B,H,W,2).")

# --- 3. Script de Treinamento ---

def train_minl(params=None):
    import os  # Garante que o módulo os esteja disponível
    # Parâmetros
    if params is not None:
        lf_resolution = (params.res_ver, params.res_hor)  # ou ajuste conforme necessário
        mi_resolution = (params.num_views_ver, params.num_views_hor)
        L = 10 if not hasattr(params, 'L') else params.L
        epochs = params.epochs
        # support explicit lr_start and lr_final in params; fall back to params.lr if provided
        lr_start = getattr(params, 'lr_start', getattr(params, 'lr', 1e-1))
        lr_final = getattr(params, 'lr_final', 1e-4)
        batch_size = getattr(params, 'batch_size', 512)
        alpha = getattr(params, 'alpha', 0)  # default to 1e-7 per paper tuning
        
    #Verifica se há uma GPU disponível (o artigo usa uma RTX 3080) [cite: 129]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")


    # Usa os dados reais carregados no início do script
    global all_coords, ground_truth_mis
    
    # --- Preparação para o Treinamento ---
    # Ensure reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    dataset = torch.utils.data.TensorDataset(all_coords, ground_truth_mis)
    # On Windows, high num_workers can cause issues; use 0 or 2 depending on availability
    safe_num_workers = 0 if (os.name == 'nt') else 2
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=safe_num_workers)
   
    # Instancia o modelo
    pos_encoder = PositionalEncoding(L=L)
    model = MiNL(coord_dim=pos_encoder.get_output_dim(), mi_resolution=mi_resolution).to(device)
    # Mostra o número de parâmetros treináveis
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Número total de parâmetros treináveis: {num_params}")
    # Ensure positional encoder uses same dtype/device when encoding inputs
    pos_encoder = pos_encoder.to(device)
    
    # Otimizador e função de perda [cite: 92, 127]
    # Initialize optimizer with the starting learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
    # Create a scheduler that decays the learning rate from lr_start -> lr_final over `epochs` steps.
    # We use an exponential schedule by returning a multiplicative factor relative to lr_start.
    def _lr_lambda(epoch, start=lr_start, final=lr_final, total=epochs):
        # epoch in [0, total-1]; returns factor to multiply lr_start
        if total <= 1:
            return final / start
        ratio = final / start
        return ratio ** (epoch / max(1, total - 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: _lr_lambda(e))
    loss_fn = nn.MSELoss() # A parte L2 da perda

    wandb_active = hasattr(params, 'wandb_active') and params.wandb_active if params is not None else False

    print("Iniciando o treinamento...")
    for epoch in range(epochs):
        total_l2_sum = 0.0   # sum of per-sample squared errors across epoch
        total_entropy = 0.0
        total_samples = 0
        for batch_coords, batch_mis_gt in dataloader:
            batch_coords, batch_mis_gt = batch_coords.to(device), batch_mis_gt.to(device)
            # Zera os gradientes
            optimizer.zero_grad()

            # Forward pass
            encoded_coords = pos_encoder(batch_coords)
            predicted_mis = model(encoded_coords)

            # Cálculo da perda L2 conforme a formula (norma L2, não o quadrado):
            # para cada amostra i, computa-se ||Mpre_i - Mgt_i||_2 (raiz da soma dos quadrados); depois faz média sobre B
            diff = predicted_mis - batch_mis_gt
            per_sample_sq = diff.view(diff.size(0), -1).pow(2).sum(dim=1)  # (B,) soma dos quadrados por amostra
            per_sample_norm = per_sample_sq.sqrt()  # (B,) norma L2 por amostra
            l2_loss_batch = per_sample_norm.mean()  # (1/B) sum_i ||...||_2

            # Regularização L1 nos pesos do modelo (||theta||_1)
            l1_norm = sum(p.abs().sum() for p in model.parameters())

            # Perda usada para backward (mesma forma do artigo aplicada por batch)
            loss = l2_loss_batch + alpha * l1_norm

            # Backward pass e otimização
            loss.backward()
            optimizer.step()

            # Acumula para estatísticas de época (média por amostra correta) — usamos a norma L2 por amostra
            n = batch_mis_gt.size(0)
            total_l2_sum += per_sample_norm.sum().item()  # soma das normas L2 por amostra
            # Entropia: pondera pela quantidade de amostras no batch
            pred_flat = predicted_mis.detach().cpu().reshape(predicted_mis.shape[0], -1)
            pred_flat = pred_flat / (pred_flat.sum(dim=1, keepdim=True) + 1e-8)
            entropy = - (pred_flat * (pred_flat + 1e-8).log()).sum(dim=1).mean().item()
            total_entropy += entropy * n
            total_samples += n
            

        # Loss média por amostra: (1/total_samples) * sum_i ||Mpre_i - Mgt_i||^2
        avg_l2 = total_l2_sum / total_samples if total_samples > 0 else 0.0
        # L1 regularizer (compute on current model state to report)
        l1_epoch = sum(p.abs().sum().item() for p in model.parameters())
        avg_loss = avg_l2 + alpha * l1_epoch
        avg_entropy = total_entropy / total_samples if total_samples > 0 else 0.0
        if (epoch + 1) % 1 == 0:
            print(f"Época [{epoch+1}/{epochs}], loss: {avg_loss:.8f}, entropia: {avg_entropy:.6f}, lr: {optimizer.param_groups[0]['lr']}")
        if wandb_active:
            wandb.log({"loss": avg_loss, "entropia": avg_entropy, "lr": optimizer.param_groups[0]["lr"]}, step=epoch)

        # Salva a imagem reconstruída a cada 20 épocas
        if (epoch + 1) % 20 == 0 or (epoch + 1) == epochs:
            model.eval()
            with torch.no_grad():
                batch_size_pred = 512
                pred_mis = []
                for i in range(0, all_coords.shape[0], batch_size_pred):
                    batch_coords = all_coords[i:i+batch_size_pred].to(device)
                    encoded_coords = pos_encoder(batch_coords)
                    batch_pred = model(encoded_coords)
                    pred_mis.append(batch_pred.cpu())
                pred_mis = torch.cat(pred_mis, dim=0)  # (N, C, mi_h, mi_w)
            mi_h, mi_w = pred_mis.shape[2:4]
            n_blocks_y = N_BLOCKS_Y
            n_blocks_x = N_BLOCKS_X
            C = pred_mis.shape[1]
            pred_mis_np = (pred_mis.numpy() * 255).clip(0,255).astype(np.uint8)
            if C == 1:
                pred_mis_np = pred_mis_np[:,0]
            if C > 1:
                mosaic = np.zeros((n_blocks_y*mi_h, n_blocks_x*mi_w, C), dtype=np.uint8)
            else:
                mosaic = np.zeros((n_blocks_y*mi_h, n_blocks_x*mi_w), dtype=np.uint8)
            idx = 0
            for by in range(n_blocks_y):
                for bx in range(n_blocks_x):
                    if idx >= pred_mis_np.shape[0]:
                        continue
                    y0 = by * mi_h
                    x0 = bx * mi_w
                    if C > 1:
                        block = pred_mis_np[idx].transpose(1,2,0) if pred_mis_np[idx].shape[0] == C else pred_mis_np[idx]
                        mosaic[y0:y0+mi_h, x0:x0+mi_w, :] = block
                    else:
                        mosaic[y0:y0+mi_h, x0:x0+mi_w] = pred_mis_np[idx]
                    idx += 1
            import os
            output_dir = "output_ankylosaurus"
            os.makedirs(output_dir, exist_ok=True)
            from PIL import Image
            out_path = os.path.join(output_dir, f"reconstrucao_epoca_{epoch+1}.png")
            Image.fromarray(mosaic).save(out_path)
            print(f"Mosaico salvo em {out_path}")
        # Step the scheduler once per epoch (after logging/printing)
        try:
            scheduler.step()
        except Exception:
            # In case scheduler is not defined for some reason, ignore silently
            pass


    print("Treinamento concluído.")

    # --- 4. Exemplo de Decodificação ---
    print("Decodificando uma microimagem de exemplo...")
    model.eval() # Coloca o modelo em modo de avaliação
    # Pega uma coordenada de exemplo (centro da imagem)
    example_coord = torch.tensor([[0.5, 0.5]]).to(device)
    with torch.no_grad():
        encoded_coord = pos_encoder(example_coord)
        decoded_mi = model(encoded_coord)
    decoded_mi_np = decoded_mi.squeeze(0).cpu().permute(1, 2, 0).numpy()
    center_idx = (all_coords - example_coord.cpu()).pow(2).sum(1).argmin()
    gt_mi = ground_truth_mis[center_idx].cpu()
    gt_mi_np = gt_mi.permute(1, 2, 0).numpy()
    print("Ground truth selecionada (tensor):", gt_mi)
    print("Ground truth selecionada (numpy):", gt_mi_np)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(gt_mi_np)
    axes[0].set_title("Microimagem Original (Ground Truth)")
    axes[0].axis('off')
    axes[1].imshow(decoded_mi_np)
    axes[1].set_title("Microimagem Decodificada pela MiNL")
    axes[1].axis('off')
    plt.show()

    # --- Reconstrução e visualização do mosaico completo (predito e ground truth) ---
    print("Reconstruindo e exibindo o mosaico completo (predito e ground truth)...")
    model.eval()
    with torch.no_grad():
        batch_size_pred = 512
        pred_mis = []
        for i in range(0, all_coords.shape[0], batch_size_pred):
            batch_coords = all_coords[i:i+batch_size_pred].to(device)
            encoded_coords = pos_encoder(batch_coords)
            batch_pred = model(encoded_coords)
            pred_mis.append(batch_pred.cpu())
        pred_mis = torch.cat(pred_mis, dim=0)  # (N, C, mi_h, mi_w)
    mi_h, mi_w = pred_mis.shape[2:4]
    n_blocks_y = N_BLOCKS_Y
    n_blocks_x = N_BLOCKS_X
    C = pred_mis.shape[1]
    pred_mis_np = (pred_mis.numpy() * 255).clip(0,255).astype(np.uint8)
    if C == 1:
        pred_mis_np = pred_mis_np[:,0]
    if C > 1:
        mosaic_pred = np.zeros((n_blocks_y*mi_h, n_blocks_x*mi_w, C), dtype=np.uint8)
    else:
        mosaic_pred = np.zeros((n_blocks_y*mi_h, n_blocks_x*mi_w), dtype=np.uint8)
    idx = 0
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            if idx >= pred_mis_np.shape[0]:
                continue
            y0 = by * mi_h
            x0 = bx * mi_w
            if C > 1:
                block = pred_mis_np[idx].transpose(1,2,0) if pred_mis_np[idx].ndim==3 and pred_mis_np[idx].shape[0]==C else pred_mis_np[idx]
                mosaic_pred[y0:y0+mi_h, x0:x0+mi_w, :] = block
            else:
                mosaic_pred[y0:y0+mi_h, x0:x0+mi_w] = pred_mis_np[idx]
            idx += 1
    # Ground truth
    gt_mis = ground_truth_mis.cpu().numpy()
    mi_h_gt, mi_w_gt = gt_mis.shape[2:4]
    C_gt = gt_mis.shape[1]
    if C_gt == 1:
        gt_mis_np = (gt_mis[:,0] * 255).clip(0,255).astype(np.uint8)
    else:
        gt_mis_np = (gt_mis.transpose(0,2,3,1) * 255).clip(0,255).astype(np.uint8)
    if C_gt > 1:
        mosaic_gt = np.zeros((n_blocks_y*mi_h_gt, n_blocks_x*mi_w_gt, C_gt), dtype=np.uint8)
    else:
        mosaic_gt = np.zeros((n_blocks_y*mi_h_gt, n_blocks_x*mi_w_gt), dtype=np.uint8)
    idx = 0
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            if idx >= gt_mis_np.shape[0]:
                continue
            y0 = by * mi_h_gt
            x0 = bx * mi_w_gt
            if C_gt > 1:
                block = gt_mis_np[idx]
                mosaic_gt[y0:y0+mi_h_gt, x0:x0+mi_w_gt, :] = block
            else:
                mosaic_gt[y0:y0+mi_h_gt, x0:x0+mi_w_gt] = gt_mis_np[idx]
            idx += 1
    # Exibe lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(mosaic_gt)
    axes[0].set_title("Mosaico Ground Truth Completo")
    axes[0].axis('off')
    axes[1].imshow(mosaic_pred)
    axes[1].set_title("Mosaico Predito Completo")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    # --- 5. Reconstrução e salvamento do mosaico final ---
    print("Reconstruindo e salvando o mosaico final...")
    model.eval()
    with torch.no_grad():
        batch_size_pred = 512
        pred_mis = []
        for i in range(0, all_coords.shape[0], batch_size_pred):
            batch_coords = all_coords[i:i+batch_size_pred].to(device)
            encoded_coords = pos_encoder(batch_coords)
            batch_pred = model(encoded_coords)
            pred_mis.append(batch_pred.cpu())
        pred_mis = torch.cat(pred_mis, dim=0)  # (N, C, mi_h, mi_w)
    mi_h, mi_w = pred_mis.shape[2:4]
    n_blocks_y = N_BLOCKS_Y
    n_blocks_x = N_BLOCKS_X
    C = pred_mis.shape[1]
    pred_mis_np = (pred_mis.numpy() * 255).clip(0,255).astype(np.uint8)
    if C == 1:
        pred_mis_np = pred_mis_np[:,0]
    # Garante shape igual ao ground truth
    if C > 1:
        mosaic = np.zeros((n_blocks_y*mi_h, n_blocks_x*mi_w, C), dtype=np.uint8)
    else:
        mosaic = np.zeros((n_blocks_y*mi_h, n_blocks_x*mi_w), dtype=np.uint8)
    idx = 0
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            if idx >= pred_mis_np.shape[0]:
                continue
            y0 = by * mi_h
            x0 = bx * mi_w
            if C > 1:
                # Corrige para (mi_h, mi_w, C) se necessário
                block = pred_mis_np[idx].transpose(1,2,0) if pred_mis_np[idx].shape[0] == C else pred_mis_np[idx]
                mosaic[y0:y0+mi_h, x0:x0+mi_w, :] = block
            else:
                mosaic[y0:y0+mi_h, x0:x0+mi_w] = pred_mis_np[idx]
            idx += 1
    from PIL import Image
    Image.fromarray(mosaic).save("reconstrucao_final.png")
    print(f"Mosaico reconstruído salvo como reconstrucao_final.png. Shape final: {mosaic.shape}")


# --- Ajuste para carregar imagens reais e montar dataset de coordenadas e microimagens ---
def load_coords_and_microimages(input_img_path, gt_img_path, mi_resolution=(11,11)):
    # Carrega imagem de coordenadas de entrada
    input_img = Image.open(input_img_path)
    arr_in = np.array(input_img)
    if arr_in.ndim == 2:
        arr_in = arr_in[..., None]
    H, W = arr_in.shape[:2]
    C = arr_in.shape[2]

    # Carrega ground truth (mosaico de microimagens)
    gt_img = Image.open(gt_img_path)
    arr_gt = np.array(gt_img)
    if arr_gt.ndim == 2:
        arr_gt = arr_gt[..., None]
    H_gt, W_gt = arr_gt.shape[:2]
    C_gt = arr_gt.shape[2]
    mi_h, mi_w = mi_resolution
    n_blocks_y = H_gt // mi_h
    n_blocks_x = W_gt // mi_w
    global N_BLOCKS_Y, N_BLOCKS_X
    N_BLOCKS_Y = n_blocks_y
    N_BLOCKS_X = n_blocks_x
    assert n_blocks_y * n_blocks_x == H * W, "Número de microimagens não bate com número de coordenadas!"

    # Gera coordenadas normalizadas (0..1)
    coords = []
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            u = by / (n_blocks_y - 1) if n_blocks_y > 1 else 0.0
            v = bx / (n_blocks_x - 1) if n_blocks_x > 1 else 0.0
            coords.append([u, v])
    coords = np.array(coords, dtype=np.float32)

    # Extrai microimagens 11x11
    microimgs = []
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y0 = by * mi_h
            x0 = bx * mi_w
            block = arr_gt[y0:y0+mi_h, x0:x0+mi_w]
            if C_gt == 1:
                block = block[..., 0]
            microimgs.append(block)
    microimgs = np.stack(microimgs, axis=0)  # (N, mi_h, mi_w, C) ou (N, mi_h, mi_w)

    # Ajusta formato para torch: (N, C, mi_h, mi_w)
    microimgs = torch.from_numpy(microimgs.astype(np.float32) / 255.0)
    if microimgs.ndim == 3:
        microimgs = microimgs.unsqueeze(1)  # (N, 1, mi_h, mi_w)
    else:
        microimgs = microimgs.permute(0, 3, 1, 2)  # (N, C, mi_h, mi_w)
    coords = torch.from_numpy(coords)
    return coords, microimgs

# --- Use sempre dados reais ---
input_img_path = '/mnt/c/Users/lucas/Documents/Mestrado/MINL/Dataset/Multiview_cropped/Ankylosaurus.png'
gt_img_path = '/mnt/c/Users/lucas/Documents/Mestrado/MINL/Dataset/Original_11x11_center/Ankylosaurus.png'
mi_resolution = (11, 11)
all_coords, ground_truth_mis = load_coords_and_microimages(input_img_path, gt_img_path, mi_resolution)



