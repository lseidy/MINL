# MiNL: Treinamento MI-wise para Light Fields em imagens de lentelet

Este repositório implementa o MiNL (Micro-Images based Neural Representation for Light Fields) em modo MI-wise, como descrito no paper “MiNL: Micro-Images based Neural Representation for Light Fields”.

Em uma imagem de lentelet, cada bloco U×V (tipicamente 11×11) é uma micro‑imagem angular de uma microlente. O MiNL aprende um mapeamento das coordenadas da microlente (x, y) para as cores de toda a micro‑imagem (C, U, V), em vez de prever pixel a pixel. Isso acelera a decodificação e evita “vazamento” entre micro‑imagens adjacentes.

Arquivo principal: `minl_new.py`.

## Requisitos

- Python 3.9+ (testado)
- PyTorch + CUDA (opcional, para GPU)
- torchvision
- Pillow (PIL)
- matplotlib

Instalação básica:
```powershell
py -m pip install torch torchvision pillow matplotlib
```

Para GPU, instale a versão de Torch compatível com sua CUDA (veja instruções no site da PyTorch).

## Formato dos dados (imagem de lentelet)

Entrada: uma imagem RGB única (C, Htot, Wtot) onde cada bloco U×V é uma micro‑imagem.
- Por padrão, U=V=11 (`--micro-size 11`).
- A função `reshape_lenslet` transforma para um tensor 5D: (C, Hm, Wm, U, V), onde:
  - Hm = Htot // U e Wm = Wtot // V (bordas são recortadas para múltiplos de U=V).
  - Cada posição (y, x) em (Hm, Wm) corresponde a uma micro‑imagem angular de shape (C, U, V).

Você pode ajustar `--micro-size` se seu dataset usar outro passo.

## Pipeline do treinamento

1) Carregamento e reshape
- Carrega a imagem com `load_image`.
- Converte de (C, Htot, Wtot) para (C, Hm, Wm, U, V) com `reshape_lenslet`.

2) Dataset MI-wise
- `LensletMicroDataset` itera sobre microlentes (Hm×Wm).
- Entrada (x_in): Position Encoding (Fourier) das coordenadas (x, y) normalizadas no grid Hm×Wm.
- Alvo (y_gt): micro‑imagem angular (C, U, V) do light field.

3) Modelo
- `SimpleMLP`: recebe x_in e gera um vetor (C·U·V) que é remodelado para (C, U, V).
- `SmallCNN`: refina a micro‑imagem no domínio angular (U, V) por canal C.

4) Função de perda
- loss = L2(recon, GT) + alpha · L1(pesos), onde:
  - L2: soma dos quadrados por amostra e média no batch (equivalente à MSE até uma constante de escala).
  - L1: regularização nos pesos do MLP+CNN, controlada por `--alpha`.

5) Otimizador e scheduler
- Adam com `--lr-start` e scheduler linear até `--lr-end` ao final de `--epochs`.

6) Métricas e logs
- Cross‑entropy H(p, q) em bits (métrica de monitoramento):
  - q: softmax das previsões achatadas.
  - p: GT não‑negativo e normalizado (ReLU + normalização).
  - Apenas para log, não entra na loss.
- Salva `training_metrics.png` a cada época (loss e cross‑entropy).

7) Reconstruções (mosaicos)
- Antes do treinamento: salva `mosaic_ep0.png` (modelo não treinado).
- Durante o treinamento: se `--save-recon`, salva `mosaic_epK.png` a cada `--save-interval` épocas.
- Ao final: se a última época não alinhou com o intervalo, salva `mosaic_final_ep{epochs}.png`.

Todos os arquivos são salvos em `--save-dir` (padrão: `output_minl/` do seu usuário). Se omitido ou inválido, usa a pasta da imagem.

## Laço de treinamento (explicação)

Para cada época:
- Zera acumuladores.
- Para cada minibatch do DataLoader:
  - Move dados para `device` (CPU/GPU).
  - Forward:
    - `mlp_out = mlp(xb)`  → shape (B, C·U·V).
    - `mlp_micro = mlp_out.view(-1, C, U, V)`.
    - `pred = cnn(mlp_micro)` → (B, C, U, V).
  - Loss:
    - `loss_l2 = ||pred − yb||²` por amostra, média no batch.
    - `l1_reg = soma |pesos|` em MLP e CNN.
    - `loss = loss_l2 + alpha*l1_reg`.
  - Backprop e atualização:
    - `loss.backward(); optimizer.step()`.
  - Métrica (sem gradiente):
    - Achata `pred` e `yb` para (B, N), N=C·U·V.
    - `q = softmax(pred)`, `p = normalize(ReLU(yb))`.
    - `xent = −∑ p log2 q` (média no batch); acumula.
- `scheduler.step()` e logs por época.
- Salva métricas e, se habilitado, o mosaico daquela época.

## Uso

Exemplo (PowerShell no Windows):
```powershell
py c:\Users\lucas\Documents\Mestrado\MINL\minl_new.py `
  --image "C:\Users\lucas\Documents\Mestrado\MINL\Dataset\Original_11x11_center\Ankylosaurus.png" `
  --epochs 50 --batch-size 4096 --save-recon --save-interval 10 `
  --L 6 --include-input --alpha 1e-4 --lr-start 1e-2 --lr-end 1e-4 `
  --micro-size 11 --num-workers 0 --save-dir "C:\Users\lucas\Documents\Mestrado\MINL\output_minl"
```

Notas:
- `--micro-size`: tamanho U=V de cada micro‑imagem (11 por padrão).
- `--include-input`: concatena (x, y) originais ao PE.
- `--resize W H`: opcional; redimensiona a imagem antes do reshape (cuidado para manter múltiplos de U=V).
- `--num-workers` no Windows: 0 (seguro) ou 2 (se RAM suficiente).
- `--batch-size`: ajuste conforme sua GPU/CPU (comece menor se houver OOM).

## Saídas esperadas

- `mosaic_ep0.png`: reconstrução inicial (sem treino).
- `mosaic_epK.png`: reconstruções periódicas (se `--save-recon`).
- `mosaic_final_ep{epochs}.png`: reconstrução da última época (se necessário).
- `training_metrics.png`: gráfico de loss e cross‑entropy por época (sobrescrevido a cada época).
- Logs com shapes, contagem de parâmetros e LR por época.

## GPU

- O script usa GPU automaticamente se `torch.cuda.is_available()` for True.
- `pin_memory` é ativado quando em CUDA e os `.to(device, non_blocking=True)` aceleram a transferência.
- Para confirmar no console:
  - Verifique `device=cuda` nos prints.
  - Opcional: imprima `torch.cuda.get_device_name(0)`.

## Dicas de desempenho

- Reduza `--batch-size` em caso de “CUDA out of memory”.
- Diminua `--L` (menos frequências no PE) ou o `hidden` do MLP se o in/out ficar muito grande.
- `--num-workers 0` no Windows evita problemas de spawn/memória.
- Mantenha `--save-interval` moderado para não gastar tempo de I/O.

## Solução de problemas

- Shape mismatch ao `view(-1, C, U, V)`:
  - Verifique U=V= `--micro-size` e se as dimensões da imagem são múltiplos de U=V (o código recorta bordas).
  - Veja o log “Light field lenslet -> (C,Hm,Wm,U,V)”.
- Arquivo não encontrado:
  - Caminhos Windows vs WSL são normalizados por `normalize_path`; passe o caminho no formato do seu SO.
- Execução na CPU (lenta):
  - Instale PyTorch com CUDA compatível e garanta que `torch.cuda.is_available()` retorna True.
- O treino não melhora:
  - Ajuste LR (`--lr-start`, `--lr-end`), `alpha`, `hidden`, `L`, `epochs`.
  - Verifique se a imagem é realmente uma lenslet 11×11.

## Como isso se alinha ao paper

- MI-wise: mapeia (x, y) de microlentes → micro‑imagem (C, U, V), em vez de “pixel-wise”.
- Decodificação rápida: uma forward por microlente retorna toda a micro‑imagem.
- Sem vazamento: as micro‑imagens são preditas bloco a bloco, preservando fronteiras.

## Estrutura dos arquivos

- `minl_new.py`: script principal (treino, métricas e salvamento).
- `networks/mlp.py`: `SimpleMLP` (entrada PE → vetor C·U·V).
- `networks/cnn.py`: `SmallCNN` (refino no domínio angular (U, V)).
