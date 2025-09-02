import os
import wandb
from minl import train_minl

# Parâmetros de exemplo (ajuste conforme necessário)
class Params:
    wandb_active = True
    model = 'MINL'
    dataset_name = 'Ankylosaurus'
    dataset_path = '/mnt/c/Users/lucas/Documents/Mestrado/MINL/Dataset/Multiview_cropped/Ankylosaurus.png'
    num_views_ver = 11
    num_views_hor = 11
    res_ver = 432
    res_hor = 622
    epochs = 250
    batch_size = 5000
    # Learning-rate schedule matching the paper: start at 1e-2 and decay to 1e-4
    lr_start = 1e-1
    lr_final = 1e-4
    lr = lr_start
    loss = 'MSE+L1'
    lr_scheduler = 'None'
    lr_gamma = 0.1
    lr_step_size = 50
    optimizer = 'Adam'
    config_name = 'run_minl_ankylosaurus'
    # Optional: specify the wandb entity (team or user) to send runs to
    # Default to your team 'minl'
    wandb_entity = 'minl'
    skip_connections = False
    num_filters = 32
    context_size = 0
    predictor_size = 0
    transforms = None
    crop_mode = None
    prune_step = None
    target_sparsity = None

params = Params()

if params.wandb_active:
    # Prefer using an environment variable for the API key for safety
    # Ensure you set WANDB_API_KEY to your own key to avoid sending runs to another account
    api_key = "61d0b03bf56ea248acb4e0c5066dbce0ed05fc82"
    if api_key:
        try:
            wandb.login(key=api_key, force=True)
        except Exception:
            wandb.login()
    else:
        try:
            wandb.login()
        except Exception:
            print('wandb login failed or skipped; continuing without login')
            print('To avoid sending runs to another account, set $env:WANDB_API_KEY and/or $env:WANDB_ENTITY before running.')

    wandb.init(
    project="MINL",
    name=params.config_name,
    entity=getattr(params, 'wandb_entity', None),
    config={
            "architecture": params.model,
            "dataset": params.dataset_name,
            "dataset name": params.dataset_path,
            "views ver": params.num_views_ver,
            "views hor": params.num_views_hor,
            "epochs": params.epochs,
            "batch size": params.batch_size,
            "learning_rate_start": getattr(params, 'lr_start', params.lr),
            "learning_rate_final": getattr(params, 'lr_final', None),
            "Loss": params.loss,
            "scheduler": params.lr_scheduler,
            "lr-gamma": params.lr_gamma,
            "lr-step": params.lr_step_size,
            "optimizer": params.optimizer,
            "name": params.config_name,
            "Skip Connections": params.skip_connections,
            "Num-Filters": params.num_filters,
            "Context Size": params.context_size,
            "Predictor Size": params.predictor_size,
            "Transforms": params.transforms,
            "Crop-mode": params.crop_mode,
            "Prune Step": params.prune_step,
            "Target Sparsity": params.target_sparsity
        }
    )
    # Print confirmation about where the run was created
    print('wandb run entity:', getattr(wandb.run, 'entity', None))
    print('wandb run project:', getattr(wandb.run, 'project', None))
    print('wandb run id:', getattr(wandb.run, 'id', None))

# Chama o treinamento do minl.py, logando métricas no wandb
def train_and_log():
    # Use a função de treinamento do minl.py, mas logue loss, lr, entropia no wandb
    # Supondo que minl.train_minl() seja adaptável para logging externo
    # Carrega dados e modelo conforme minl.py
    # (reutiliza o código já ajustado do minl.py)
    # ...
    # Aqui, copie o loop de treinamento do minl.py, mas adicione:
    # wandb.log({"loss": avg_loss, "lr": optimizer.param_groups[0]["lr"], "entropia": entropia}, step=epoch)
    #
    # Exemplo de logging:
    # for epoch in range(epochs):
    #     ...
    #     wandb.log({"loss": avg_loss, "lr": optimizer.param_groups[0]["lr"], "entropia": entropia}, step=epoch)
    #
    # Ao final:
    # wandb.finish()
    train_minl(params)

if __name__ == "__main__":
    train_and_log()
