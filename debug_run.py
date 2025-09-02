# One-epoch smoke test to trigger debug diagnostics in minl.train_minl
import traceback
try:
    from types import SimpleNamespace
    from minl import train_minl

    params = SimpleNamespace()
    params.res_ver = 432
    params.res_hor = 622
    params.num_views_ver = 11
    params.num_views_hor = 11
    params.epochs = 1
    params.batch_size = 8
    params.lr_start = 1e-2
    params.lr_final = 1e-4
    params.wandb_active = False

    print('Starting debug_run: calling train_minl(params)')
    train_minl(params)
    print('debug_run finished')
except Exception as e:
    print('Exception in debug_run:')
    traceback.print_exc()
