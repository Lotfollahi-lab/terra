"""
Adapted from Assran, M. et al. Self-supervised learning from images with a Joint-Embedding Predictive Architecture.
Proc. IEEE Comput. Soc. Conf. Comput. Vis. Pattern Recognit. 15619–15629 (2023);
https://github.com/facebookresearch/ijepa/blob/main/main.py (05.06.2024).
"""

import argparse
import pdb
import multiprocessing as mp

import pprint
import yaml

from src.nichejepa.utils.distributed import init_distributed
from src.nichejepa.train_sweep import main as app_main
import wandb
from src.nichejepa.logistic_reg import logistic_
import pandas as pd
from src.nichejepa.nmi_ari import compute_nmi_ari

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')
parser.add_argument(
       '--seed', type=int,
        help='seed value for random initialization')
parser.add_argument(
    '--do_sweep', action='store_true',
    help='flag to enable or disable sweeping'
)
parser.add_argument(
    '--test', action='store_true',
    help='flag to enable or disable sweeping'
    )
parser.add_argument(
    '--task', type=str, required=True,
    help='name of the task to perform'
)

def process_main(rank, args, world_size, devices,data):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {args.fname}')

    # -- load script params
    params = None
    with open(args.fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
    params['seed'] = args.seed
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size),port=40313)
    logger.info(f'Running... (rank: {rank}/{world_size})')
    app_main(args=params,data=data)

def sweep_func(args):
    num_gpus = len(args.devices)
    manager = mp.Manager()
    data = manager.list()
    processes = []
    if not args.do_sweep:
       config = {
       'pred_enc_depth': 42,
       "learnable": 1,
       "ema": 0.999,
       "context_mask_size": 700,
       'n_targets': 6,
       'epochs' : 0,
       'top_k' : 127,
       'top_layer':1,
       'enc_emb_dim':768,
      }
       wandb.init(project="nichejepa-sweep",config=config)
    else:
       wandb.init(project="nichejepa-sweep")
    if args.test:
        data=[]
        process_main(0, args, num_gpus, args.devices,data)
    else:
      for rank in range(num_gpus):

         p = mp.Process(
            target=process_main,
            args=(rank, args, num_gpus, args.devices,data)
         )
         p.start()
         processes.append(p)
      for p in processes:
          p.join()
    final_df = pd.DataFrame(list(data))
    print(final_df.shape)
    print(final_df)
    #final_df.to_csv("final_df.csv", index=False)
    if args.task == 'cell_type':
       print(final_df['cell_type'].value_counts())
       df_nmi_ari = compute_nmi_ari(final_df,wandb.config.enc_emb_dim,'cell_type')
    elif args.task == 'niche_type':
        print(final_df['niche_type'].value_counts())
        df_nmi_ari = compute_nmi_ari(final_df,wandb.config.enc_emb_dim,'niche_type')
    test_f1_cell, test_f1_niche = logistic_(final_df,num_features=wandb.config.enc_emb_dim)
    if args.task == 'cell_type':    
        wandb.log({"f1_test": test_f1_cell, 'nmi_score':df_nmi_ari.loc[0,'nmi_score'], 'ari_score':df_nmi_ari.loc[0,'ari_score'], 'df_nmi_ari':df_nmi_ari})
    elif args.task == 'niche_type': 
       wandb.log({"f1_test": test_f1_niche, 'nmi_score':df_nmi_ari.loc[0,'nmi_score'], 'ari_score':df_nmi_ari.loc[0,'ari_score'], 'df_nmi_ari':df_nmi_ari})
    print(df_nmi_ari)
if __name__ == '__main__':
    __spec__ = None
    #mp.set_start_method('spawn') # TODO: uncomment
    args = parser.parse_args()
    sweep_config = {
    'method': 'random',  # 'grid' or 'bayes' are other options
    'metric': {
        'name': 'nmi_score',
        'goal': 'maximize'
    },
    'parameters': {
        #'pred_enc_depth': {'values': [41,42,43,44,51,52,53,54,55,61,62,63,64,65,66]},
        'pred_enc_depth': {'values': [31]},
        #'pred_emb_dim': {'values': [192,384,768]},
        #'epochs': {'values': [0]},
        'learnable': {'values': [1]},
        'ema': {
            'distribution': 'uniform',
            "max": 1, "min": 0},
        'enc_emb_dim': {'values': [768]},
        #'enc_emb_dim': {'values': [384]},
        #'pred_emb_dim': {'values': [768]},
        'context_mask_size': {
            'distribution': 'int_uniform',
            'min': 300,
            'max': 786
            },
        'n_targets': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 9
        },
        'target_mask_size': {
            'distribution': 'int_uniform',
            'min': 10,
            'max': 30
            },
        'epochs': {
            'distribution': 'int_uniform',
            'min': 20,
            'max': 40
        },
        'top_layer': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 3
        }
    }}
    if args.do_sweep:
      sweep_id = wandb.sweep(sweep_config, project="nichejepa-sweep")
      wandb.agent(sweep_id, function=lambda: sweep_func(args=args), count=10000)
    else:
        sweep_func(args=args)
