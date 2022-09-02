import pandas as pd
import wandb
import argparse

def combine_logs(n, random_state):
    wandb.init(project="milc-bsnip2", entity="cedwards57", name=f"R{random_state} KFOLD", settings=wandb.Settings(start_method='fork'))
    train_dfs = []
    valid_dfs = []
    for i in range(n):
        train_dfs.append(pd.read_csv(f"/data/users2/cedwards57/MILC/logs/bsnip/rs{random_state}/{i+1}_{n}/train.csv"))
        valid_dfs.append(pd.read_csv(f"/data/users2/cedwards57/MILC/logs/bsnip/rs{random_state}/{i+1}_{n}/valid.csv"))
    
    epochs = train_dfs[0].shape[0]
    final_train_df = pd.DataFrame(columns=train_dfs[0].columns)
    final_valid_df = pd.DataFrame(columns=valid_dfs[0].columns)
    for epoch in range(epochs):
        new_train_row = {}
        new_valid_row = {}
        wandb_row = {}
        for metric in train_dfs[0].columns:
            train_metrics = []
            valid_metrics = []
            for t_fold, v_fold in zip(train_dfs, valid_dfs):
                train_metrics.append(t_fold.at[epoch, metric])
                valid_metrics.append(v_fold.at[epoch, metric])
            new_train_row.update({metric: sum(train_metrics)/len(train_metrics)})
            new_valid_row.update({metric: sum(valid_metrics)/len(valid_metrics)})
            wandb_row.update({f"{metric}_epoch/train": sum(train_metrics)/len(train_metrics)})
            wandb_row.update({f"{metric}_epoch/valid": sum(valid_metrics)/len(valid_metrics)})
        wandb.log(wandb_row)
        final_train_df.append(new_train_row, ignore_index=True)
        final_valid_df.append(new_valid_row, ignore_index=True)
            
    
    final_train_df.to_csv(f"/data/users2/cedwards57/MILC/logs/bsnip/rs{random_state}/KFOLD/train.csv")
    final_valid_df.to_csv(f"/data/users2/cedwards57/MILC/logs/bsnip/rs{random_state}/KFOLD/valid.csv")
    
parser = argparse.ArgumentParser()
parser.add_argument(
    "-n",
    "--num-folds",
    dest="n",
    type=int,
    default=5,
    help="Number of folds",
)
parser.add_argument(
    "-r",
    "--random-state",
    dest="random_state",
    type=int,
    default=42,
    help="Number of folds",
)

combine_logs(args.n, args.random_state)