import datetime
import argparse
import numpy as np
import dgl
import torch
from models.model import Framework
from models.utils import GraphDataLoader_Classification, \
    AUC, RMSE, MAE, \
    GraphDataLoader_Regression
from torch.optim import Adam
from data.split_data import get_classification_dataset, get_regression_dataset, scaffold_split, random_scaffold_split
import warnings
import pandas as pd
import time
import random
import optuna
import json
from models.utils import append_matrix_to_file
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial import distance
from matplotlib import pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT = f'/root'

def configure(args):
    configs = {}

    configs['dataset_name'] = args.dataset
    configs['max_iter'] = args.epoch
    configs['device'] = args.device
    configs['prompt'] = args.prompt
    configs['cluster'] = args.cluster
    configs['n_jobs'] = args.n_jobs
    configs['exp_num'] = args.exp_num
    configs['atom_in_dim'] = 37  # atom feature init dim
    configs['bond_in_dim'] = 13  # bond feature init dim
    configs['ss_node_in_dim'] = 50  # func group node feature init dim
    configs['ss_edge_in_dim'] = 37  # func group edge feature init dim
    configs['mol_in_dim'] = 167  # molecule fingerprint init dim
    configs['motif_mpnn_layers'] = 1
    configs['atom_mpnn_layers'] = 1
    configs['residual'] = True
    configs['attention'] = True  # whether to use attention pooling
    configs['input_norm'] = 'layer'

    if args.dataset in ['Tox21', 'ClinTox', 'ToxCast', 'SIDER']:
        configs['metric'] = AUC
        configs['task_type'] = 'classification'
        configs['dataset'], configs['task_pos_weights'] = get_classification_dataset(
            args.dataset, args.n_jobs)
        configs['criterion'] = torch.nn.BCEWithLogitsLoss(
            pos_weight=configs['task_pos_weights'].to(args.device))
    elif args.dataset in ['QM8', 'QM9']:
        configs['metric'] = MAE
        configs['task_type'] = 'regression'
        configs['dataset'] = get_regression_dataset(args.dataset, args.n_jobs)
        configs['criterion'] = torch.nn.MSELoss(reduction='none')

    configs['dist_loss'] = torch.nn.MSELoss(reduction='none')

    return configs


def train_evaluate(args_dict):
    def evaluate(loader):
        model.eval()
        traYAll = []
        traPredictAll = []

        for i, batch in enumerate(loader):
            gs, labels, masks = None, None, None
            if args_dict['task_type'] == 'classification':
                gs, labels, masks = batch
                masks = masks.to(args_dict['device']).float()
            else:
                gs, labels = batch
            traYAll += labels.detach().cpu().numpy().tolist()
            gs = gs.to(args_dict['device'])
            labels = labels.to(args_dict['device']).float()

            with torch.no_grad():
                logits_atom, logits_motif,atom_weights, motif_weights = model(gs, args_dict['prompt'])
            if args_dict['task_type'] == 'classification':
                logits = (torch.sigmoid(logits_atom) +
                          torch.sigmoid(logits_motif))/2
            else:
                logits = (logits_atom + logits_motif) / 2
            traPredictAll += logits.detach().cpu().numpy().tolist()

        return args_dict['metric'](np.array(traYAll), np.array(traPredictAll))


    max_score_list = []
    max_aupr_list = []

    for seed in range(args_dict['seed'], args_dict['seed'] + args_dict['exp_num']):
        print('seed:', seed)
        print('trial:', seed - args_dict['seed'])


        torch.manual_seed(seed)
        np.random.seed(seed)

        best_val_score = 0 if args_dict['task_type'] == 'classification' else 999
        best_val_aupr = 0 if args_dict['task_type'] == 'classification' else 999
        best_epoch = 0
        best_test_score = 0
        best_test_aupr = 0

        train_metric = []
        val_metric = []
        test_metric = []

        train_set, val_set, test_set = random_scaffold_split(
            dataset=args_dict['dataset'], smiles_list=args_dict['dataset'].smiles, seed=seed)
        print(
            f"{args_dict['dataset_name']}(Train Set:{len(train_set)}, Val Set:{len(val_set)}, Test Set:{len(test_set)})")
        train_loader, val_loader, test_loader = None, None, None
        if args_dict['task_type'] == 'classification':
            train_loader = GraphDataLoader_Classification(
                dataset=train_set, batch_size=args_dict['batch_size'], shuffle=True, drop_last=False)
            val_loader = GraphDataLoader_Classification(
                dataset=val_set, batch_size=args_dict['batch_size'], shuffle=False, drop_last=False)
            test_loader = GraphDataLoader_Classification(
                dataset=test_set, batch_size=args_dict['batch_size'], shuffle=False, drop_last=False)
        else:
            train_loader = GraphDataLoader_Regression(
                dataset=train_set, batch_size=args_dict['batch_size'], shuffle=True, drop_last=False)
            val_loader = GraphDataLoader_Regression(
                dataset=val_set, batch_size=args_dict['batch_size'], shuffle=False, drop_last=False)
            test_loader = GraphDataLoader_Regression(
                dataset=test_set, batch_size=args_dict['batch_size'], shuffle=False, drop_last=False)

        model = Framework(
            test_set.dataset.labels.shape[-1], args_dict).to(args_dict['device'])

        opt = Adam(model.parameters(), lr=args_dict['lr'])
        print("Total number of paramerters in networks is {} ".format(
            sum(x.numel() for x in model.parameters())))

        for epoch in range(args_dict['max_iter']):
            model.train()
            traYAll = []
            traPredictAll = []

            if epoch > 0 and args_dict['prompt'] == True and args_dict['cluster'] == True:

                model.prompter.reset_cluster_setting()
                model.prompter.HierarchicalCluster()

            for i, batch in enumerate(train_loader):
                gs, labels, masks = None, None, None
                if args_dict['task_type'] == 'classification':
                    gs, labels, masks = batch
                    masks = masks.to(args_dict['device']).float()
                else:
                    gs, labels = batch
                traYAll += labels.detach().cpu().numpy().tolist()
                gs = gs.to(args_dict['device'])
                labels = labels.to(args_dict['device']).float()

                atom_pred, fg_pred, atom_weights, motif_weights = model(gs, args_dict['prompt'])

                ##############################################
                if args_dict['task_type'] == 'classification':
                    logits = (torch.sigmoid(atom_pred) +
                              torch.sigmoid(fg_pred))/2
                    dist_atom_fg_loss = args_dict['dist_loss'](torch.sigmoid(
                        atom_pred), torch.sigmoid(fg_pred)).mean()
                else:
                    logits = (atom_pred+fg_pred)/2
                    dist_atom_fg_loss = args_dict['dist_loss'](
                        atom_pred, fg_pred).mean()

                loss_atom = args_dict['criterion'](atom_pred, labels).mean()
                loss_motif = args_dict['criterion'](fg_pred, labels).mean()
                loss = loss_motif+loss_atom + \
                    args_dict['dist_weight']*dist_atom_fg_loss
                ##################################################
                opt.zero_grad()
                loss.backward(retain_graph=True)
                opt.step()
                if args_dict['prompt'] == True and args_dict['cluster'] == True:
                    model.prompter.updateGradMatrix()
                traPredictAll += logits.detach().cpu().numpy().tolist()
            # print('*********train*********')
            train_score, train_AUPRC = args_dict['metric'](
                np.array(traYAll), np.array(traPredictAll))

            val_score, val_AUPRC = evaluate(val_loader)
            test_score, test_AUPRC = evaluate(test_loader)

            if args_dict['task_type'] == 'classification':
                if best_val_score < val_score:
                    best_val_score = val_score
                    best_test_score = test_score
                    best_epoch = epoch

                if best_val_aupr < val_AUPRC:
                    best_val_aupr = val_AUPRC
                    best_test_aupr = test_AUPRC
                    best_epoch = epoch
                print('#####################')
                print(
                    "-------------------Epoch {}-------------------".format(epoch))
                print("Train AUROC: {}".format(train_score),
                      " Train AUPRC: {}".format(train_AUPRC))
                print("Val AUROC: {}".format(val_score),
                      " Val AUPRC: {}".format(val_AUPRC))
                print("Test AUROC: {}".format(test_score),
                      " Test AUPRC: {}".format(test_AUPRC))

                train_metric.append(train_score)
                val_metric.append(val_score)
                test_metric.append(test_score)

            elif args_dict['task_type'] == 'regression':
                if best_val_score > val_score:
                    best_val_score = val_score
                    best_test_score = test_score
                    best_epoch = epoch
                print('#####################')
                print(
                    "-------------------Epoch {}-------------------".format(epoch))
                print("Train MAE: {}".format(train_score))
                print("Val MAE: {}".format(val_score))
                print('Test MAE: {}'.format(test_score))

                train_metric.append(train_score)
                val_metric.append(val_score)
                test_metric.append(test_score)

        max_score_list.append(best_test_score)
        max_aupr_list.append(best_test_aupr)
        print('best model in epoch ', best_epoch)
        print('best val score is ', best_val_score)
        print('test score in this epoch is', best_test_score)
        if args_dict['task_type'] == 'classification':
            print('best val aupr is ', best_val_aupr)
            print('corresponding best test aupr is ', best_test_aupr)

    if args_dict['task_type'] == 'classification':
        print("AUROC:\n")
    else:
        print("MAE:\n")
    print(max_score_list)
    print(np.mean(max_score_list), '+-', np.std(max_score_list))
    if args_dict['task_type'] == 'classification':
        print("AUPRC:\n")
        print(np.mean(max_aupr_list), '+-', np.std(max_aupr_list))

    return np.mean(max_score_list)


def objective(trial, configs):
    args_dict = configs
    args_dict['lr'] = trial.suggest_loguniform('learning_rate', 1e-4, 1e-3)
    args_dict['seed'] = trial.suggest_int('seed', 1, 100)
    args_dict['batch_size'] = trial.suggest_int('batch_size', 64, 200)
    args_dict['num_neurons'] = trial.suggest_int('num_neurons', 200, 300)
    args_dict['dist_weight'] = trial.suggest_float(
        'dist_weight', 0.0001, 0.001)
    args_dict['drop_rate'] = trial.suggest_float('drop_rate', 0.0, 0.5)
    args_dict['hid_dim'] = trial.suggest_categorical('hid_dim', [64, 96])
    args_dict['step'] = trial.suggest_categorical('step', [1, 2, 3])
    args_dict['agg_op'] = trial.suggest_categorical(
        'agg_op', ['max', 'mean', 'sum'])
    args_dict['mol_FP'] = trial.suggest_categorical(
        'mol_FP', ['both', 'none'])
    args_dict['gating_func'] = trial.suggest_categorical(
        'gating_func', ['Softmax', 'Sigmoid', 'Identity'])
    args_dict['heads'] = trial.suggest_categorical('heads', [8, 32])
    args_dict['init_type'] = trial.suggest_categorical(
        'init_type', ['normal', 'he', 'xavier'])
    args_dict['input_norm'] = trial.suggest_categorical(
        'input_norm', ['layer', 'batch'])

    return train_evaluate(args_dict)


def main(args):

    configs = configure(args)

    if configs['task_type'] == 'classification':
        direction = 'maximize'
    else:
        direction = 'minimize'

    study = optuna.create_study(direction=direction)
    study.optimize(lambda trial: objective(
        trial, configs), n_trials=args.n_trials)

    best_trial = study.best_trial

    print(f"best metric: {best_trial.value}")
    print(f"best param: {best_trial.params}")

    results = {
        'best_value': best_trial.value,
        'best_parameters': best_trial.params
    }

    abla = 'all'
    if configs['prompt'] == False:
        abla = 'wo_pro'
    elif configs['cluster'] == False:
        abla = 'wo_cls'

    file_path = f"{ROOT}/HiPM/results/{configs['dataset_name']}/best_trial_results_{abla}.json"
    with open(file_path, 'w') as outfile:
        json.dump(results, outfile, indent=4)

    print(f"More details have been saved at {file_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, choices=['Tox21', 'ClinTox', 'ToxCast', 'SIDER', 'QM8', 'QM9'],
                   default='Tox21', help='dataset name')
    p.add_argument('--exp_num', type=int, default=1,
                   help='the number of conducting the experiments')
    p.add_argument('--epoch', type=int, default=200, help='train epochs')
    p.add_argument('--device', type=str, default='cuda:0',
                   help='fitting device')
    p.add_argument('--n_trials', type=int, default=20, help='n_trials')
    p.add_argument('--n_jobs', type=int, default=5,
                   help='num of threads for the handle of the dataset')
    p.add_argument('--prompt', type=bool, default=True,
                   help='Add prompt or not')
    p.add_argument('--cluster', type=bool, default=True, 
                   help='cluster or not')
    args = p.parse_args()
    main(args)
