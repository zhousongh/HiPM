from functools import partial
import dgl.backend as F
import dgllife.data as dgldata
from .mol_graph import smiles_to_Molgraph, ATOM_FEATURIZER, BOND_FEATURIZER
import numpy as np
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold
from dgl.data.utils import Subset
import pandas as pd
from dgllife.data import MoleculeCSVDataset
from itertools import compress
from models.utils import zscore_normalize


def generate_scaffold(smiles, include_chirality=False):
    """ Obtain Bemis-Murcko scaffold from smiles
    :return: smiles of scaffold """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


def random_scaffold_split(dataset, smiles_list, task_idx=None, null_value=0,
                          frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/
        chainer_chemistry/dataset/splitters/scaffold_splitter.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
        examples with null value in specified task column of the data.y tensor
        prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
        task_idx is provided
    :param frac_train, frac_valid, frac_test: fractions, floats
    :param seed: seed
    :return: train, valid, test slices of the input dataset obj """

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx is not None:
        # filter based on null values in task_idx get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(
        np.array(list(scaffolds.values()), dtype=object))

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    return [Subset(dataset, train_idx),
            Subset(dataset, valid_idx),
            Subset(dataset, test_idx)]


def get_classification_dataset(dataset: str, n_jobs: int,):

    assert dataset in ['Tox21', 'ClinTox', 'ToxCast', 'SIDER']

    def get_task_pos_weights(labels, masks):
        num_pos = F.sum(labels, dim=0)
        num_indices = F.sum(masks, dim=0)
        task_pos_weights = (num_indices - num_pos) / num_pos
        return task_pos_weights

    dataset_name = dataset.lower()

    mol_g = partial(smiles_to_Molgraph)

    data = None
    file_path = rf"data/datasets/{dataset_name}.csv"
    save_path = rf"cache/{dataset_name}_dglgraph.bin"
    if dataset_name == 'sider':
        df = pd.read_csv(file_path)
        data = MoleculeCSVDataset(df, mol_g, ATOM_FEATURIZER, BOND_FEATURIZER, load=False,
                                  cache_file_path=save_path, smiles_column='smiles', n_jobs=n_jobs)
    else:
        data = getattr(dgldata, dataset)(mol_g,
                                         ATOM_FEATURIZER,
                                         BOND_FEATURIZER,
                                         n_jobs=n_jobs,
                                         cache_file_path=save_path)
    return data, get_task_pos_weights(data.labels, data.mask)


def get_regression_dataset(dataset: str, n_jobs: int):

    assert dataset in ['QM8', 'QM9']

    mol_g = partial(smiles_to_Molgraph)

    dataset_name = dataset.lower()

    data = None
    file_path = rf"data/datasets/{dataset_name}.csv"
    save_path = rf"cache/{dataset_name}_dglgraph.bin"
    df = pd.read_csv(file_path)
    if dataset_name == 'qm9':
        data = MoleculeCSVDataset(df, mol_g, ATOM_FEATURIZER, BOND_FEATURIZER, load=False, cache_file_path=save_path, smiles_column='smiles', n_jobs=n_jobs,
                                  task_names=['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv'])
    elif dataset_name == 'qm8':
        data = MoleculeCSVDataset(df, mol_g, ATOM_FEATURIZER, BOND_FEATURIZER, load=False, cache_file_path=save_path, smiles_column='smiles', n_jobs=n_jobs,
                                  task_names=['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0', 'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM'])
    else:
        data = getattr(dgldata, dataset)(mol_g,
                                         ATOM_FEATURIZER,
                                         BOND_FEATURIZER,
                                         n_jobs=n_jobs,
                                         cache_file_path=save_path)
        
    data.labels = zscore_normalize(data.labels)
    
    return data
