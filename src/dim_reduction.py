import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import umap

import argparse
from tqdm import tqdm

import visualization_utils
import GE_data_preparation
from dim_reduction_utils import ConstructUMAPGraph, UMAPLoss, Encoder, UMAPDataset


def main(config):
    # activate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    config.num_gpus = num_gpus

    np.random.seed(42)

    data = GE_data_preparation.get_ge_structure_data()  # struc x genes
    structure_list = np.array(GE_data_preparation.get_structure_list())
    gene_list = np.array(GE_data_preparation.get_gene_list())

    print(f'{data.shape=}, {len(structure_list)=}, {len(gene_list)=}')

    if config.expr_normalization == 'global':
        data /= data.max()
    elif config.expr_normalization == 'row':
        data = data / data.max(axis=1).reshape(-1, 1)
    elif config.expr_normalization == 'column':
        data = data / data.max(axis=0)
    else:
        raise ValueError('Invalid normalization selected!')
    # elif config.expr_normalization == 'none':

    # only use most expressed genes?
    num_genes = None  # set to None for all genes
    ind = np.argsort(data.sum(axis=0))[::-1][:num_genes]
    data = data[:, ind]
    print(f'Gene subset: {data.shape=}, {len(structure_list)=}, {len(gene_list)=}')

    embedding = None
    n_components = 3
    if config.dim_red == 'pca':
        pca = PCA(n_components=n_components)
        embedding = pca.fit_transform(data)
    elif config.dim_red == 'tsne':
        tsne = TSNE(n_components=n_components,
                    init='random')
        embedding = tsne.fit_transform(data)
    elif config.dim_red == 'umap':
        reducer = umap.UMAP(n_components=n_components)
        embedding = reducer.fit_transform(data)
    elif config.dim_red == 'ae':
        raise NotImplementedError
    elif config.dim_red == 'para_umap':
        graph_constructor = ConstructUMAPGraph(metric='euclidean', n_neighbors=15, batch_size=1000, random_state=42)
        epochs_per_sample, head, tail, weight = graph_constructor(data)

        dataset = UMAPDataset(data, epochs_per_sample, head, tail, weight, device='cuda', batch_size=1000)

        criterion = UMAPLoss(device=device, min_dist=0.1, batch_size=1000, negative_sample_rate=5,
                             edge_weight=None, repulsion_strength=1.0)

        model = Encoder(input_dim=data.shape[1],
                        output_dim=3).to(device)
        print(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        train_losses = []
        for epoch in range(20):
            train_loss = 0.
            for batch_to, batch_from in tqdm(dataset.get_batches()):
                optimizer.zero_grad()
                embedding_to = model(batch_to)
                embedding_from = model(batch_from)
                loss = criterion(embedding_to, embedding_from)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

            train_losses.append(train_loss)
            print('epoch: {}, loss: {}'.format(epoch, train_loss))


    else:
        raise ValueError('Invalid dimensionality reduction method selected!')

    embedding = (embedding - embedding.min())/(embedding.max() - embedding.min())

    if embedding.shape[1] == 2:
        embedding = np.hstack((embedding, np.zeros((data.shape[0], 1))))
    elif embedding.shape[1] == 1:
        embedding = np.repeat(embedding.flatten(), 3).reshape(len(embedding), 3)

    visualization_utils.visualize_embedding(data=embedding,
                                            structure_list=structure_list,
                                            save_name=config.dim_red+'_',
                                            res=config.res,
                                            slice_rescaling=True)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim_red", type=str, default='pca')

    parser.add_argument("--num_proteins", type=int, default=-1)
    # if set to 0, model will perform regression task instead of classification
    parser.add_argument("--expr_threshold", type=float, default=0.1)
    parser.add_argument("--expr_normalization", type=str,
                        default='global')  # options are 'none', 'column', 'row' and 'global'
    parser.add_argument("--res", type=int, default=50)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fold", type=int, default=-1)

    parser.add_argument("--conv_method", type=str, default='GCNConv')

    parser.add_argument("--quiet", action='store_true')

    config = parser.parse_args()

    # run predictor
    main(config=config)