import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn import metrics
import networkx as nx

import argparse
import pickle

from tqdm import tqdm
import sys

import torch

import torch_geometric
import torch_geometric.nn as nn
from torch_geometric.data import Data, Batch, Dataset
import torch_geometric.loader as loader

import GE_data_preparation
import PPI_utils
import connectivity_data


class ConnNetworkData:
    def __init__(self, config):
        self.config = config
        print('Loading data...')
        self.gene_list = GE_data_preparation.get_gene_list()
        # alias_mapping = GE_data_preparation.get_alias_to_STRING_prot_mapping()
        # gene_id_to_symbol_mapping = GE_data_preparation.get_gene_id_to_symbol_mapping()

        if config.conn == 'fun':
            self.orig_structure_list = connectivity_data.get_funn_struc_list()
            self.structure_list = [struc for struc in self.orig_structure_list
                                   if struc in GE_data_preparation.get_structure_list()]

            self.conn_mat = connectivity_data.get_funn_conn_mat(threshold=0.5)
        elif config.conn == 'struc':
            self.orig_structure_list = connectivity_data.get_struc_struc_list()
            self.structure_list = [struc for struc in self.orig_structure_list
                                   if struc in GE_data_preparation.get_structure_list()]

            self.conn_mat = connectivity_data.get_struc_conn_matrix()
            self.conn_mat = self.conn_mat>0.1
        elif config.conn == 'eff':
            self.orig_structure_list = connectivity_data.get_eff_struc_list()
            self.structure_list = [struc for struc in self.orig_structure_list
                                   if struc in GE_data_preparation.get_structure_list()]

            self.conn_mat = connectivity_data.calculate_eff_conn(threshold=0.1)
        elif config.conn == 'all':
            # self.orig_structure_list = set(connectivity_data.get_funn_struc_list()) &
            self.orig_structure_list = connectivity_data.get_struc_struc_list()
            self.orig_structure_list = connectivity_data.get_eff_struc_list()


        print(f'{self.structure_list[:5]=}')

        '''
        self.protein_list = []
        gene_list = []
        for gene in self.gene_list:
            map_gene = gene_id_to_symbol_mapping[gene]
            if map_gene in alias_mapping.keys():
                prot = alias_mapping[gene_id_to_symbol_mapping[gene]]
                if prot not in self.protein_list:
                    gene_list.append(map_gene)
                    self.protein_list.append(prot)
        self.gene_list = gene_list

        print('gene_list', self.gene_list[:10])
        print('protein_list', self.protein_list[:10])
        print('structure_list', self.structure_list[:10])
        '''

        '''
        # prune protein and gene list to proteins present in STRING, i.e. nodes with at least one link
        self.PPI_graph = PPI_utils.get_PPI_graph(min_score=700)
        self.protein_list = [p for p in self.protein_list if p in self.PPI_graph.nodes()]
        '''
        self.PPI_graph = nx.Graph()
        self.protein_list = self.gene_list

        self.PPI_graph = self.PPI_graph.subgraph(self.protein_list)
        # self.gene_list = [g for g in self.gene_list if alias_mapping[g] in self.protein_list]
        print('PPI_graph nodes/edges:', len(self.PPI_graph.nodes()), len(self.PPI_graph.edges()))


        # prune structure to ones present in AMBA structure ontology
        # self.ge_data = GE_data_preparation.get_mapped_GEs(self.gene_list, self.structure_list)
        self.ge_data = GE_data_preparation.get_GEs(self.gene_list, self.structure_list)
        print(f'{self.ge_data.shape=}')

        self.num_genes = len(self.gene_list)
        self.num_structures = len(self.structure_list)

        config.num_genes = self.num_genes
        config.num_structures = self.num_structures
        print(f'Genes present: {self.num_genes}\t|\tStructures present: {self.num_structures}')
        # print('Proteins present:', len(self.protein_list))

    def build_data(self, config):
        # normalize structure representation data
        if config.expr_normalization == 'global':
            self.ge_data /= self.ge_data.max()
        elif config.expr_normalization == 'row':
            self.ge_data = self.ge_data / self.ge_data.max(axis=1).reshape(-1, 1)
        elif config.expr_normalization == 'column':
            self.ge_data = self.ge_data / self.ge_data.max(axis=0)
        else:
            raise ValueError('Invalid normalization selected!')

        # build y_data
        self.y_data = self.conn_mat

        self.y_data = torch.tensor(self.y_data, dtype=torch.float)
        self.y_data = connectivity_data.prune_conn_mat(conn_mat=self.y_data,
                                                       structure_sub_list=self.structure_list,
                                                       orig_structure_list=self.orig_structure_list)
        # print('len(self.protein_list', len(self.protein_list))
        print('y_data.size():', self.y_data.size())

        # build PyTorch geometric graph
        print("Building index dict ...")
        print(len(self.protein_list))
        self.protein_to_index_dict = {protein: index for index, protein in enumerate(self.protein_list)}
        print("Building edge list ...")
        forward_edges_list = [(self.protein_to_index_dict[node1], self.protein_to_index_dict[node2]) for node1, node2 in
                              list(self.PPI_graph.edges())]
        backward_edges_list = [(self.protein_to_index_dict[node1], self.protein_to_index_dict[node2]) for node2, node1
                               in list(self.PPI_graph.edges())]

        self.edge_list = torch.tensor(np.transpose(np.array(forward_edges_list + backward_edges_list)),
                                      dtype=torch.long)


    def get(self):
        data_list = []
        for i, _ in enumerate(self.structure_list):
            full_PPI_graph = Data(x=torch.tensor(self.ge_data[i, :], dtype=torch.float).view(-1,1),
                                       edge_index=self.edge_list)

            data_list.append(full_PPI_graph)

        return data_list

    def __len__(self):
        return self.num_structures


class Conn_dataset(Dataset):
    def __init__(self, data_list, y_data, indices):
        super(Conn_dataset, self).__init__()
        self.data_list = data_list
        self.num_structs = len(data_list)
        self.y_data = y_data

        self.indices = indices

    def get(self, index):
        index = self.indices[index]

        i = index//self.num_structs
        j = index%self.num_structs
        return self.data_list[i], self.data_list[j], self.y_data[i,j]

    def __getitem__(self, index):
        # check whether multiple indices are given
        if not isinstance(index, int):
            raise TypeError("Index is not of type int.")
        index = self.indices[index]

        i = index//self.num_structs
        j = index%self.num_structs
        return self.data_list[i], self.data_list[j], self.y_data[i,j]

    def __len__(self):
        return len(self.indices)


class Conn_SiameseNN(torch.nn.Module):
    def __init__(self, config, num_genes, conv_method, lat_dim=16, out_dim=3):
        super(Conn_SiameseNN, self).__init__()

        self.config = config
        self.num_genes = num_genes
        self.conv_method = conv_method

        self.lat_dim = lat_dim
        self.out_dim = out_dim

        self.input_linear = torch.nn.Linear(in_features=1, out_features=self.lat_dim)
        self.output_linear = torch.nn.Linear(in_features=self.lat_dim, out_features=1)
        self.red_linear = torch.nn.Linear(in_features=self.num_genes, out_features=self.out_dim, bias=False)

        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        self.relu = torch.nn.LeakyReLU(0.2)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.dropout3 = torch.nn.Dropout(0.5)

        self.norm = torch.nn.LayerNorm(num_genes)

        if conv_method == 'flat':
            pass
        elif 'GCNConv' in conv_method:
            self.conv1 = nn.GCNConv(self.lat_dim, self.lat_dim, cached=True, improved=True)
            self.conv2 = nn.GCNConv(self.lat_dim, self.lat_dim, cached=True, improved=True)
            self.conv3 = nn.GCNConv(self.lat_dim, self.lat_dim, cached=True, improved=True)
        elif 'GENConv' in conv_method:
            conv1 = nn.GENConv(self.lat_dim, self.lat_dim, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm1 = torch.nn.LayerNorm(self.lat_dim, elementwise_affine=True)

            conv2 = nn.GENConv(self.lat_dim, self.lat_dim, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm2 = torch.nn.LayerNorm(self.lat_dim, elementwise_affine=True)

            conv3 = nn.GENConv(self.lat_dim, self.lat_dim, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')

            norm3 = torch.nn.LayerNorm(self.lat_dim, elementwise_affine=True)

            act = torch.nn.LeakyReLU(0.2, inplace=True)

            self.conv1 = nn.DeepGCNLayer(conv1, norm1, act, block='res', dropout=0.5)
            self.conv2 = nn.DeepGCNLayer(conv2, norm2, act, block='res', dropout=0.5)
            self.conv3 = nn.DeepGCNLayer(conv3, norm3, act, block='res', dropout=0.5)
        elif 'GATConv' in conv_method:
            self.conv1 = nn.GATConv(self.lat_dim, self.lat_dim, heads=4, dropout=0.2, add_self_loops=False)
            self.conv2 = nn.GATConv(self.lat_dim * 4, self.lat_dim, heads=1, dropout=0.2, add_self_loops=False)


    def encode(self, data_1_x, data_1_edge_index):
        data_1_x = data_1_x.view(-1, 1)

        data_1_x = self.relu(self.input_linear(data_1_x))

        if not self.conv_method == 'flat':
            print(data_1_x.size(), data_1_edge_index.size(), data_1_edge_index.max())
            data_1_x = self.conv1(data_1_x, data_1_edge_index)
            print('bumm-1')
            # data_1_x = self.conv2(data_1_x, data_1_edge_index)

        data_1_x = self.relu(self.output_linear(data_1_x))
        data_1_x = self.dropout1(data_1_x)

        # print("cos:", self.cos(data_1_x, data_2_x))
        data_1_x = self.norm(data_1_x.view(-1, self.num_genes))
        data_1_x = self.red_linear(data_1_x)

        return data_1_x


    def forward(self, data):
        data_1_edge_index, data_1_batch, edge_attr = data.edge_index, data.batch, data.edge_attr
        data_2_edge_index, data_2_batch, edge_attr = data.edge_index, data.batch, data.edge_attr

        data_1_x = data.x
        data_2_x = data.x2

        data_1_x = data_1_x.reshape(-1,1)
        data_2_x = data_2_x.reshape(-1,1)

        data_1_x = self.encode(data_1_x, data_1_edge_index)
        data_2_x = self.encode(data_2_x, data_2_edge_index)

        ret_val =  self.cos(data_1_x, data_2_x).view(-1,1)

        return ret_val


def BCELoss_ClassWeights(input, target, pos_weight):
    # input (n, d)
    # target (n, d)
    # class_weights (1, d)
    input = torch.clamp(input,min=1e-7,max=1-1e-7)
    target = target.view(-1,1)
    weighted_bce = - pos_weight*target * torch.log(input) - 1*(1 - target) * torch.log(1 - input)
    final_reduced_over_batch = weighted_bce.sum(axis=0)
    return final_reduced_over_batch


def train(config, model, device, train_s1_loader, y_loader, optimizer, epoch, class_weight):
    num_samples = len(train_s1_loader.dataset)
    if not config.quiet and len(train_s1_loader.dataset)>1:
        print('Training on {} samples...'.format(len(train_s1_loader.dataset)))
    sys.stdout.flush()
    model.train()
    return_loss = 0

    model.train()
    for batch_idx, dataset in enumerate(zip(train_s1_loader, y_loader)):
        optimizer.zero_grad()

        s1_data = dataset[0]
        y_data = torch.tensor(np.array(dataset[1]))

        # data_mat = torch.cat([data_obj.x for dataset in [s1_data, s2_data] for data_obj in dataset], 0)
        batch = Batch.from_data_list(s1_data).to(device)
        y_data = y_data.to(device)
        # print('bumm0', len(s1_data), batch)
        output = model(batch).sigmoid()

        # my implementation of BCELoss
        output = torch.clamp(output, min=1e-7, max=1 - 1e-7)

        # model task as classification task with highly imbalanced data
        loss = BCELoss_ClassWeights(input=output.view(-1, 1),
                                    target=y_data.view(-1, 1),
                                    pos_weight=class_weight)
        loss = loss / num_samples

        return_loss += loss.item()
        loss.backward()
        optimizer.step()
        if not config.quiet and batch_idx % 10 == 0 and epoch%1==0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * output.size(0),
                                                                           len(train_s1_loader.dataset),
                                                                           100. * batch_idx / len(train_s1_loader),
                                                                           loss.item()))
            sys.stdout.flush()
    print(f'{return_loss=}')
    return return_loss

def predicting(model, device, s1_loader, y_loader, round=False, quiet_mode=False):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    if not quiet_mode:
        print('Make prediction for {} samples...'.format(len(s1_loader.dataset)))
    with torch.no_grad():
        for dataset in zip(s1_loader, y_loader):

            s1_data = dataset[0]
            y_data = torch.tensor(np.array(dataset[1]))

            # data_mat = torch.cat([data_obj.x for dataset in [s1_data, s2_data] for data_obj in dataset], 0)
            batch = Batch.from_data_list(s1_data)#.to(device)

            y_data = y_data.to(device)

            output = model(batch).sigmoid()

            total_preds = torch.cat((total_preds, output.cpu()), 0)

            total_labels = torch.cat((total_labels.view(-1, 1), y_data.view(-1, 1).float().cpu()), 0)

    if round:
        return total_labels.round().numpy().flatten(), total_preds.numpy().round().flatten()
    else:
        return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def main(config):
    # activate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    config.num_gpus = num_gpus

    np.random.seed(42)

    # build data
    config.num_proteins = None if config.num_proteins == -1 else config.num_proteins
    network_data = ConnNetworkData(config=config)

    num_genes = network_data.num_genes
    num_structures = network_data.num_structures
    # generate indices for proteins
    kf = KFold(n_splits=config.num_folds, random_state=42, shuffle=True)
    X = np.zeros((num_structures**2,1))

    network_data.build_data(config)

    dataset = network_data.get()

    print('Convolutional update method:', config.conv_method)

    results = []
    fold = 0

    model = None
    for train_pair_indices, test_pair_indices in kf.split(X):
        fold += 1
        if config.fold != -1 and fold != config.fold:
            continue
        print("Fold:", fold)

        print('Fetching data...')
        print('\nDone.\n')

        train_dataset = Conn_dataset(dataset, network_data.y_data, train_pair_indices)
        valid_dataset = Conn_dataset(dataset, network_data.y_data, test_pair_indices)

        # calculate weight of positive samples for binary classification task
        positives = network_data.y_data.reshape(-1)[train_pair_indices].sum()
        neg_to_sum_ratio = (len(train_pair_indices)-positives)/positives #Get negatives/positives ratio
        class_weight = neg_to_sum_ratio
        print('Neg/pos ratio:', neg_to_sum_ratio)

        # build DataLoaders
        print('Building data loader ...')
        train_loader = loader.DataListLoader(train_dataset, config.batch_size, shuffle=True)
        valid_loader = loader.DataListLoader(valid_dataset, config.batch_size, shuffle=False)

        print('Initializing model ...')
        model = Conn_SiameseNN(config=config,
                               num_genes=num_genes,
                               conv_method=config.conv_method,
                               lat_dim=config.lat_dim,
                               out_dim=config.emb_dim).to(device)
        # model = nn.DataParallel(model).to('cuda')
        print("Model total parameters", sum(p.numel() for p in model.parameters()))

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        sys.stdout.flush()


        # Preparing training data
        s1_data_list = []
        s2_data_list = []
        y_data_list = []
        for batch_idx, d in enumerate(train_loader):
            s1_data_list.extend([tup[0] for tup in d])
            s2_data_list.extend([tup[1] for tup in d])
            y_data_list.extend([tup[2] for tup in d])

        vectors = [s2_data.x for s2_data in s2_data_list]
        for i in range(len(vectors)):
            s1_data_list[i].x2 = vectors[i]

        s1_data_loader = loader.DataListLoader(s1_data_list, batch_size=config.batch_size, shuffle=False)
        y_data_loader = loader.DataListLoader(y_data_list, batch_size=config.batch_size, shuffle=False)

        # Preparing validation data
        valid_s1_data_list = []
        valid_s2_data_list = []
        valid_y_data_list = []
        for batch_idx, d in enumerate(valid_loader):
            valid_s1_data_list.extend([tup[0] for tup in d])
            valid_s2_data_list.extend([tup[1] for tup in d])
            valid_y_data_list.extend([tup[2] for tup in d])

        valid_vectors = [valid_s2_data.x for valid_s2_data in valid_s2_data_list]
        for i in range(len(valid_vectors)):
            valid_s1_data_list[i].x2 = valid_vectors[i]

        valid_s1_data_loader = loader.DataListLoader(valid_s1_data_list, batch_size=config.batch_size, shuffle=False)
        valid_y_data_loader = loader.DataListLoader(valid_y_data_list, batch_size=config.batch_size, shuffle=False)

        ret = None
        best_epoch = 0
        best_score = 0
        for epoch in range(1, config.num_epochs + 1):
            loss = train(config=config,
                         model=model,
                         device=device,
                         train_s1_loader=s1_data_loader,
                         y_loader=y_data_loader,
                         optimizer=optimizer,
                         epoch=epoch,
                         class_weight=class_weight)

            if epoch%5==0 and not config.quiet:
                print('Train loss:', loss)
            sys.stdout.flush()

            if epoch%5 == 0:
                if not config.quiet:
                    print('Predicting for validation data...')
                file='../results/pred_' +config.conv_method+'_fold_'+str(config.fold) + '_results'
                with open(file=file, mode='a') as f:
                    train_labels, train_predictions = predicting(model=model,
                                                               device=device,
                                                               s1_loader=s1_data_loader,
                                                               y_loader=y_data_loader,
                                                               round=False,
                                                               quiet_mode=config.quiet)

                    test_labels, test_predictions = predicting(model=model,
                                                     device=device,
                                                     s1_loader=valid_s1_data_loader,
                                                     y_loader=valid_y_data_loader,
                                                     round=False,
                                                     quiet_mode=config.quiet)

                    print('pred_eval', train_labels.max(), train_predictions.max(), train_labels.min(), train_predictions.min(), train_predictions.shape)

                    new_AUROC = metrics.roc_auc_score(test_labels, test_predictions)
                    if new_AUROC > best_score:
                        best_score = new_AUROC
                        best_epoch = epoch
                        if not config.quiet:
                            print(f"New best AUROC score: {best_score}|\tEpoch:{best_epoch}")

                    print('Train:', 'Acc, ROC_AUC, matthews_corrcoef',
                          round(metrics.accuracy_score(train_labels, train_predictions.round()),4),
                          round(metrics.roc_auc_score(train_labels, train_predictions),4),
                          round(metrics.matthews_corrcoef(train_labels, train_predictions.round()),4))

                    print('Test:', 'Acc, ROC_AUC, matthews_corrcoef',
                          round(metrics.accuracy_score(test_labels, test_predictions.round()),4),
                          round(metrics.roc_auc_score(test_labels, test_predictions),4),
                          round(metrics.matthews_corrcoef(test_labels, test_predictions.round()),4))

            sys.stdout.flush()
        print(f"Best AUROC score: {best_score}|\tEpoch:{best_epoch}")
        results.append(ret)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    model.eval()
    preds = []
    with torch.no_grad():
        for b in tqdm(range(0,len(dataset), config.batch_size)):
            batch_data = dataset[b:b+config.batch_size]
            batch = Batch.from_data_list(batch_data)

            output = model.encode(batch.x, batch.edge_index)
            preds = np.append(preds, output.detach().cpu().numpy())

    preds = preds.reshape(network_data.num_structures, config.emb_dim)

    with open(f'../results/{config.conn}_conn_{config.emb_dim}_emb_dim_embeddings.npy', 'wb') as f:
        np.save(f, preds)
    with open(f'../results/{config.conn}_conn_{config.emb_dim}_emb_dim_struc_list.txt', 'w') as f:
        for s in network_data.structure_list:
            print(s, file=f)



if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_proteins", type=int, default=-1)
    # if set to 0, model will perform regression task instead of classification
    parser.add_argument("--conn_threshold", type=float, default=0.5)
    parser.add_argument("--expr_normalization", type=str, default='column') # options are 'none', 'column', 'row' and 'global'
    parser.add_argument("--conn", type=str, default='fun') # options are 'fun', 'struc', 'eff' and 'all'

    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fold", type=int, default=-1)

    parser.add_argument("--lat_dim", type=int, default=4)
    parser.add_argument("--emb_dim", type=int, default=3)

    parser.add_argument("--conv_method", type=str, default='flat')  # options are 'flat', 'GCNConv', 'GATConv', 'GENConv', 'KerGNN'

    parser.add_argument("--quiet", action='store_true')

    config = parser.parse_args()

    # run predictor
    main(config=config)