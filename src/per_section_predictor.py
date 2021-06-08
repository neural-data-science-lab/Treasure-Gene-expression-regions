import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
import math

import argparse
import pickle
from tqdm import tqdm
import sys

import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.data as data

import GE_data_preparation
import PPI_utils


class PerSectionNetworkData:
    def __init__(self, config):
        self.config = config

        print('Loading data...')
        gene_list = GE_data_preparation.get_gene_list()
        alias_mapping = GE_data_preparation.get_alias_to_STRING_prot_mapping()
        self.gene_list = [gene for gene in gene_list if gene in alias_mapping.keys()]
        self.protein_list = [alias_mapping[gene] for gene in self.gene_list]
        self.structure_list = GE_data_preparation.get_structure_list()

        self.num_genes = len(self.gene_list)
        self.num_structures = 1
        print(f'Genes present: {self.num_genes}|\tStructures present: {self.num_structures}')

    def build_data(self, config):
        # build protein features
        filename = 'protein_representation/DeepGOPlus/results/prot_to_encoding_dict.pkl'
        with open(file=filename, mode='rb') as f:
            protein_to_feature_dict = pickle.load(f)
        self.protein_embeddings = torch.Tensor([protein_to_feature_dict[protein] for protein in self.protein_list])

        GE_struct_mat = GE_data_preparation.get_ge_structure_data()
        max_expr_struct = np.argmax(GE_struct_mat.sum(axis=0))
        self.y_data = GE_data_preparation.get_GEs(self.gene_list, [self.structure_list[max_expr_struct]])
        self.y_data = (self.y_data > config.threshold).astype(float)
        self.y_data = torch.tensor(self.y_data)
        print('y_data.shape:', self.y_data.shape)
        print(f'Chosen structure_id: {self.structure_list[max_expr_struct]}')

        # build PyTorch geometric graph
        self.PPI_graph = PPI_utils.get_PPI_graph(min_score=700)
        print('PPI_graph nodes/edges:', len(self.PPI_graph.nodes()), len(self.PPI_graph.edges()))

        print("Building index dict ...")
        self.protein_to_index_dict = {protein: index for index, protein in enumerate(self.protein_list)}
        print("Building edge list ...")
        forward_edges_list = [(self.protein_to_index_dict[node1], self.protein_to_index_dict[node2]) for node1, node2 in
                              list(self.PPI_graph.edges())]
        backward_edges_list = [(self.protein_to_index_dict[node1], self.protein_to_index_dict[node2]) for node2, node1
                               in list(self.PPI_graph.edges())]

        self.edge_list = torch.tensor(np.transpose(np.array(forward_edges_list + backward_edges_list)),
                                      dtype=torch.long)

        print("Building feature matrix ...")
        self.train_prots = config.train_prots
        self.train_mask = np.zeros(self.num_genes)
        self.train_mask[self.train_prots] = 1

    def get(self):
        data_list = []

        # build protein mask

        y = torch.tensor(self.y_data).view(-1)

        full_PPI_graph = data.Data(x=self.protein_embeddings,
                              edge_index=self.edge_list,
                              y=y)

        data_list.append(full_PPI_graph)

        return data_list

    def __len__(self):
        return 1

class GE_PerSectionDataset(nn.Module):
    def __init__(self, data_list):
        super(GE_PerSectionDataset, self).__init__()
        self.data_list = data_list

    def get(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class GE_PerSectionPredNet(nn.Module):
    def __init__(self, config, num_prots, num_features, conv_method):
        super(GE_PerSectionPredNet, self).__init__()

        self.config = config
        self.num_prots = num_prots
        self.num_features = num_features

        self.input_linear = nn.Linear(in_features=8192, out_features=200)
        self.output_linear = nn.Linear(in_features=200, out_features=1)

        self.relu = nn.ReLU()

        if 'GCNConv' in conv_method:
            self.conv1 = torch_geometric.nn.GCNConv(200, 200, cached=True, improved=True)
            self.conv2 = torch_geometric.nn.GCNConv(200, 200, cached=True, improved=True)
            self.conv3 = torch_geometric.nn.GCNConv(200, 200, cached=True, improved=True)
        elif 'GENConv' in conv_method:
            conv1 = torch_geometric.nn.GENConv(200, 200, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm1 = torch.nn.LayerNorm(200, elementwise_affine=True)

            conv2 = torch_geometric.nn.GENConv(200, 200, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm2 = torch.nn.LayerNorm(200, elementwise_affine=True)

            conv3 = torch_geometric.nn.GENConv(200, 200, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm3 = torch.nn.LayerNorm(200, elementwise_affine=True)

            act = torch.nn.LeakyReLU(0.2, inplace=True)

            self.conv1 = torch_geometric.nn.DeepGCNLayer(conv1, norm1, act, block='res', dropout=0.5)
            self.conv2 = torch_geometric.nn.DeepGCNLayer(conv2, norm2, act, block='res', dropout=0.5)
            self.conv3 = torch_geometric.nn.DeepGCNLayer(conv3, norm3, act, block='res', dropout=0.5)

    def forward(self, PPI_data_object):
        PPI_x, PPI_edge_index, PPI_batch, edge_attr = PPI_data_object
        PPI_x = PPI_x.view(-1,8192)
        batch_size = PPI_x.size(1)
        print('batch_size:', batch_size)

        PPI_x = self.relu(self.input_linear(PPI_x))
        PPI_x = self.conv1(PPI_x, PPI_edge_index)
        PPI_x = self.conv2(PPI_x, PPI_edge_index)
        PPI_x = self.conv3(PPI_x, PPI_edge_index)

        PPI_x = self.output_linear(PPI_x)

        return PPI_x.sigmoid()

def BCELoss_ClassWeights(input, target, pos_weight):
    # input (n, d)
    # target (n, d)
    # class_weights (1, d)
    input = torch.clamp(input,min=1e-7,max=1-1e-7)
    target = target.view(-1,1)
    weighted_bce = - pos_weight*target * torch.log(input) - 1*(1 - target) * torch.log(1 - input)
    final_reduced_over_batch = weighted_bce.sum(axis=0)
    return final_reduced_over_batch

def train(config, model, device, train_loader, optimizer, epoch, neg_to_pos_ratio, train_mask):
    print('Training on {} samples...'.format(len(train_loader.dataset))) #.dataset)))
    sys.stdout.flush()
    model.train()
    return_loss = 0

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()

        output = model(data)

        y = torch.Tensor(np.array([graph_data.y.numpy() for graph_data in data])).float().to(output.device)

        # my implementation of BCELoss
        output = torch.clamp(output, min=1e-7, max=1 - 1e-7)

        pos_weight = neg_to_pos_ratio
        loss = BCELoss_ClassWeights(input=output[:, train_mask == 1].view(-1, 1),
                                    target=y[:, train_mask == 1].view(-1, 1), pos_weight=pos_weight)
        loss = loss / (config.num_proteins)

        # loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))(input=output[:, train_mask==1].view(-1, 1), target=y[:, train_mask==1].view(-1, 1),)
        # loss = nn.BCELoss(reduction='mean')(input=output[help_mask==1].view(-1, 1), target=y[help_mask==1].view(-1, 1))
        return_loss += loss
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * output.size(0),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            sys.stdout.flush()
    return return_loss

def quick_predicting(model, device, loader, round=False):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            # data = data.to(device)
            output = model(data)  # .sigmoid()
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            y = torch.Tensor(np.array([graph_data.y.numpy() for graph_data in data]))
            total_labels = torch.cat((total_labels.view(-1, 1), y.view(-1, 1).float().cpu()), 0)

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
    network_data = PerSectionNetworkData(config=config)

    num_proteins = network_data.num_genes
    num_structures = network_data.num_structures
    # generate indices for proteins
    kf = KFold(n_splits=config.num_folds, random_state=42, shuffle=True)
    X = np.zeros((num_proteins,1))

    # build for help matrix for indices
    help_matrix = np.arange(num_proteins * num_structures).reshape((num_proteins, num_structures))

    print('Convolution update method:', config.arch)

    results = []
    fold = 0
    for train_protein_indices, test_protein_indices in kf.split(X):
        fold += 1
        if config.fold != -1 and fold != config.fold:
            continue
        print("Fold:", fold)

        config.train_prots = train_protein_indices
        network_data.build_data(config)

        # build train data over whole dataset with help matrix

        print('Fetching data...')
        train_dataset = network_data.get()
        print('\nDone.\n')

        train_dataset = GE_PerSectionDataset(train_dataset)

        # Calculate weights
        positives = network_data.y_data[train_protein_indices, :].sum()
        len_to_sum_ratio = (network_data.num_structures * len(train_protein_indices)-positives)/positives #Get negatives/positives ratio
        print('Neg/pos ratio:', len_to_sum_ratio)

        train_mask = network_data.train_mask
        test_mask = 1-network_data.train_mask

        # build DataLoaders
        print('Building data loader ...')
        train_loader = data.DataListLoader(train_dataset, config.batch_size, shuffle=True)
        # valid_loader = data.DataLoader(valid_dataset, config.batch_size, shuffle=False)

        print('Initializing model ...')
        model = GE_PerSectionPredNet(config,
                                     num_prots=num_proteins,
                                     num_features=8192,
                                     conv_method=config.arch)
        model = nn.DataParallel(model).to(device)
        print("model total parameters", sum(p.numel() for p in model.parameters()))

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)#, momentum=0.9)

        sys.stdout.flush()

        ret = None
        best_AUROC = 0
        for epoch in range(1, config.num_epochs + 1):
            loss = train(config=config,
                         model=model,
                         device=device,
                         train_loader=train_loader,
                         optimizer=optimizer,
                         epoch=epoch,
                         neg_to_pos_ratio=len_to_sum_ratio,
                         train_mask=train_mask)
            print('Train loss:', loss)
            sys.stdout.flush()

            if epoch%3 == 0:
                print('Predicting for validation data...')
                file='../results/quick_pred_' +config.arch+'_fold_'+str(config.fold) + '_results'
                with open(file=file, mode='a') as f:
                    labels, predictions = quick_predicting(model, device, train_loader, round=False)

                    # get train and test predictions
                    train_labels = labels.reshape((num_proteins, num_structures))[:, train_mask==1].flatten()
                    train_predictions = predictions.reshape((num_proteins, num_structures))[:, train_mask==1].flatten()

                    test_labels = labels.reshape((num_proteins, num_structures))[train_mask==0, :].flatten()
                    test_predictions = predictions.reshape((num_proteins, num_structures))[train_mask==0, :].flatten()

                    print('pred_eval', train_labels.max(), train_predictions.max(), train_labels.min(), train_predictions.min(), train_predictions.shape)
                    print('pred_eval', test_labels.max(), test_predictions.max(), test_labels.min(), test_predictions.min(), test_predictions.shape)

                    print('Train:', 'Acc, ROC_AUC, f1, matthews_corrcoef',
                          metrics.accuracy_score(train_labels, train_predictions.round()),
                          metrics.roc_auc_score(train_labels, train_predictions),
                          metrics.matthews_corrcoef(train_labels, train_predictions.round()), file=f)

                    print('Test:', 'Acc, ROC_AUC, f1, matthews_corrcoef',
                          metrics.accuracy_score(test_labels, test_predictions.round()),
                          metrics.roc_auc_score(test_labels, test_predictions),
                          metrics.matthews_corrcoef(test_labels, test_predictions.round()), file=f)

            sys.stdout.flush()
        results.append(ret)

        if torch.cuda.is_available():
            torch.cuda.synchronize()


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_proteins", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.1)

    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--fold", type=int, default=-1)

    config = parser.parse_args()

    # run predictor
    main(config=config)