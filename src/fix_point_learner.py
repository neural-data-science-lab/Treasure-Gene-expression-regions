import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.cluster import KMeans

import argparse
import pickle
from tqdm import tqdm
import sys

import torch
import torch_geometric
import torch_geometric.nn as nn
import torch_geometric.data as data
import torch_geometric.loader as loader

import GE_data_preparation
import PPI_utils


class PerSectionNetworkData:
    def __init__(self, config):
        self.config = config

        print('Loading data...')
        gene_list = GE_data_preparation.get_gene_list()
        alias_mapping = GE_data_preparation.get_alias_to_STRING_prot_mapping()
        gene_id_to_symbol_mapping = GE_data_preparation.get_gene_id_to_symbol_mapping()

        self.gene_list = [gene_id_to_symbol_mapping[gene] for gene in gene_list
                          if gene_id_to_symbol_mapping[gene] in alias_mapping.keys()]
        self.protein_list = [alias_mapping[gene] for gene in self.gene_list]

        self.structure_list = GE_data_preparation.get_structure_list() # full structure list
        struc_onto_graph, struc_annotation_mapping = GE_data_preparation.get_structure_ontology()
        # prune structure to ones present in AMBA structure ontology
        self.structure_list = [struc for struc in self.structure_list if struc in struc_onto_graph.nodes()]
        # prune structure list w.r.t. threshold w.r.t. similarity metric specified
        GE_struc_mat = GE_data_preparation.get_GEs(self.gene_list, self.structure_list)
        max_expr_struc_ind = np.argsort(GE_struc_mat.sum(axis=1))[-1] # sort struct indices by maximum expressions
        max_expr_struc = self.structure_list[max_expr_struc_ind]

        print('max_expr_struc', max_expr_struc)

        struc_similarity_matrix = GE_data_preparation.get_similarity_matrix(struc_onto_graph, similarity_metric='dist')
        struc_neighbourhood = np.array(list(struc_onto_graph.nodes()))[struc_similarity_matrix[list(struc_onto_graph.nodes()).index(max_expr_struc),:]<=config.struc_sim_threshold]

        self.structure_list = [struc for struc in self.structure_list if struc in struc_neighbourhood]

        print('gene_list', self.gene_list[:10])
        print('protein_list', self.protein_list[:10])
        print('structure_list', self.structure_list[:10])

        # prune protein and gene list to proteins present in STRING, i.e. nodes with at least one link
        self.PPI_graph = PPI_utils.get_PPI_graph(min_score=700)
        self.PPI_graph = self.PPI_graph.subgraph(self.protein_list)
        self.protein_list = [p for p in self.protein_list if p in self.PPI_graph.nodes()]
        self.gene_list = [g for g in self.gene_list if alias_mapping[g] in self.protein_list]
        print('PPI_graph nodes/edges:', len(self.PPI_graph.nodes()), len(self.PPI_graph.edges()))

        self.num_genes = len(self.gene_list)
        self.num_structures = len(self.structure_list)

        config.num_genes = self.num_genes
        config.num_structures = self.num_structures
        print(f'Genes present: {self.num_genes}\t|\tStructures present: {self.num_structures}')
        print('Proteins present:', len(self.protein_list))

    def build_data(self, config):
        self.y_data = GE_data_preparation.get_GEs(self.gene_list, self.structure_list)

        if config.expr_normalization == 'none':
            pass
        elif config.expr_normalization == 'global':
            # normalize by globally maximum value
            self.y_data = self.y_data / self.y_data.max()
        elif config.expr_normalization == 'row':
            # normalize row-wise, i.e. w.r.t. each structure
            self.y_data = self.y_data/self.y_data.max(axis=1).reshape(-1,1)
        elif config.expr_normalization == 'column':
            # normalize column-wise, i.e. w.r.t. each gene
            self.y_data = self.y_data / self.y_data.max(axis=0).reshape(1, -1)
        else:
            raise ValueError("Unsupported normalization mode selected for config.expr_normalization!")

        if not config.expr_threshold == 0:
            # apply plain threshold
            self.y_data = (self.y_data > config.expr_threshold)

        self.y_data = torch.tensor(self.y_data).float()

        print('len(self.protein_list', len(self.protein_list))
        print('y_data.size():', self.y_data.size())

        # build protein features
        self.protein_embeddings = torch.transpose(self.y_data, 0, 1)

        self.embedding_size = self.protein_embeddings.size()[1]
        config.embedding_size = self.embedding_size

        # build PyTorch geometric graph
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
        self.train_mask = np.zeros(self.num_genes).astype(int)
        self.train_mask[self.train_prots] = 1

        # cluster self.y_data values with kmeans for sample weighting
        if config.expr_threshold == 0:
            kmeans = KMeans(n_clusters=5, random_state=42).fit(self.y_data.numpy()[:,self.train_mask].reshape(-1,1))
            self.cluster_ids = kmeans.predict(self.y_data.numpy().reshape(-1,1)).reshape(self.y_data.shape)

            self.cluster_ids = torch.tensor(self.cluster_ids)

    def get(self):
        data_list = []

        y = self.y_data.view(self.num_structures, self.num_genes)
        for i, _ in enumerate(self.structure_list):
            full_PPI_graph = data.Data(x=self.protein_embeddings,
                                  edge_index=self.edge_list,
                                  y=y[i, :].view(1,-1))
            if self.config.expr_threshold == 0:
                full_PPI_graph.cluster_ids = self.cluster_ids[i, :]
            data_list.append(full_PPI_graph)
        return data_list

    def __len__(self):
        return self.num_structures

class GE_PerSectionDataset(data.Dataset):
    def __init__(self, data_list):
        super(GE_PerSectionDataset, self).__init__()
        self.data_list = data_list

    def get(self, index):
        return self.data_list[index]

    def __getitem__(self, index):
        # check whether multiple indices are given
        if not isinstance(index, int):
            raise TypeError("Index is not of type int.")
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)

    def _download(self):
        pass

    def _process(self):
        pass


class GE_FixPoint_PredNet(torch.nn.Module):
    def __init__(self, config, num_prots, num_features, conv_method):
        super(GE_FixPoint_PredNet, self).__init__()

        self.config = config
        self.num_prots = num_prots
        self.num_features = num_features
        self.conv_method = conv_method

        self.lat_size = 10

        self.input_linear = torch.nn.Linear(in_features=self.num_features, out_features=self.lat_size)
        self.output_linear = torch.nn.Linear(in_features=self.lat_size, out_features=1)

        self.relu = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.dropout3 = torch.nn.Dropout(0.5)

        if conv_method == 'flat':
            pass
        elif 'GCNConv' == conv_method:
            self.conv1 = nn.GCNConv(self.lat_size, self.lat_size, cached=True, improved=True)
            self.conv2 = nn.GCNConv(self.lat_size, self.lat_size, cached=True, improved=True)
            self.conv3 = nn.GCNConv(self.lat_size, self.lat_size, cached=True, improved=True)
        elif 'GCNConv_pruned' == conv_method:
            self.conv1 = nn.GCNConv(self.lat_size, self.lat_size, cached=True, add_self_loops=False)
            self.conv2 = nn.GCNConv(self.lat_size, self.lat_size, cached=True, add_self_loops=False)
            self.conv3 = nn.GCNConv(self.lat_size, self.lat_size, cached=True, add_self_loops=False)
        elif 'SAGEConv' == conv_method:
            self.conv1 = nn.SAGEConv(self.lat_size, self.lat_size)
            self.conv2 = nn.SAGEConv(self.lat_size, self.lat_size)
            self.conv3 = nn.SAGEConv(self.lat_size, self.lat_size)
        elif 'GENConv' in conv_method:
            conv1 = nn.GENConv(self.lat_size, self.lat_size, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm1 = torch.nn.LayerNorm(self.lat_size, elementwise_affine=True)

            conv2 = nn.GENConv(self.lat_size, self.lat_size, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm2 = torch.nn.LayerNorm(self.lat_size, elementwise_affine=True)

            conv3 = nn.GENConv(self.lat_size, self.lat_size, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')

            norm3 = torch.nn.LayerNorm(self.lat_size, elementwise_affine=True)

            act = torch.nn.LeakyReLU(0.2, inplace=True)

            self.conv1 = nn.DeepGCNLayer(conv1, norm1, act, block='res', dropout=0.5)
            self.conv2 = nn.DeepGCNLayer(conv2, norm2, act, block='res', dropout=0.5)
            self.conv3 = nn.DeepGCNLayer(conv3, norm3, act, block='res', dropout=0.5)

    def forward(self, data):
        PPI_x, PPI_edge_index, PPI_batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        PPI_x = PPI_x.view(-1,self.num_features)

        PPI_x = self.relu(self.input_linear(PPI_x))
        # PPI_x = self.dropout1(PPI_x)
        if not self.conv_method=='flat':
            PPI_x = self.conv1(PPI_x, PPI_edge_index)
            PPI_x = self.relu(PPI_x)
            PPI_x = self.conv2(PPI_x, PPI_edge_index)
            PPI_x = self.relu(PPI_x)
            # PPI_x = self.conv3(PPI_x, PPI_edge_index)

        PPI_x = self.output_linear(PPI_x)

        # PPI_x =  PPI_x.sigmoid()

        return PPI_x.view(-1, self.num_prots, 1)


def BCELoss_ClassWeights(input, target, pos_weight):
    # input (n, d)
    # target (n, d)
    # class_weights (1, d)
    input = torch.clamp(input,min=1e-7,max=1-1e-7)
    target = target.view(-1,1)
    weighted_bce = - pos_weight*target * torch.log(input) - 1*(1 - target) * torch.log(1 - input)
    final_reduced_over_batch = weighted_bce.sum(axis=0)
    return final_reduced_over_batch

def weighted_l1_loss(pred, target, weight):
    # pred (n, 1)
    # target (n, 1)
    # weight (n, 1)
    target = target.view(-1, 1)

    l1_loss = torch.abs(pred-target) * weight
    reduced_l1_loss = l1_loss.sum()
    return reduced_l1_loss

def train(config, model, device, train_loader, optimizer, epoch, class_weight, train_mask):
    if not config.quiet and len(train_loader.dataset)>1:
        print('Training on {} samples...'.format(len(train_loader.dataset)))
    sys.stdout.flush()
    model.train()
    return_loss = 0

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()

        output = model(data).squeeze()

        y = torch.cat([graph_data.y for graph_data in data]).float().to(output.device).squeeze()

        if not config.expr_threshold == 0:
            pos_weight = class_weight
            # apply sigmoid only in case of binary classification
            output = output.sigmoid()

            # my implementation of BCELoss
            output = torch.clamp(output, min=1e-7, max=1 - 1e-7)

            # model task as classification task with highly imbalanced data
            loss = BCELoss_ClassWeights(input=output[:, train_mask == 1].view(-1, 1),
                                        target=y[:, train_mask == 1].view(-1, 1),
                                        pos_weight=pos_weight)
            loss = loss / (config.num_genes * config.num_structures)
        else:
            # model task as regression task with highly imbalanced data
            cluster_ids = torch.Tensor(np.array([graph_data.cluster_ids.numpy() for graph_data in data]))

            num_clusters = len(class_weight)
            class_weight = class_weight / class_weight.max()

            # distribute respective class weights calculated by k-means to each label
            point_wise_weights = torch.zeros(cluster_ids.size())
            for i in range(num_clusters):
                point_wise_weights[cluster_ids==i] = class_weight.sum()/class_weight[i]

            # output = output.sigmoid()

            loss = weighted_l1_loss(pred=output[:, train_mask == 1].view(-1, 1),
                                    target=y[:, train_mask == 1].view(-1, 1),
                                    weight=point_wise_weights[:, train_mask == 1].to(output.device).view(-1, 1))
            # calculate mean over all genes and present structures
            loss = loss / (config.num_genes * config.num_structures)

        return_loss += loss
        loss.backward()
        optimizer.step()
        if not config.quiet and batch_idx % 10 == 0 and epoch%1==0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * output.size(0),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            sys.stdout.flush()
    return return_loss

def quick_predicting(model, device, loader, round=False, quiet_mode=False):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    if not quiet_mode:
        print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            output = model(data) # .sigmoid()
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            y = torch.cat([graph_data.y for graph_data in data])
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

    print('Convolutional update method:', config.conv_method)

    results = []
    fold = 0
    for train_protein_indices, test_protein_indices in kf.split(X):
        fold += 1
        if config.fold != -1 and fold != config.fold:
            continue
        print("Fold:", fold)

        config.train_prots = train_protein_indices
        network_data.build_data(config)

        print('Fetching data...')
        train_dataset = network_data.get()
        print('\nDone.\n')

        train_dataset = GE_PerSectionDataset(train_dataset)

        # Calculate weights for imbalanced data
        if config.expr_threshold==0:
            # calculate prevalence of each cluster calculated by k-means for regression task
            num_clusters = len(network_data.cluster_ids.unique())
            class_weight = np.zeros(num_clusters)
            for i in range(num_clusters):
                class_weight[i] = (network_data.cluster_ids.numpy()==i).sum()
            class_weight = class_weight/class_weight.sum()
            print('Class weight:', class_weight)

        else:
            # calculate weight of positive samples for binary classification task
            positives = network_data.y_data[:, train_protein_indices].sum()
            len_to_sum_ratio = (network_data.num_structures * len(train_protein_indices)-positives)/positives #Get negatives/positives ratio
            class_weight = len_to_sum_ratio
            print('Neg/pos ratio:', len_to_sum_ratio)

        train_mask = network_data.train_mask
        test_mask = 1-network_data.train_mask

        # build DataLoaders
        print('Building data loader ...')
        train_loader = loader.DataListLoader(train_dataset, config.batch_size, shuffle=True)
        # valid_loader = loader.DataLoader(valid_dataset, config.batch_size, shuffle=False)

        print('Initializing model ...')
        model = GE_FixPoint_PredNet(config,
                                    num_prots=num_proteins,
                                    num_features=config.embedding_size,
                                    conv_method=config.conv_method)
        model = nn.DataParallel(model).to('cuda')
        print("Model total parameters", sum(p.numel() for p in model.parameters()))

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)#, momentum=0.9)

        sys.stdout.flush()

        ret = None
        best_epoch = 0
        best_score = math.inf if config.expr_threshold == 0 else 0
        for epoch in range(1, config.num_epochs + 1):
            loss = train(config=config,
                         model=model,
                         device=device,
                         train_loader=train_loader,
                         optimizer=optimizer,
                         epoch=epoch,
                         class_weight=class_weight,
                         train_mask=train_mask)

            if epoch%5==0 and config.quiet:
                print('Train loss:', loss)
            sys.stdout.flush()

            if epoch%5 == 0:
                if not config.quiet:
                    print('Predicting for validation data...')
                file='../results/quick_pred_' +config.conv_method+'_fold_'+str(config.fold) + '_results'
                with open(file=file, mode='a') as f:
                    labels, predictions = quick_predicting(model, device, train_loader, round=False, quiet_mode=config.quiet)

                    # get train and test predictions
                    train_labels = labels.reshape((num_structures, num_proteins))[:, train_mask==1].flatten()
                    train_predictions = predictions.reshape((num_structures, num_proteins))[:, train_mask==1].flatten()

                    test_labels = labels.reshape((num_structures, num_proteins))[:, train_mask==0].flatten()
                    test_predictions = predictions.reshape((num_structures, num_proteins))[:, train_mask==0].flatten()

                    # print('pred_eval', train_labels.max(), train_predictions.max(), train_labels.min(), train_predictions.min(), train_predictions.shape)
                    # print('pred_eval', test_labels.max(), test_predictions.max(), test_labels.min(), test_predictions.min(), test_predictions.shape)

                    if config.expr_threshold == 0:
                        for data in train_loader:
                            cluster_ids = torch.Tensor(np.array([graph_data.cluster_ids.numpy() for graph_data in data]))
                        num_clusters = len(class_weight)

                        norm_class_weight = class_weight / class_weight.max()
                        # distribute respective class weights calculated by k-means to each label
                        point_wise_weights = torch.zeros(cluster_ids.size())
                        for i in range(num_clusters):
                            point_wise_weights[cluster_ids == i] = norm_class_weight.sum()/norm_class_weight[i]

                        train_loss = weighted_l1_loss(pred=torch.tensor(train_predictions),
                                                      target=torch.tensor(train_labels),
                                                      weight=point_wise_weights[:, train_mask==1].view(-1))
                        train_loss = train_loss/(config.num_genes * config.num_structures)

                        test_loss = weighted_l1_loss(pred=torch.tensor(test_predictions),
                                                     target=torch.tensor(test_labels),
                                                     weight=point_wise_weights[:, train_mask == 0].view(-1))
                        test_loss = test_loss / (config.num_genes * config.num_structures)

                        print(f'Train Loss: {train_loss}|\tTest Loss: {test_loss}')
                        print('min/max preds:', train_predictions.min(), train_predictions.max())

                        if test_loss < best_score:
                            best_score = test_loss
                            best_epoch = epoch
                            print(f"New best AUROC score: {best_score}|\tEpoch:{best_epoch}")
                    else:
                        new_AUROC = metrics.roc_auc_score(test_labels, test_predictions)
                        if new_AUROC > best_score:
                            best_score = new_AUROC
                            best_epoch = epoch
                            if not config.quiet:
                                print(f"New best AUROC score: {best_score}|\tEpoch:{best_epoch}")

                        if not config.quiet:
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


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_proteins", type=int, default=-1)
    # if set to 0, model will perform regression task instead of classification
    parser.add_argument("--expr_threshold", type=float, default=0.1)
    parser.add_argument("--expr_normalization", type=str, default='global') # other options are 'none', 'column' and 'row'
    parser.add_argument("--struc_sim_threshold", type=float, default=2.0) # similarity measure is distance over structure ontology graph

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