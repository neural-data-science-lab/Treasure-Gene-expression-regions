import numpy as np
import networkx as nx

import pandas as pd

import pickle
import json
from tqdm import tqdm
import copy

import PPI_utils


def get_protein_to_EnsemblProtein_id():
    protein_to_Ensembl_protein_id_mapping = {}

    filename = '../data/STRING_data/10090.protein.aliases.v11.0.txt'
    with open(file=filename, mode='r') as f:
        # skip header
        f.readline()

        for line in tqdm(f, total=2224814): #@TODO total not accurate anymore
            protein_id, alias, source = line.strip().split('\t')
            if source.strip() == 'Ensembl_protein_id':
                protein_to_Ensembl_protein_id_mapping[protein_id.strip()] = alias.strip()

    return protein_to_Ensembl_protein_id_mapping

def write_ge_data():
    path = '../data/mouse_expression/'

    gene_file = 'mouse_expression_data_sets.csv'
    full_gene_data = pd.read_csv(path+gene_file, index_col=0)

    '''
    For all gene dataset pairs:
        - add gene to list if not present
        - look at grid_file url
        - go to subdir
        - parse structure unionize file for intensity values
    '''


    print('Parsing mouse gene expression data...')
    gene_list = []
    structure_list = []
    for index, row in tqdm(full_gene_data.iterrows(), total=25898):
        if not row['plane_of_section'] == 'coronal':
            continue

        # collect present genes and sections
        gene = str(row['gene_id'])
        if gene not in gene_list:
            gene_list.append(gene)

        file_sub_path = row['structure_unionizes_file_url'].split('mouse_expression')[1][1:]
        section_data = pd.read_csv(path+file_sub_path, index_col=0)
        for section_index, section_row in section_data.iterrows():
            structure_id = str(int(section_row['structure_id']))
            if structure_id not in structure_list:
                structure_list.append(structure_id)

    print('Done.')
    print(f'Genes: {len(gene_list)}|\tStructures: {len(structure_list)}')

    # re-parse all files for intensity values
    print('Parsing expression values...')
    ge_structure_data = np.zeros((len(gene_list), len(structure_list)))
    for _, row in tqdm(full_gene_data.iterrows(), total=25898):
        if not row['plane_of_section'] == 'coronal':
            continue

        gene = str(row['gene_id'])
        gene_index = gene_list.index(gene)

        file_sub_path = row['structure_unionizes_file_url'].split('mouse_expression')[1][1:]
        section_data = pd.read_csv(path + file_sub_path, index_col=0)
        for section_index, section_row in section_data.iterrows():
            structure_id = str(int(section_row['structure_id']))
            structure_index = structure_list.index(structure_id)

            expression_density = float(section_row['expression_density'])
            ge_structure_data[gene_index, structure_index] = expression_density
    print('Done.')

    print('Writing parsed files to disk...')
    # write gene_list
    with open(path+'gene_list.pkl', mode='wb') as f:
        pickle.dump(gene_list, f, pickle.HIGHEST_PROTOCOL)

    # write structure_list
    with open(path+'structure_list.pkl', mode='wb') as f:
        pickle.dump(structure_list, f, pickle.HIGHEST_PROTOCOL)

    # dump ge_structure_matrix to file
    with open(path+'ge_structure_mat.pkl', mode='wb') as f:
        pickle.dump(ge_structure_data, f, pickle.HIGHEST_PROTOCOL)
    print('Done.')

def get_gene_list():
    filename = '../data/mouse_expression/gene_list.pkl'
    with open(file=filename, mode='rb') as f:
        return pickle.load(f)

def get_structure_list():
    filename = '../data/mouse_expression/structure_list.pkl'
    with open(file=filename, mode='rb') as f:
        return pickle.load(f)

def get_ge_structure_data():
    filename = '../data/mouse_expression/ge_structure_mat.pkl'
    with open(file=filename, mode='rb') as f:
        return pickle.load(f)

def get_gene_id_to_symbol_mapping():
    # parses mouse atlas data and returns an internal gene_id to generic gene_symbol mapping
    # in form of a dictionary
    path = '../data/mouse_expression/'
    gene_file = 'mouse_expression_data_sets.csv'
    full_gene_data = pd.read_csv(path + gene_file, index_col=0)

    return_dict = {}
    for _, row in full_gene_data.iterrows():
        return_dict[str(row['gene_id'])] = row['gene_symbol']

    return return_dict

def get_alias_to_STRING_prot_mapping():
    # Parses STRING alias file and returns an alias to STRING_id protein dictionary
    filename = '../data/STRING_data/10090.protein.aliases.v11.0.txt'

    print('Parsing protein aliases...')
    return_dict = {}
    with open(file=filename, mode='r') as f:
        # skip header
        f.readline()

        for line in f:
            prot, alias, source = line.strip().split('\t')
            return_dict[alias.strip()] = prot.strip()

    return return_dict

def get_DeepGOPlus_feature_dict():
    # build protein features
    filename = 'protein_representation/DeepGOPlus/results/prot_to_encoding_dict'
    with open(file=filename + '.pkl', mode='rb') as f:
        return pickle.load(f)

def get_GEs(gene_list, structure_list):
    # prune gene_expression structure matrix to given genes and given structures
    # returns matrix GE of shape len(structure_list) x len(gene_list)
    gene_id_to_symbol_mapping = get_gene_id_to_symbol_mapping()
    orig_gene_list = [gene_id_to_symbol_mapping[gene] for gene in get_gene_list()]

    gene_indices = [orig_gene_list.index(gene) for gene in gene_list]

    orig_structure_list = get_structure_list()
    structure_indices = [orig_structure_list.index(struct) for struct in structure_list]

    return np.transpose(get_ge_structure_data()[gene_indices,:][:,structure_indices])

def get_structure_ontology():
    """
    Parses the Allen Mouse Brain Atlas structural ontology and returns the corresponding networkx
    graph together with a mapping from structure id to annotated data, e.g. name, hemisphere, ...
    :return:        nx graph representing ontology, dict from ids to dict representing annotation
    """

    # See https://rdrr.io/github/AllenInstitute/cocoframer/src/R/ontology.R
    # Download link: http://api.brain-map.org/api/v2/structure_graph_download/1.json
    filename = '../data/mouse_expression/1.json'

    print('Parsing structure ontology json...')
    onto_graph = nx.DiGraph()
    id_to_data_mapping = {}
    with open(file=filename, mode='r') as f:
        def rec_children_parsing(structure_entity):
            id = str(structure_entity['id'])
            if id not in id_to_data_mapping.keys():
                id_to_data_mapping[id] = structure_entity.copy()
                id_to_data_mapping[id].pop('children', None)
            else:
                print('Same ID is multiple times in 1.json:', id)

            if structure_entity['children']:
                for child in structure_entity['children']:
                    onto_graph.add_edge(id, str(child['id']), label='has_child')
                    onto_graph.add_edge(str(child['id']), id, label='has_parent')
                    rec_children_parsing(child)

        onto_json = json.load(f)
        rec_children_parsing(onto_json['msg'][0])
    return onto_graph, id_to_data_mapping

def get_similarity_matrix(G, similarity_metric='dist'):
    # Takes graph and metric for measuring similarity type as inputs and calculates
    # the corresponding similarity matrix ordered w.r.t. G.nodes()/inherent node ordering

    if similarity_metric=='simrank':
        # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.similarity.simrank_similarity.html#networkx.algorithms.similarity.simrank_similarity
        sim = nx.simrank_similarity(G, max_iterations=10)
        mat = np.array([[sim[u][v] for v in G] for u in G])
    elif similarity_metric=='dist':
        sim = nx.all_pairs_shortest_path_length(G)
        mat = np.array([[d[v] for v in G]  for _, d in sim])
        # mat = 1- (mat/mat.max()) @TODO remove comment for normalization
    elif similarity_metric=='panther':
        raise NotImplementedError('See Networkx similarity panther for implementation...')
    return mat


if __name__ == '__main__':
    write_ge_data()

    # graph, mapping = parse_structure_ontology()
    # print(len(graph.nodes()), len(graph.edges()))

