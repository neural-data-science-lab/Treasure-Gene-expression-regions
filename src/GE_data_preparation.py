import numpy as np
import networkx as nx

import pandas as pd

import pickle
import json
from tqdm import tqdm
import os

# import PPI_utils


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
    gene_list = set()
    structure_list = set()

    path = '../data/mouse_expression/structure_data/'
    # parse all genes and structures
    print('Preparing gene and structure lists...')
    for file in tqdm(list(os.listdir(path))):
        if not file.endswith('.csv'): continue
        with open(file=path+file, mode='r') as f:
            f.readline()  # skip header
            for line in f:
                _, struc_id, expr_energy, expr_dens, data_set_id, section_plane, gene_acro = line.strip().split(',')[:7]
                gene_entrez = line.strip().split(',')[-1]
                gene_list.add(gene_entrez)
                structure_list.add(struc_id)

    gene_list = list(gene_list)
    structure_list = list(structure_list)
    print(f'{len(gene_list)=}, {len(structure_list)=}')

    print('Writing gene-structure matrix...')
    ge_struc_mat = np.zeros((len(structure_list), len(gene_list)))
    for file in tqdm(list(os.listdir(path))):
        if not file.endswith('.csv'): continue
        with open(file=path + file, mode='r') as f:
            f.readline()  # skip header
            for line in f:
                _, struc_id, expr_energy, expr_dens, data_set_id, section_plane, gene_acro = line.strip().split(',')[:7]
                gene_entrez = line.strip().split(',')[-1]
                structure_index = structure_list.index(struc_id)
                gene_index = gene_list.index(gene_entrez)
                ge_struc_mat[structure_index, gene_index] = float(expr_energy)

    # Write structure_list, gene_list, expression_density_mat to disk
    path = '../data/mouse_expression/'
    with open(path + 'gene_list.pkl', mode='wb') as f:
        pickle.dump(gene_list, f, pickle.HIGHEST_PROTOCOL)

    # write structure_list
    with open(path + 'structure_list.pkl', mode='wb') as f:
        pickle.dump(structure_list, f, pickle.HIGHEST_PROTOCOL)

    # dump ge_structure_matrix to file
    with open(path + 'ge_structure_mat.pkl', mode='wb') as f:
        pickle.dump(ge_struc_mat, f, pickle.HIGHEST_PROTOCOL)
    print('Done.')


def get_ge_structure_data():
    filename = '../data/mouse_expression/ge_structure_mat.pkl'
    with open(file=filename, mode='rb') as f:
        return pickle.load(f)


def get_gene_list():
    filename = '../data/mouse_expression/gene_list.pkl'
    with open(file=filename, mode='rb') as f:
        return pickle.load(f)


def get_structure_list():
    filename = '../data/mouse_expression/structure_list.pkl'
    with open(file=filename, mode='rb') as f:
        return pickle.load(f)


def get_protein_list():
    gene_list = get_gene_list()
    gene_id_to_symbol_mapping = get_gene_id_to_symbol_mapping()
    alias_mapping = get_alias_to_STRING_prot_mapping()
    gene_list = [gene_id_to_symbol_mapping[gene] for gene in gene_list
                 if gene_id_to_symbol_mapping[gene] in alias_mapping.keys()]
    protein_list = [alias_mapping[gene] for gene in gene_list]

    return protein_list


def get_gene_id_to_symbol_mapping():
    # parses mouse atlas data and returns an internal gene_id to generic gene_symbol mapping
    # in form of a dictionary
    path = '../data/mouse_expression/'
    gene_file = 'geneInfo.csv'

    return_dict = {}

    with open(path+gene_file, mode='r') as f:
        f.readline()  # skip header
        for line in f:
            split_line = line.strip().split(',')
            gene_id, acronym = split_line[-1], split_line[1]
            return_dict[gene_id] = acronym
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


def get_GEs(gene_list=None, structure_list=None):
    # prune gene_expression structure matrix to given genes and given structures
    # returns matrix GE of shape len(structure_list) x len(gene_list)

    orig_gene_list = get_gene_list()
    if gene_list:
        gene_indices = [orig_gene_list.index(gene) for gene in gene_list]
    else:
        gene_indices = list(range(len(orig_gene_list)))

    orig_structure_list = get_structure_list()
    if structure_list:
        structure_indices = [orig_structure_list.index(struct) for struct in structure_list
                             if struct in orig_structure_list]
    else:
        structure_indices = list(range(len(orig_structure_list)))

    return get_ge_structure_data()[structure_indices,:][:,gene_indices]


def get_mapped_GEs(gene_list=None, structure_list=None):
    gene_id_to_symbol_mapping = get_gene_id_to_symbol_mapping()
    orig_gene_list = [gene_id_to_symbol_mapping[gene] for gene in get_gene_list()]

    if gene_list:
        gene_indices = [orig_gene_list.index(gene) for gene in gene_list]
    else:
        gene_indices = list(range(len(orig_gene_list)))

    orig_structure_list = get_structure_list()
    if structure_list:
        structure_indices = [orig_structure_list.index(struct) for struct in structure_list
                             if struct in orig_structure_list]
    else:
        structure_indices = list(range(len(orig_structure_list)))

    return get_ge_structure_data()[structure_indices,:][:,gene_indices]


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
        mat = np.array([[d[v] for v in G] for _, d in sim])
        # mat = 1- (mat/mat.max()) @TODO remove comment for normalization
    elif similarity_metric=='panther':
        raise NotImplementedError('See Networkx similarity panther for implementation...')
    return mat


def parse_structure_metadata():
    path = '../data/mouse_expression/structures.csv'

    return_list = []
    with open(file=path, mode='r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            line = line.strip()
            split_line = line.split(',')

            if len(split_line) > len(header):
                name = line.split('"')[1]
                first_occ = line.index('"')
                split_line = line[:first_occ] + line[line.index('"', first_occ+1)+1:]
                split_line = split_line.split(',')
                vals = {}
                for i, col in enumerate(header):
                    vals[col] = split_line[i]
                vals['name'] = name
                return_list.append(vals)

            else:
                vals = {}
                for i,col in enumerate(header):
                    vals[col] = split_line[i]
                return_list.append(vals)

    return return_list


if __name__ == '__main__':
    # write_ge_data()
    l = parse_structure_metadata()
    print([i for i in l if i['hemisphere_id'] != '3'])

    # graph, mapping = parse_structure_ontology()
    # print(len(graph.nodes()), len(graph.edges()))




