import numpy as np

from tqdm import tqdm
import os

from .. import GE_data_preparation


def parse_MGI_MP_annotations():
    path = '../../data/DL2vec_data/'
    filename = 'MGI_GenePheno.rpt'

    MGI_entity_list = []
    annotation_list = []
    with open(file=path+filename, mode='r') as f:
        for line in f:
            split_line = line.split('\t')
            MGI_entity, MP_class = split_line[2], split_line[4]
            annotation_list.append((MGI_entity, MP_class))
            MGI_entity_list.append(MGI_entity)

    print('MGI_entities present:', len(MGI_entity))
    print('MGI annotations present:', len(annotation_list))

    return MGI_entity_list, annotation_list

def parse_MGI_GO_annotations():
    path = '../../data/DL2vec_data/'
    filename = 'gene_association.mgi'

    MGI_entity_list = []
    annotation_list = []
    with open(file=path + filename, mode='r') as f:
        for line in f:
            # skip header lines and comments
            if line.startswith('!'):
                continue

            split_line = line.split('\t')
            MGI_entity, relation_type, GO_class = split_line[1], split_line[3], split_line[4]
            if 'NOT' not in relation_type:
                annotation_list.append((MGI_entity, GO_class))
                MGI_entity_list.append(MGI_entity)

    print('MGI_entities present:', len(MGI_entity))
    print('GO annotations present:', len(annotation_list))
    print('GO classes present:', len(set([tup[1] for tup in annotation_list])))

    return MGI_entity_list, annotation_list

def write_annotation_files():
    path = '../../data/DL2vec_data/'
    onto_prefix = "<http://purl.obolibrary.org/obo/{entity}>"

    MP_MGI_entities, MP_annotations = parse_MGI_MP_annotations()
    GO_MGI_entities, GO_annotations = parse_MGI_GO_annotations()

    alias_mapping = GE_data_preparation.get_alias_to_STRING_prot_mapping()
    alias_mapping = {k:v for k,v in alias_mapping.items() if 'MGI' in k} # prune alias mapping to MGI names

    print('Writing annotation files to disc...')

    # write MP annotations
    filename = 'MP_annotations'
    print(f'... writing {path+filename}.')
    with open(file=path+filename, mode='w') as f:
        for MGI_entity, MP_term in MP_annotations:
            if MGI_entity in alias_mapping.keys():
                MGI_entity = alias_mapping[MGI_entity]
                onto_term = onto_prefix.format(entity=MP_term.replace(':', '_'))
                f.write(MGI_entity+' '+onto_term)

    # write GO annotations
    filename = 'GO_annotations'
    print(f'... writing {path+filename}.')
    with open(file=path + filename, mode='w') as f:
        for MGI_entity, GO_term in GO_annotations:
            if MGI_entity in alias_mapping.keys():
                MGI_entity = alias_mapping[MGI_entity]
                onto_term = onto_prefix.format(entity=GO_term.replace(':', '_'))
                f.write(MGI_entity + ' ' + onto_term)

    print('Done.')

def output_example_DL2vec_command(prefix='MP_', workers=48, embedsize=200):
    print('Running DL2vec embedding generator ...')
    path = '../../../data/DL2vec_data/'
    onto = path + 'phenomenet.owl'
    asso = path + (prefix + '_' if not prefix.endswith('_') else '')  + 'annotation_file'
    outfile = path + (prefix + '_' if not prefix.endswith('_') else '') + 'embedding_model'
    ents = path + prefix + 'entity_list'
    command = 'python runDL2vec.py -embedsize {embedsize} -ontology {onto} -associations {asso} -outfile {outfile} -entity_list {ents} -num_workers {num_workers}'.format(
        embedsize=embedsize,
        onto=onto,
        asso=asso,
        outfile=outfile,
        ents=ents,
        num_workers=workers)

    print('Command:', command)


def test_parse_files():
    MGI_list, anno_list = parse_MGI_MP_annotations()
    print(MGI_list[:5], anno_list[:5])
    print('---------------------------------------------------------')

    MGI_list, anno_list = parse_MGI_GO_annotations()
    print(MGI_list[:5], anno_list[:5])


if __name__ == '__main__':
    test_parse_files()
