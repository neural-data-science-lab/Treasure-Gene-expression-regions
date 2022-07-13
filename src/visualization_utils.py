import numpy as np
import nrrd
import networkx as nx

from PIL import Image
import matplotlib.pyplot as plt

import os
from tqdm import tqdm

import GE_data_preparation


def get_voxel_to_section_map(res=50):
    if res not in [10, 25, 50, 100]:
        raise ValueError('Invalid resolution for annotation volume selected!')

    path = '../data/CCFv3/ccf_2017/'
    ANO, metaANO =nrrd.read(path + f'annotation_{res}.nrrd')  # @TODO have a look at metaANO and its data

    print(f'{ANO.shape=}')
    return ANO


def visualize_embedding(data, structure_list, structure_set=None, save_name='', res=50, slice_rescaling=False):
    """
    Maps the generated embeddings per section to brain and saves the respective
    :param data:                len(structure_list) x 3
    :param structure_list:      list of structures, w.r.t. giving order to above data, as numbers
    :param structure_set:       selected subset of structures to be displayed
    :param res:                 resolution of the annotation volume and the resulting image in $\my m$
    :param save_name:           custom name for file name
    :param slice_rescaling:     rescale each slice to full color range

    :return:                    None
    """

    data = np.array(data)
    if not data.shape == (len(structure_list), 3): raise ValueError("Invalid data shape!")

    anotation_matrix = get_voxel_to_section_map(res=res)
    structure_list = [int(struc) for struc in structure_list]

    if structure_set:
        structure_set = [int(struc) for struc in structure_set]
    else:
        structure_set = structure_list

    coloured_volume = np.zeros((*anotation_matrix.shape, 3))
    for struc in tqdm(set(structure_list) & set(structure_set) & set(np.unique(anotation_matrix))):
        coloured_volume[anotation_matrix==struc, :] = data[structure_list.index(struc)]

    # Save images to results
    scaling_factor = 2
    num_slices = 2
    for i in range(1, num_slices):
        im_mat = coloured_volume[i*(anotation_matrix.shape[0]//num_slices),:,:,:].copy()
        if slice_rescaling: im_mat = rescale_image(im_mat)

        print(im_mat.min(), im_mat.max())
        im = Image.fromarray(np.uint8(im_mat*255))
        im = im.resize(tuple([scaling_factor*x for x in im.size]), Image.ANTIALIAS)
        coro_save_path = f'../results/{save_name}ano_coronal_{res}_res_slice_{i}.png'
        im.save(coro_save_path)

    for i in range(1, num_slices):
        im_mat = coloured_volume[:,:,i*(anotation_matrix.shape[0]//num_slices),:].copy()
        if slice_rescaling: im_mat = rescale_image(im_mat)

        im = Image.fromarray(np.uint8(im_mat*255))
        im = im.resize(tuple([scaling_factor*x for x in im.size]), Image.ANTIALIAS)
        sagi_save_path = f'../results/{save_name}ano_sagittal_{res}_res_slice_{i}.png'
        im.save(sagi_save_path)

    print(f'Images saved to:\n\t{coro_save_path}\n\t{sagi_save_path}')


def rescale_image(im_mat):
    zero_mask = im_mat == 0
    im_mat -= np.partition(np.unique(im_mat), 1)[1]  # select second smallest element i.e. \neq 0
    im_mat /= im_mat.max()
    im_mat[zero_mask] = 0
    return im_mat


def draw_onto_graph():
    import pydot
    from networkx.drawing.nx_pydot import graphviz_layout

    orig_onto_graph, _ = GE_data_preparation.get_structure_ontology()
    onto_graph = nx.DiGraph()
    onto_graph.add_edges_from([(u, v) for u, v in orig_onto_graph.edges()
                               if orig_onto_graph[u][v]['label'] == 'has_child'])

    pos = graphviz_layout(onto_graph, prog="dot")

    nx.draw(onto_graph, pos)
    plt.savefig("../results/structure_ontology.png")  # save as png


if __name__ == '__main__':
    draw_onto_graph()