import numpy as np
import math
import nrrd
import networkx as nx

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors

from tqdm import tqdm

import GE_data_preparation
import connectivity_data


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


def draw_ge_histo(n_bins=100, normalization='column', mode='percent'):

    from matplotlib.ticker import PercentFormatter

    mat = GE_data_preparation.get_ge_structure_data()

    print(f'{mat.shape=}, {mat.max()=}, {mat.min()=}')

    if normalization == 'global':
        mat /= mat.max()
    elif normalization == 'row':
        mat = mat / mat.max(axis=1).reshape(-1, 1)
    elif normalization == 'column':
        mat = mat / mat.max(axis=0)

    print(f'{mat.shape=}, {mat.max()=}, {mat.min()=}')

    mat = mat.reshape(-1)
    print(f'{(mat==0).sum()}')

    fig, axs = plt.subplots(1, 1, tight_layout=True)

    if mode=='normal':
        # N is the count in each bin, bins is the lower-limit of the bin
        N, bins, patches = axs.hist(mat, bins=n_bins)

        # we need to normalize the data to 0..1 for the full range of the colormap
        norm = colors.Normalize(bins.min(), bins.max())

        # Now, we'll loop through our objects and set the color of each accordingly
        for thisbin, thispatch in zip(bins, patches):
            color = plt.cm.viridis(norm(thisbin))
            thispatch.set_facecolor(color)
        axs.set(xlabel='Normalized expression energy', ylabel='Bin counts')

    elif mode=='percent':
        # We can also normalize our inputs by the total number of counts
        N, bins, patches = axs.hist(mat, bins=n_bins, weights=np.ones(len(mat)) / len(mat))
        fracs = N/N.max()

        norm = colors.Normalize(bins.min(), bins.max())

        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)

        # Now we format the y-axis to display percentage
        axs.yaxis.set_major_formatter(PercentFormatter(1))
        axs.set(xlabel='Normalized expression energy', ylabel='Bin shares')
    else: raise ValueError

    plt.savefig(f'../thesis/plotted_figures/{normalization}_{mode}_ge_data_histo.png')


def visualize_conn_graph_3d(conn_graph, structure_list, prefix='', threshold=0.1):

    conn_graph = conn_graph>threshold
    print(f'{conn_graph.shape=} {conn_graph.sum()=}')
    map_dict = {}

    max_value = -math.inf
    min_value = math.inf
    with open(file='../data/CCFv3/ccf_2017/structure_centers.csv', mode='r') as f:
        f.readline()  # skip header
        for line in f:
            struc_id, x,y,z, ref_space_id = line.strip().split(',')
            x = float(x)
            y = float(y)
            z = float(z)
            map_dict[struc_id] = (x,y,z)
            val = -x+y-z

            if val>max_value: max_value = val
            if val<min_value: min_value = val

    values = np.array(list(map_dict.values()))
    x_max, x_min = values[:, 0].max(), values[:, 0].min()
    y_max, y_min = values[:, 1].max(), values[:, 1].min()
    z_max, z_min = values[:, 2].max(), values[:, 2].min()
    print(x_max, x_min, y_max, y_min, z_max, z_min)


    plt.rcParams["figure.figsize"] = [20, 20]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()

    ax = fig.add_subplot(projection="3d")

    for s, coords in map_dict.items():
        x,y,z = np.array(coords, dtype=np.float32)
        x = (x-x_min)/(x_max-x_min)
        y = (y-y_min)/(y_max-y_min)
        z = (z-z_min)/(z_max-z_min)
        ax.scatter(x, y, z, color=(x,y,z), s=100)#c=(-x+y-z-min_value)/(max_value-min_value)*100)

    missed_counter = 0
    for i, s1 in enumerate(structure_list):
        for j, s2 in enumerate(structure_list):
            if conn_graph[i,j] <= threshold: continue
            if s1 not in map_dict or s2 not in map_dict:
                missed_counter += 1
                continue

            x,y,z = np.array(list(zip(map_dict[s1], map_dict[s2])), dtype=np.float32)
            x = (x - x_min) / (x_max - x_min)
            y = (y - y_min) / (y_max - y_min)
            z = (z - z_min) / (z_max - z_min)
            ax.plot(x, y, z, color=(x.mean(), y.mean(), z.mean()), alpha=0.5)
    print(f'{missed_counter=}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig(f'../thesis/plotted_figures/{prefix}_3d_conn_plot.png')


def plot_baseline_signals(type='random'):

    if type=='random':
        # Random density function
        x = [0,1]
        y = [1,1]
    elif type=='constant':
        # plot Dirac fuction at point t

        t = 0.3
        def ddf(x, sig):
            val = np.zeros_like(x)
            val[(-(1 / (2 * sig)) <= x) & (x <= (1 / (2 * sig)))] = 1
            return val
        n_points = 1000
        x = np.linspace(0,1, n_points)
        y = np.zeros_like(x)
        t_index = int(t * n_points)
        y[t_index-1 : t_index+2] = 1
    elif type=='small_noise':
        t = 0.3
        mu, sigma = t, 0.05
        n_points = 1000
        x = np.linspace(0,1, n_points)
        y = 1/(2*math.pi*sigma**2)**(1/2) * math.e ** (-(x-mu)**2/(2*sigma**2))

    else: raise ValueError

    plt.figure(figsize=(6, 4))
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('p(x)')

    plt.title('Probability density function p(x)')
    plt.savefig(f'../thesis/plotted_figures/{type}_density_function.png')


if __name__ == '__main__':
    # draw_onto_graph()
    # draw_ge_histo(n_bins=40, normalization='row', mode='normal')

    # mat = connectivity_data.calculate_eff_conn(threshold=0.1)
    # struc_list = connectivity_data.get_eff_struc_list()
    # visualize_conn_graph_3d(mat, struc_list, prefix='', threshold=0.1)

    for type in ['random', 'constant', 'small_noise']:
        plot_baseline_signals(type)
