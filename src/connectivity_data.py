import numpy as np
import pandas as pd

import networkx as nx


from tqdm import tqdm
import pickle
import os

import GE_data_preparation


def write_structure_subgroups():
    from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

    path = '../data/'
    # adapted from https://allensdk.readthedocs.io/en/latest/_static/examples/nb/mouse_connectivity.html#Structure-Signal-Unionization
    mcc = MouseConnectivityCache()
    structure_tree = mcc.get_structure_tree()

    from allensdk.api.queries.ontologies_api import OntologiesApi
    oapi = OntologiesApi()

    structure_set_ids = structure_tree.get_structure_sets()

    struc_df = pd.DataFrame(oapi.get_structure_sets(structure_set_ids))

    print('Writing structure sets...')
    filename = 'structure_sets.csv'
    with open(file=path+filename, mode='w') as f:
        print('description\tstructure_set_id\tstructures_in_set', file=f)
        for idx, row in struc_df.iterrows():
            description, set_id = row['description'], row['id']

            summary_structures = structure_tree.get_structures_by_set_id([set_id])

            summary_structures = pd.DataFrame(summary_structures)
            ids_in_set = [str(id) for id in summary_structures['id']]
            print(description+'\t'+str(set_id)+'\t'+(','.join(ids_in_set)), file=f)
    print('Done.')


def get_structure_subgroups():
    file = '../data/structure_sets.csv'
    return_list = []
    with open(file=file,mode='r') as f:
        # skip header
        f.readline()
        for line in f:
            description, set_id, structure_list = line.strip().split('\t')
            structure_list = structure_list.split(',')
            return_list.append((description, set_id, structure_list))

    return return_list


def download_connectivity_data():
    from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    path = '../data/connectivity_data/structural_connectivity/'

    # if not os.path.exists(path+'all_experiments.csv') \
    # or not os.path.exists(path+'structure_list'):
    mcc = MouseConnectivityCache()
    all_experiments = mcc.get_experiments(dataframe=True)

    print("%d total experiments" % len(all_experiments))

    structure_tree = mcc.get_structure_tree()

    # save all experiments to disc
    # all_experiments.to_csv(path+'all_experiments.csv')

    # Uncomment for using structures with single injected structure
    # structure_list = list(set(sum([struc_list for struc_list in all_experiments['injection_structures'] if len(struc_list) == 1], [])))
    # Use this for just taking primary injection structures
    structure_list = list(set([s for s in all_experiments['primary_injection_structure']]))
    print(f'{len(structure_list)} structures present.')

    '''
    else:
        # parse existent structure list all_experiments data frame
        structure_list = []
        with open(file=path+'structure_list', mode='r') as f:
            for line in f:
                structure_list.append(int(line.strip()))

        all_experiments = pd.read_csv(path+'all_experiments.csv')
    '''

    all_experiment_ids = [id for id in all_experiments['id']]

    # structure_unionizes = mcc.get_structure_unionizes(all_experiment_ids)

    for hemisphere_id in [3]:
        # 1: left hemisphere
        # 2: right hemisphere
        # 3: overall
        print('Processing hemisphere_id:', hemisphere_id)
        hemisphere_mat_file = path + 'projection_matrix_data_hemisphere_' + str(hemisphere_id) + '.pkl'
        if not os.path.exists(hemisphere_mat_file):
            mcc = MouseConnectivityCache()
            print('Downloading projection matrix... (This might take some time.)')
            projection_matrix_data = mcc.get_projection_matrix(experiment_ids=all_experiment_ids,
                                                               projection_structure_ids=structure_list,
                                                               hemisphere_ids=[hemisphere_id],
                                                               parameter='projection_density')

            with open(file=hemisphere_mat_file, mode='wb') as f:
                pickle.dump(projection_matrix_data, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(file=hemisphere_mat_file, mode='rb') as f:
                projection_matrix_data = pickle.load(f)

        # distributing projection data to variables
        row_labels = projection_matrix_data['rows']
        column_labels = projection_matrix_data['columns']
        projection_matrix = projection_matrix_data['matrix']

        # prune structure list to entities present in data retrieved from projection matrix
        structure_list = [struc for struc in structure_list if struc in [col['structure_id'] for col in column_labels]]
        with open(file=path + 'structure_list', mode='w') as f:
            f.write('\n'.join(map(str, structure_list)))
        print(f'{len(structure_list)} structures also present in projection data.')

        struc_conn_mat = np.zeros((len(structure_list), len(structure_list)))

        num_ignored_experiments = 0
        for idx, row in tqdm(all_experiments.iterrows(), total=len(all_experiments.index)):
            injection_struc, experiment_id = row['primary_injection_structure'], row['id']
            # transform string like '[1,2,3]' to list [1,2,3]
            # injection_struc = list(map(int, injection_struc[1:-1].replace(' ', '').split(',')))
            # if not len(injection_struc) == 1:
            # continue

            experiment_row = projection_matrix[row_labels.index(experiment_id), :]

            # for struc in injection_struc:
            if injection_struc not in structure_list:
                continue

            struc_index = structure_list.index(injection_struc)
            for id_col, col in enumerate(column_labels):
                # construct mapping from structure_list indices to corresponding column label index
                col_struc = col['structure_id']
                if col_struc not in structure_list:
                    continue
                col_struc_index = structure_list.index(col_struc)

                # struc_conn_mat[struc_index, structure_list.index(col_struc)] = experiment_row[id_col]
                experiment_row = np.nan_to_num(experiment_row)
                if struc_conn_mat[struc_index, col_struc_index] < experiment_row[id_col]:
                    struc_conn_mat[struc_index, col_struc_index] = experiment_row[id_col]
                    struc_conn_mat[col_struc_index, struc_index] = experiment_row[id_col]

        struc_conn_mat = np.nan_to_num(struc_conn_mat)

        '''
        # test anomalies in struc_conn_mat
        for i in range(len(structure_list)):
            if struc_conn_mat[i,i] < struc_conn_mat[i,:].max():
                max_ind = struc_conn_mat[i].argmax()
                print(structure_list[i], structure_list[max_ind])
        '''

        np.save(path+'struct_conn_hemisphere_'+str(hemisphere_id)+'.npy', struc_conn_mat)
        print(f'Number of ignored experiments: {num_ignored_experiments}')
    print('Done.')


def get_struc_conn_matrix(hemisphere_id=3):
    path = '../data/connectivity_data/structural_connectivity/'
    return_mat = np.load(path+'struct_conn_hemisphere_'+str(hemisphere_id)+'.npy')
    diagonal_vec = np.diag(return_mat).copy()
    # print(diagonal_vec.max(), diagonal_vec.min(), diagonal_vec.mean(), diagonal_vec.shape)

    return return_mat  # / diagonal_vec


def get_struc_struc_list():
    path = '../data/connectivity_data/structural_connectivity/'
    structure_list = []
    with open(file=path + 'structure_list', mode='r') as f:
        for line in f:
            structure_list.append(line.strip())
    return structure_list


def get_struc_conn_graph(threshold=0.5):
    struc_conn_mat = get_struc_conn_matrix()
    struc_conn_struc_list = get_struc_conn_list()

    G = nx.from_numpy_array(struc_conn_mat > threshold)
    for i, node in enumerate(G.nodes):
        G.nodes[node]['struc_name'] = struc_conn_struc_list[i]
        print(G.nodes[node], i)

    return G


def write_functional_conn_data(aggr='mean'):
    path = '../data/connectivity_data/functional_connectivity/'

    dims = np.zeros(3)
    conn_mat = None
    struc_list = []

    with open(file=path+'data/outpath/' + 'Baseline.mat') as f:
        print('Parsing functional connectivity matrix...')
        index = 0
        parse_mat = True
        for line in f:
            if parse_mat:
                if line.strip() == '# name: matrix' and f.readline().strip() == '# type: matrix':
                    next(f) # skip ndims
                    dims = np.array(list(map(int, f.readline().strip().split(' '))))
                    conn_mat = np.zeros(np.prod(dims))
                    print('conn_mat.shape', conn_mat.shape)
                    continue

                if not dims.any(): continue
                line = line.strip()
                if not line:
                    parse_mat = False
                    continue

                conn_mat[index] = float(line)
                index += 1

            else:
                if line.strip() == '# name: <cell-element>' and f.readline().strip() == '# type: sq_string':
                    for i in range(2): next(f)
                    struc_list.append(f.readline().strip())

        conn_mat = conn_mat.reshape(dims)
    print('conn_mat.shape', conn_mat.shape)

    if aggr == 'max':
        conn_mat = conn_mat.max(axis=2)
    if aggr == 'mean':
        conn_mat = conn_mat.mean(axis=2)
    else:
        raise ValueError("No valid aggregation method chosen in functional connectivity parser!")

    print('Writing processed files to disk...')
    with open(file=path + 'funn_structs.txt', mode='w') as f:
        for struc in struc_list:
            print(struc, file=f)

    with open(file=path + 'funn_conn_mat.npy', mode='wb') as f:
        np.save(f, conn_mat)

    print('Done.')


def get_orig_funn_struc_list():
    path = '../data/connectivity_data/functional_connectivity/'
    struc_list = []
    with open(file=path + 'funn_structs.txt', mode='r') as f:
        for line in f:
            struc = line.strip()
            struc_list.append(struc)

    return struc_list


def get_funn_struc_list():
    path = '../data/connectivity_data/functional_connectivity/'
    return_struc_list = []
    orig_struc_list = get_orig_funn_struc_list()

    meta_data = GE_data_preparation.parse_structure_metadata()

    for struc in orig_struc_list:
        struc = struc[2:]
        for entry in meta_data:
            if struc == entry['name']:
                if entry['id'] not in return_struc_list:
                    return_struc_list.append(entry['id'])
                break

    return return_struc_list


def get_funn_conn_mat(threshold=0.5):
    path = '../data/connectivity_data/functional_connectivity/'

    with open(file=path + 'funn_conn_mat.npy', mode='rb') as f:
        return reduce_to_single_hemisphere(np.load(f))>threshold


def reduce_to_single_hemisphere(mat, mode='mean'):
    orig_struc_list = get_orig_funn_struc_list()
    struc_list = []

    for struc in orig_struc_list:
        struc = struc[2:]
        if struc not in struc_list: struc_list.append(struc)

    l_indices = []
    r_indices = []
    for j, struc in enumerate(struc_list):
        for i, orig_struc in enumerate(orig_struc_list):
            if orig_struc == 'L ' + struc:
                l_indices.append(i)
            elif orig_struc == 'R ' + struc:
                r_indices.append(i)
    l_indices = np.array(l_indices)
    r_indices = np.array(r_indices)

    mat = mat.reshape((1, len(orig_struc_list), len(orig_struc_list)))
    return_mat = np.concatenate((mat[:, l_indices, :][:, :, l_indices],
                                 # mat[:, l_indices, :][:, :, r_indices],
                                 # mat[:, r_indices, :][:, :, l_indices],
                                 mat[:, r_indices, :][:, :, r_indices]
                                 ), axis=0)

    if mode=='max':
        return_mat = return_mat.max(axis=0)
    elif mode=='mean':
        return_mat = return_mat.mean(axis=0)
    else:
        raise ValueError

    return return_mat


def get_funn_conn_graph(threshold=0.5):
    fun_conn_mat = get_funn_conn_mat()
    struc_list = get_funn_struc_list()

    G = nx.from_numpy_array(fun_conn_mat>threshold)
    for i, node in enumerate(G.nodes):
        G.nodes[node]['struc_name'] = struc_list[i]

    return G


def calculate_eff_conn(threshold=0.1):
    path = '../data/connectivity_data/functional_connectivity/data/Baseline/Control/'
    data_list = []
    for individual in os.listdir(path):
        filename = path+individual+'/fMRI/regr/MasksTCsSplit.GV.txt'
        data = np.transpose(np.genfromtxt(filename))
        betas = pseudo_iv_betas(data)
        betas = np.abs(betas)
        # betas[betas!=0] = np.log(betas[betas!=0])
        betas /= betas.max()
        betas = betas>threshold
        print(f'{betas.shape=} {betas.max()=} {betas.sum()=} {betas.min()=}')
        data_list.append(betas)
        data_list.append(np.transpose(betas))  # symmetrize adjacency matrix

    betas = np.array(data_list).max(axis=0)
    betas = reduce_to_single_hemisphere(betas, mode='max')
    return betas


def iv_betas(activations, instruments):
    """
    See https://github.com/KordingLab/fmri-iv for more information.

    Estimate causal connection strengths using instrumental variable method.
    :param activations: NxM time series of activations for N neurons/regions
    :param instruments: NxM time series N instruments each of which corresponds to 1 region
    only directly affects its corresponding neuron (i.e. satisfies the IV criteria for
    that node of the causal graph).
    Typically, the instrument might be binary and represent whether its neuron is locally
    inhibited from firing, but any linear relationship between instrument and neuron should work.
    Instruments should satisfy the IV criteria for their corresponding neurons/regions, i.e.
    have no direct causal influence on other neurons.
    :return: a matrix of "beta" coefficients representing the extent of directed causal influence
    between each neuron and every other neuron at a later time step, using the IV method.
    beta[i, j] = estimated effect of region j on region i.
    """

    assert instruments.shape == activations.shape, "Activations and instruments must be the same shape"

    n_regions, n_times = activations.shape

    assert n_times >= 2, "Must have at least 2 timepoints for IV analysis"

    # define z (instrument), x (timepoint 1), and y (timepoint 2)
    z = instruments[:, :-1]
    x = activations[:, :-1]
    y = activations[:, 1:]

    # do 2SLS for each region
    beta = np.zeros((n_regions, n_regions))

    for kR1 in range(n_regions):
        w, b = np.linalg.lstsq(np.vstack([z[kR1], np.ones(n_times - 1)]).T, x[kR1], rcond=None)[0]
        x_pred = z[kR1] * w + b

        for kR2 in range(n_regions):
            beta[kR2, kR1], _ = np.linalg.lstsq(np.vstack([x_pred, np.ones(n_times - 1)]).T, y[kR2], rcond=None)[0]

    return beta



def get_eff_struc_list():
    # effective connectivity is calculated on same AIDAmri data
    return get_funn_struc_list()


def pseudo_iv_betas(activations, sd_threshold=2, log_transform_input=False):
    """
    See https://github.com/KordingLab/fmri-iv for more information.

    Try to estimate causal connection strengths, using past activity of each neuron as its own
    "instrumental variable." This is unlikely to actually satisfy the IV criteria since the
    instrument will have the same outgoing influences as the variable itself, meaning it
    does probably affect other neurons in the network. Similarly, it is going to be influenced
    by other neurons in the network.
    :param activations: NxM time series of activations for N neurons/regions
    :param sd_threshold: instrument is "on" when past activity of each neuron is below
    its mean activation minus this number times the standard deviation of its activation.
    :param log_transform_input: if true, take the natural log of the activations before calculating the pseudo-IV.
    :return: a matrix of "beta" coefficients representing the extent of directed causal influence
    between each neuron and every other neuron at a later time step, using the IV method.
    beta[i, j] = estimated effect of region j on region i.
    """
    n_times, n_regions = activations.shape

    assert n_times >= 3, "Must have at least 3 timepoints for pseudo-IV analysis"

    # define instrument, x, and y
    act_t0 = activations[:, :-1]
    if log_transform_input:
        act_t0 = np.log(act_t0)

    threshold = np.mean(activations, axis=1, keepdims=True) - sd_threshold * np.std(activations, axis=1, keepdims=True)
    z = np.array(act_t0 < threshold, dtype=np.float64)

    return iv_betas(activations[:, 1:], z)


def norm_lagged_corr(activations):
    """
    See https://github.com/KordingLab/fmri-iv for more information.

    Estimate connectivity matrix A from autocorrelation and covariance matrices
    :param activations: NxM time series of activations for N neurons/regions
    :return: estimation of A
    """

    _, n_times = activations.shape

    cov_mat = np.cov(activations)
    autocorr_mat = activations[:, 1:] @ activations[:, :-1].T / (n_times - 1)

    return autocorr_mat @ np.linalg.pinv(cov_mat)


def prune_conn_mat(conn_mat, structure_sub_list, orig_structure_list):

    struc_indices = [orig_structure_list.index(struct) for struct in structure_sub_list]
    return conn_mat[struc_indices,:][:, struc_indices]


def get_both_struc_list():
    raise NotImplementedError




if __name__ == '__main__':
    # write_structure_subgroups()
    # download_connectivity_data()

    # write_functional_conn_data()
    # get_funn_conn_graph()
    # get_struc_conn_graph()

    calculate_eff_conn()


