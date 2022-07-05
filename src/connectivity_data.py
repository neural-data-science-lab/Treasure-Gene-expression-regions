import numpy as np
import pandas as pd

from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from tqdm import tqdm
import pickle
import os


def write_structure_subgroups():
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


def get_structure_list():
    path = '../data/connectivity_data/structural_connectivity/'
    structure_list = []
    with open(file=path + 'structure_list', mode='r') as f:
        for line in f:
            structure_list.append(int(line.strip()))
    return structure_list


def write_functional_conn_data(aggr='max'):
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
    print('conn_mat.sum()', conn_mat.max())

    if aggr == 'max':
        conn_mat = conn_mat.max(axis=2)
    else:
        raise ValueError("No valid aggregation method chosen in functional connectivity parser!")

    print('Writing processed files to disk...')
    with open(file=path + 'funn_structs.txt', mode='w') as f:
        for struc in struc_list:
            print(struc, file=f)

    with open(file=path + 'funn_conn_mat.npy', mode='wb') as f:
        np.save(f, conn_mat)

    print('Done.')


def get_funn_struc_list():
    path = '../data/connectivity_data/functional_connectivity/'
    struc_list = []
    with open(file=path + 'funn_structs.txt', mode='r') as f:
        for line in f:
            struc_list.append(line.strip())

    return struc_list


def get_funn_conn_mat():
    path = '../data/connectivity_data/functional_connectivity/'

    with open(file=path + 'funn_conn_mat.npy', mode='rb') as f:
        return np.load(f)


if __name__ == '__main__':
    # write_structure_subgroups()
    # download_connectivity_data()

    write_functional_conn_data()


