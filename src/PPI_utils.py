import networkx as nx

from tqdm import tqdm
import pickle

from Bio import SeqIO


def prune_protein_protein_db(min_score=700,
                             mode=''):

    filename = "../data/STRING_data/10090.protein.links.full.v11.0.txt"
    target_filename = "../data/STRING_data/10090.protein.links." + str(min_score) + "_min_score.v11.0.txt"

    print("Processing raw human protein links file ...")
    p = 0.041 # see STRING documentation
    with open(file=filename, mode='r') as f, open(file=target_filename, mode='w') as targetfile:
        head = f.readline()
        targetfile.write(head)

        counter = 0

        for line in f:
            counter += 1
            if counter % 1000000 == 0:
                print("Processed lines:", counter)

            split_line = line.strip().split(' ')

            if mode=='experimental':
                experimental_score = (1-int(split_line[-6])/1000) * (1-int(split_line[-7])/1000)
                database_score = (1-int(split_line[-5])/1000) * (1-int(split_line[-4])/1000)
                experimental_score = int(1000 * (1-experimental_score * database_score))
                if experimental_score < min_score:
                    continue
                targetfile.write(split_line[0]+" "+ split_line[1]+" "+str(experimental_score)+'\n')
            else:
                total_score = int(split_line[15])/1000
                total_score_nop = (total_score-p)/(1-p)
                txt_score = int(split_line[14])/1000
                txt_score_nop = (txt_score - p)/(1-p)
                total_score_updated_nop = 1 - (1-total_score_nop)/(1-txt_score_nop)
                total_score_updated = total_score_updated_nop + p * (1-total_score_updated_nop)
                if total_score_updated * 1000 < min_score:
                    continue
                targetfile.write(split_line[0]+" "+ split_line[1]+" "+str(int(total_score_updated*1000))+'\n')
    print("Finished.")

def get_protein_list(min_score=700):
    return sorted(get_PPI_graph(min_score=min_score).nodes())

def write_PPI_graph(min_score=700):
    pruned_PPI_file = "../data/STRING_data/10090.protein.links." + str(min_score) + "_min_score.v11.0.txt"

    print("Building PPI graph ...")
    PPI_graph = nx.Graph()
    num_lines = sum(1 for line in open(pruned_PPI_file, 'r'))
    with open(file=pruned_PPI_file, mode='r') as f:
        f.readline() # skip header
        for line in tqdm(f, total=num_lines):
            split_line = line.split(' ')

            node_1 = split_line[0]
            node_2 = split_line[1]
            score = int(split_line[-1])

            PPI_graph.add_node(node_1)
            PPI_graph.add_node(node_2)
            PPI_graph.add_edge(node_1, node_2, score=score)
    print("Finished.")

    print('nodes', len(PPI_graph.nodes()))
    print('edges', len(PPI_graph.edges()))

    print("Writing PPI graph to disk ...")
    graph_filename = "../data/PPI_data/PPI_graph_"+str(min_score)+"_min_score"
    with open(file=graph_filename+'.pkl', mode='wb') as f:
        pickle.dump(PPI_graph, f, pickle.HIGHEST_PROTOCOL)
    print("Finished writing {}.\n".format(graph_filename))

def get_PPI_graph(min_score=700):
    filename = "../data/STRING_data/PPI_graph_" + str(min_score) + "_min_score"
    with open(file= filename+'.pkl', mode='rb') as f:
        return pickle.load(f)

def write_protein_fasta(protein_list):

    input_fasta_file = '../data/STITCH_data/10090.protein.sequences.v11.0.fa'

    return_sequences = []  # Setup an empty list
    for record in SeqIO.parse(input_fasta_file, "fasta"):
        if record.id in protein_list:
            return_sequences.append(record)

    print("Found {} PPI protein sequences of {}".format(len(return_sequences), len(protein_list)))

    SeqIO.write(return_sequences, "src/data/PPI_graph_protein_seqs.fasta", "fasta")

def write_encoded_proteins():

    protein_dir = "../models/protein_representation/"
    in_file = protein_dir+'data/PPI_graph_protein_seqs.fasta'
    out_file = protein_dir+'results/output'

    print('Execute this command in',protein_dir)
    print(f'source {protein_dir}predict.sh {in_file} {out_file}')


if __name__ == '__main__':
    prune_protein_protein_db(min_score=700)

    write_PPI_graph(min_score=700)