import os
import networkx as nx
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import matplotlib.patches as patches
from readfiles import corresp_label_file


class BrainGraph(nx.Graph):  # inheriting from networkx graph package along with the functionality we want
    def __init__(self, edge, feature_type, node_wts, target, max_num_nodes, val, thresh):
        # self.connected_subgraph = self.subgraph([0])
        super(BrainGraph, self).__init__()  # constructor of the base class
        self.edge = edge
        self.feature_type = feature_type
        self.node_wts = node_wts
        self.target = target
        self.filename = f'{target}_{edge}_{max_num_nodes}_{feature_type}_{thresh}'
        self.connected_nodes = []
        self.connected_degree = 0
        self.self_loops = []
        self.edge_weights = []
        self.val = val

    def subgraph(self):
        # create new graph and copy subgraph into it
        # pipeline changed
        H = self.__class__(self.edge, self.feature_type, self.node_wts, self.target, self.val)
        # copy node and attribute dictionaries
        H.add_nodes_from(self.connected_nodes)
        subgr_edges = []
        for x in self.connected_nodes:
            # if edge[0] in g1.nodes and edge[1] in g1.nodes:
            H.nodes[x]['label'] = self.nodes[x]['label']
            for conn in self[x]:
                subgr_edges.append((x, conn, self[x][conn]['weight']))
        H.add_weighted_edges_from(subgr_edges)
        return H

    def make_graph(self, arr, strls_num, thresh, avg_thresh):
        mat = np.triu_indices(84)
        assert len(mat[0]) == len(strls_num)
        nodes = set()
        edge_attributes = []
        strl = np.zeros((84, 84))
        for i in range(len(mat[0])):
            strl[mat[0][i], mat[1][i]] = strls_num.iloc[i]
        strl = strl + strl.T - np.diag(strl.diagonal())
        strl = np.sum(strl, axis=0)
        for j in range(len(mat[0])):
            value = float(arr.iloc[j])
            u = mat[0][j]
            v = mat[1][j]
            nodes.add(u)  # add only the nodes which have corresponding edges
            nodes.add(v)
            if avg_thresh:
                if value > 0 and u != v and max(strls_num.iloc[j] / strl[u] ,strls_num.iloc[j] / strl[v]) >= thresh:
                    edge_attributes.append((mat[0][j], mat[1][j], value))
                else:
                    self.self_loops.append(value)
            else:
                if value > 0 and u != v and strls_num.iloc[j] == True:
                    edge_attributes.append((mat[0][j], mat[1][j], value))
                else:
                    self.self_loops.append(value)

            # mean for the scores of three different labels
        assert nodes is not None
        self.add_nodes_from(nodes)
        self.add_weighted_edges_from(edge_attributes)
        self.edge_weights = []
        for u, v in self.edges:
            self.edge_weights.append(self[u][v]['weight'])

    def set_node_labels(self, node_wts, const_val=None):
        node_labels = []
        for l in self.nodes.keys():
            if node_wts == 'max':
                self.nodes[l]['label'] = max(
                    [dict(self[l])[k]['weight'] for k in dict(self[l]).keys()])  # max or max abs?
            elif node_wts == 'const':
                self.nodes[l]['label'] = const_val
            elif node_wts == 'avg':
                self.nodes[l]['label'] = np.mean([dict(self[l])[k]['weight'] for k in dict(self[l]).keys()])
            node_labels.append(self.nodes[l]['label'])
        self.node_labels = node_labels

    def normalize_node_attr(self):
        self.node_labels = scale(self.node_labels)  # standardizing the node labels
        for n in self.nodes.keys():
            self.nodes[n]['label'] = self.node_labels[n]

    def visualize_graph(self, mews, input_gr, plotting_options):
        # not using the subgraph extraction right now since we are not thresholding in any case

        if self.edge_weights != None and self.edge_weights != []:
            minima = min(self.edge_weights)
            maxima = max(self.edge_weights)
            norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima)
            mapper = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('Spectral'))
            color = []
            for v in self.edge_weights:
                color.append(mapper.to_rgba(v))
            # build a rectangle in axes coords
            fig, ax = plt.subplots()
            #print('minima and maxima', minima, maxima)
            if input_gr == True:
                plt.title(f'input to the solver: {self.filename}\n '
                          f'Number of edges {len(self.edges)}\n'
                          f' Target: {self.target}, Feature:{self.feature_type}\n'
                          f'Edge type:{self.edge}, Node weighting:{self.node_wts}')
                if self.val != None:
                    #print('constant value has been given')
                    ax.annotate(f'Node weight={self.val}', xy=(0, 1))
                nx.draw(self, **plotting_options, edge_color=color, edge_cmap=mapper.cmap, vmin=minima,
                        vmax=maxima, with_labels=False)
                plt.colorbar(mapper)
                plt.savefig(f'{mews}/outputs/figs/{self.filename}.png')

            else:
                plt.title(f'Output from the solver: {self.filename}\n Number of edges {len(self.edges)}\n'
                          f' Target: {self.target}, Feature:{self.feature_type}\n'
                          f'Edge type:{self.edge}, Node weighting: according to solver')
                nx.draw(self, **plotting_options, edge_color=color, edge_cmap=mapper.cmap, vmin=minima,
                        vmax=maxima, with_labels=False)
                plt.colorbar(mapper)
                plt.savefig(f'{mews}/outputs/figs/{self.filename}_out.png')
            plt.show()
        else:
            print('No output produced by the solver')

    def savefiles(self, mews):

        with open(f'{mews}/outputs/nodes/{self.filename}', 'w') as nodes_file:
            for x in self.nodes: #we are starting it from 1
                print(str(x+1) + ' ' * 3 + str(self.nodes[x]['label']), file=nodes_file)


        with open(f'{mews}/outputs/edges/{self.filename}', 'w') as edges_file:

            for u, v in self.edges:
                if u != v:  # just don't write these into the files and also make sure that this doesn't happen
                    print(str(u+1) + ' ' * 3 + str(v+1) + ' ' * 3 + str(self[u][v]['weight']),
                          file=edges_file)  # original file format was supposed to have 3 spaces

    def run_solver(self, mews, max_num_nodes):
        os.chdir(mews)
        # print('Current directory', os.getcwd())
        cmd = (
            f' java -Xss4M -Djava.library.path=/opt/ibm/ILOG/CPLEX_Studio1210/cplex/bin/x86-64_linux/ '
            f'-cp /opt/ibm/ILOG/CPLEX_Studio1210/cplex/lib/cplex.jar:target/gmwcs-solver.jar '
            f'ru.ifmo.ctddev.gmwcs.Main -e outputs/edges/{self.filename} '
            f'-n outputs/nodes/{self.filename} > output'
            f's/solver/{self.filename} -nm {max_num_nodes}')  # training data into the solver

        # print(cmd)
        os.system(cmd)
        os.chdir("/home/skapoor/Thesis/graphtools")
        # print('Current directory', os.getcwd())

    def read_from_file(self, mews, input_graph, mat=np.triu_indices(84)):
        if input_graph==False:
            name = '.out'
        else:
            name = ''
        if os.path.exists(f'{mews}/outputs/nodes/{self.filename}{name}') \
                and os.path.exists(f'{mews}/outputs/edges/{self.filename}{name}'):
            with open(f'{mews}/outputs/nodes/{self.filename}{name}', 'r') as nodes_file, \
                    open(f'{mews}/outputs/edges/{self.filename}{name}', 'r') as edges_file:
                if input_graph:
                    nodes = [x.split('   ') for x in nodes_file.read().split('\n')]
                    edges = [x.split('   ') for x in edges_file.read().split('\n')]
                else:
                    nodes = [x.split('\t') for x in nodes_file.read().split('\n')]
                    edges = [x.split('\t') for x in edges_file.read().split('\n')]
                nodes_e = set()
                edges_e = set()
                node_labels = {}
                for a in nodes[:-1]:
                    if a[1] != 'n/a':  # since the last line is the subnet score
                        nodes_e.add(int(a[0])-1)
                        node_labels[int(a[0])-1] = float(a[1])
                feature_indices = []

                # 0 to range(len(mat)), everything in matrix whole corresponding to this edge is feature
                self.edge_weights = []
                for existing_edge in edges[:-1]:
                    if existing_edge[-1] != 'n/a':
                        # because we wrote the node and edge names starting from 1 instead of 0
                        edges_e.add((int(existing_edge[0])-1, int(existing_edge[1])-1, float(existing_edge[2])))
                        for k in range(len(mat[0])):
                            if (int(existing_edge[0])-1, int(existing_edge[1])-1) == (mat[0][k], mat[1][k]):
                                feature_indices.append(k) #if we were writing the node names from 1 onwards how to
                                self.edge_weights.append((float(existing_edge[2])))
                                # feature_mat = whole.iloc[:, :tri]]
                dict_lut = corresp_label_file('fs_default.txt')
                self.add_nodes_from(nodes_e)
                self.connected_nodes = list(nodes_e)
                self.add_weighted_edges_from(edges_e)
                self.node_labels = []
                for l in self.nodes.keys():
                    self.nodes[l]['label'] = dict_lut[l+1]
                    self.node_labels.append(node_labels[l])

            self.edge_weights = []
            for u, v in self.edges:
                self.edge_weights.append(self[u][v]['weight'])

                # print('output graph has been read from file', 'nodes:', self.nodes, 'edges', self.edges)
            return feature_indices
        else:
            self.edge_weights = []
            self.node_labels = []
            return None

    def hist(self, mews):
        """
        Plotting the histogram of the sum of incoming edge weights
        @return:
        """
        incoming_sums = []
        for node in self.nodes:
            incoming_sums.append(sum([self[node][v]['weight'] for v in self[node].keys()]))
        # fig = plt.figure()
        plt.title('Sum of incoming edges on each node')
        plt.ylabel('Number of nodes')
        plt.xlabel('Sums')
        plt.hist(incoming_sums)
        plt.savefig(f'{mews}/outputs/figs/{self.filename}.png')
        plt.show()
