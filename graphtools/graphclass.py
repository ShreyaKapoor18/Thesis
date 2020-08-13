import os
import networkx as nx
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
class BrainGraph(nx.Graph): #inheriting from networkx graph package along with the functionality we want
    def __init__(self, edge, feature_type, node_wts, target):
        #self.connected_subgraph = self.subgraph([0])
        super(BrainGraph, self).__init__()  # constructor of the base class
        self.edge = edge
        self.feature_type = feature_type
        self.node_wts = node_wts
        self.target = target
        self.filename = f'{target}_{edge}_{node_wts}_{feature_type}'
        self.connected_nodes = []
        self.connected_degree = 0

    def subgraph(self):
        # create new graph and copy subgraph into it
        H = self.__class__(self.edge, self.feature_type, self.node_wts, self.target)
        # copy node and attribute dictionaries
        H.add_nodes_from(self.connected_nodes)
        subgr_edges = []
        for x in self.connected_nodes:
            # if edge[0] in g1.nodes and edge[1] in g1.nodes:
            H.nodes[x]['label'] = self.nodes[x]['label']
            for conn in self[x]:
                subgr_edges.append((x,conn, self[x][conn]['weight']))
        H.add_weighted_edges_from(subgr_edges)
        return H


    def make_graph(self, arr, sub_val): #maybe this function is not even needed
        mat = np.triu_indices(84)
        print('type of array', type(arr))
        arr.fillna(0, inplace=True)
        arr = np.absolute(arr)  # need to standardize after taking the absolute value
        arr = arr - sub_val
        # we want to standardize the whole array together
        # the values are x.y and the values in itself
        # standardization of the array itself, need to preserve the non zero parts only
        arr = pd.DataFrame(arr)
        arr = MinMaxScaler.fit_transform(arr)
        # before giving these values as edge values
        # try for for different types, one feature at a time maybe and then construct graph?
        nodes = set()
        edge_attributes = []
        for j in range(len(mat[0])):
            value = float(arr.iloc[j])
            edge_attributes.append((mat[0][j], mat[1][j], value))
            nodes.add(mat[0][j])  # add only the nodes which have corresponding edges
            nodes.add(mat[1][j])
        # mean for the scores of three different labels
        assert nodes is not None
        self.add_weighted_edges_from(edge_attributes)

    def set_node_labels(self, node_wts, const_val=0):
        node_labels = []
        for l in self.nodes.keys():
            if node_wts == 'max':
                self.nodes[l]['label'] = max([dict(self[l])[k]['weight'] for k in dict(self[l]).keys()])  # max or max abs?
            elif node_wts == 'const':
                self.nodes[l]['label'] = const_val
            node_labels.append(self.nodes[l]['label'])
        self.node_labels = node_labels

    def normalize_node_attr(self):
        self.node_labels = scale(self.node_labels)  # standardizing the node labels
        for n in self.nodes.keys():
            self.nodes[n]['label'] = self.node_labels[n]

    def visualize_graph(self, mews, connected, sub_val, plotting_options):

        edge_wts = []
        if connected:
            for u in self.connected_nodes:
                for v in self[u].keys():
                    #print(u,v, self[u][v]['weight'])
                    edge_wts.append(self[u][v]['weight'])
        if not connected:
            print('nodes', self.nodes)
            for u in self.nodes:
                for v in self[u].keys():
                    #print(u,v, self[u][v]['weight'])
                    edge_wts.append(self[u][v]['weight'])
        print('edge weights',edge_wts)
        if edge_wts!= None and edge_wts!=[]:
            minima = min(edge_wts)
            maxima = max(edge_wts)
            norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima)
            mapper = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('Spectral'))
            color = []
            for v in edge_wts:
                color.append(mapper.to_rgba(v))
            plt.figure()
            print('minima and maxima', minima, maxima)
            if connected:  # to visualize the connected subgraph in the input
                plt.title(f'Nodes with degree >={self.connected_degree}, input to the solver: {self.filename}\n Number of edges {len(self.edges)}\n'
                          f'Subtracted value: {sub_val}, Target: {self.target}, Feature:{self.feature_type}\n'
                          f'Edge type:{self.edge}, Node weighting:{self.node_wts}')
                nx.draw(self.subgraph(), **plotting_options, edge_color=color, edge_cmap=mapper.cmap, vmin=minima,
                        vmax=maxima, with_labels=False)
                plt.colorbar(mapper)
                plt.savefig(f'{mews}/outputs/figs/{self.filename}.png')

            else:
                plt.title(f'Output from the solver: {self.filename}\n Number of edges {len(self.edges)}\n'
                          f'Subtracted value:{sub_val}, Target: {self.target}, Feature:{self.feature_type}\n'
                          f'Edge type:{self.edge}, Node weighting:{self.node_wts}')
                nx.draw(self, **plotting_options, edge_color=color)
                plt.savefig(f'{mews}/outputs/figs/{self.filename}_out.png')
            plt.show()
        else:
            print('No output produced by the solver')

    def savefiles(self, mews):
        count = 0
        with open(f'{mews}/outputs/nodes/{self.filename}', 'w') as nodes_file:
            for x in self.nodes:
                # print(node)  # solver documentation, 1 or 2
                    print(str(x) + ' ' * 3 + str(self.nodes[x]['label']), file=nodes_file)
                    self.connected_nodes.append(x)
                    # print(str(node) + ' ' + str(0), file=nodes_file)
                    count += 1
        #self.connected_subgraph = self.subgraph(connected_nodes)

        with open(f'{mews}/outputs/edges/{self.filename}', 'w') as edges_file:
            for x in self.nodes:
                # if edge[0] in g1.nodes and edge[1] in g1.nodes:
                for conn in self[x]:
                    print(str(x) + ' ' * 3 + str(conn) + ' ' * 3 + str(self[x][conn]['weight']),
                          file=edges_file)  # original file format was supposed to have 3 spaces

    def run_solver(self, mews):
        os.chdir(mews)
        print('Current directory', os.getcwd())
        cmd = (
            f' java -Xss4M -Djava.library.path=/opt/ibm/ILOG/CPLEX_Studio1210/cplex/bin/x86-64_linux/ '
            f'-cp /opt/ibm/ILOG/CPLEX_Studio1210/cplex/lib/cplex.jar:target/gmwcs-solver.jar '
            f'ru.ifmo.ctddev.gmwcs.Main -e outputs/edges/{self.filename} '
            f'-n outputs/nodes/{self.filename} > outputs/solver/{self.filename}') #training data into the solver
        print(cmd)
        os.system(cmd)
        os.chdir("/home/skapoor/Thesis/graphtools")
        print('Current directory', os.getcwd())

    def read_from_file(self, mews, mat=np.triu_indices(84)):
        if os.path.exists(f'{mews}/outputs/nodes/{self.filename}.out') \
                and os.path.exists(f'{mews}/outputs/edges/{self.filename}.out'):
            with open(f'{mews}/outputs/nodes/{self.filename}.out', 'r') as nodes_file, \
                    open(f'{mews}/outputs/edges/{self.filename}.out', 'r') as edges_file:
                nodes = [x.split('\t') for x in nodes_file.read().split('\n')]
                edges = [x.split('\t') for x in edges_file.read().split('\n')]
                nodes_e = set()
                edges_e = set()
                for a in nodes[:-1]:
                    if a[1] != 'n/a': # since the last line is the subnet score
                        nodes_e.add(int(a[0]))
                feature_indices = []

                # 0 to range(len(mat)), everything in matrix whole corresponding to this edge is feature
                for existing_edge in edges[:-1]:
                    if existing_edge[-1] != 'n/a':
                        edges_e.add((int(existing_edge[0]), int(existing_edge[1]), float(existing_edge[2])))
                        for k in range(len(mat[0])):
                            if (int(existing_edge[0]), int(existing_edge[1])) == (mat[0][k], mat[1][k]):
                                # all_feature_indices.extend([k, k+tri, k+2*tri]) # for the three types FA, n strl, strlen
                                # all_feature_indices.extend([k, k + tri, k + 2 * tri])
                                feature_indices.append(k)
                                # feature_mat = whole.iloc[:, :tri]]
                self.add_nodes_from(nodes_e)
                #self.set_edge_labels(edges_e)
                self.connected_nodes = nodes_e
                #assert self.nodes!= None
                self.add_weighted_edges_from(edges_e)
                self.set_node_labels([node[1] for node in nodes[:-1] if int(node[0]) in nodes_e])
        return feature_indices