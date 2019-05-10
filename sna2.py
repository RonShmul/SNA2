import pandas as pd
import networkx as nx
import numpy as np
from networkx.algorithms import community


# read from csv to graph
def load_network():
    tn_edges = pd.read_csv('thrones-network.csv')
    tn_temp = nx.from_pandas_edgelist(tn_edges, source='Node A', target='Node B', edge_attr='Weight')
    tn = nx.to_undirected(tn_temp)
    return tn


# preprocess for the graph
def preprocess(tn):  # Q1
    remove = [edge for edge in tn.edges().items() if edge[1]['Weight'] < 7]  # todo: only nodes with degree > z
    remove_list = [remove[i][0] for i in range(len(remove))]
    tn.remove_edges_from(remove_list)
    isolated = list(nx.isolates(tn)) # isolate and remove the unconnected nodes
    tn.remove_nodes_from(isolated)
    return tn

#Q2 - clustering coefficient, diameter, density
def graph_attr(tn):
    clustering_coefficient = nx.clustering(tn)
    avg_cluster = nx.average_clustering(tn)
    density = nx.density(tn)
    diameter = nx.diameter(tn)
    avg_path_length = nx.average_shortest_path_length(tn)
    print('cluster for all: ', clustering_coefficient)
    print('average cluster: ', avg_cluster)
    print('density: ', density)
    print('diameter: ', diameter)
    print('avg_path_length: ', avg_path_length)

# Q3 - centrality and top 10 characters
def centrality(tn):
    degree_cen = nx.degree_centrality(tn)
    eigen_cen = nx.eigenvector_centrality(tn)
    closeness_cen = nx.closeness_centrality(tn)
    between_cen = nx.betweenness_centrality(tn)
    print('degree: ', degree_cen)
    print('eigen: ', eigen_cen)
    print('closeness: ', closeness_cen)
    print('betweeness: ', between_cen)
    return degree_cen, eigen_cen, closeness_cen, between_cen

def top_10(degree, eigen, closeness, between):
    top_10_degree = dict(sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10])
    top_10_eigen = dict(sorted(eigen.items(), key=lambda x: x[1], reverse=True)[:10])
    top_10_closeness = dict(sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10])
    top_10_between = dict(sorted(between.items(), key=lambda x: x[1], reverse=True)[:10])
    # print('top ten degree centrality\n')
    print(top_10_degree)
    # print('top ten eigen centrality\n')
    print(top_10_eigen)
    # print('top ten closeness centrality\n')
    print(top_10_closeness)
    # print('top ten between centrality\n')
    print(top_10_between)
    return top_10_between, top_10_closeness, top_10_eigen, top_10_degree

#Q4 - find correlation between centrality measures, find exceptional
def correlation(between, closeness, eigen, degree):
    between_list = list(between.keys())
    closeness_list = list(closeness.keys())
    eigen_list = list(eigen.keys())
    degree_list = list(degree.keys())
    return list(set(between_list) & set(closeness_list) & set(eigen_list) & set(degree_list))

#Q5 - communities
def find_communities(tn):
    communities_generator = community.girvan_newman(tn)
    first_iter_comm = tuple(sorted(c) for c in next(communities_generator))
    second_iter_comm = tuple(sorted(c) for c in next(communities_generator))
    third_iter_comm = tuple(sorted(c) for c in next(communities_generator))
    comm_dict1 = dict(enumerate(first_iter_comm))
    comm_dict2 = dict(enumerate(second_iter_comm))
    comm_dict3 = dict(enumerate(third_iter_comm))
    print(comm_dict1)
    print(comm_dict2)
    print(comm_dict3)
    #statistics todo: look for statistics, maybe in the presentation
    # two_communities = next(communities_generator)
    # three_communities = next(communities_generator)
    # four_communities = next(communities_generator)
    # nodes = tn.number_of_nodes()
    # first_comm_2 = len(two_communities[0])/nodes
    # second_comm_2 = len(two_communities[1])/nodes
    # first_comm_3 = len(three_communities[0]) / nodes
    # second_comm_3 = len(three_communities[1]) / nodes
    # third_comm_3 = len(three_communities[2]) / nodes
    # first_comm_4 = len(four_communities[0]) / nodes
    # second_comm_4 = len(four_communities[1]) / nodes
    # third_comm_4 = len(four_communities[2]) / nodes
    # fourth_comm_4= len(four_communities[3]) / nodes

# Q6 - graph with communities and centrality measure
def centrality_communities_graph():
    pass


# Q7 - link prediction
def jaccard_lp(tn):
    pred_jc = nx.jaccard_coefficient(tn)
    pred_dict = {}
    for u, v, p in pred_jc:
        pred_dict[(u, v)] = p
    return sorted(pred_dict.items(), key=lambda x:x[1], reverse=True)[:10]

def preferential_lp(tn):
    pred_pa = nx.preferential_attachment(tn)
    pred_dict = {}
    for u, v, p in pred_pa:
        pred_dict[(u, v)] = p
    return sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)[:10]


m_tn = load_network()
nx.draw(m_tn, with_labels=True)
tn_new = nx.Graph(m_tn)
# print(nx.info(tn_new))
tn_clean = preprocess(tn_new)
# print(nx.info(tn_clean))
nx.draw(tn_clean, with_labels=True)
nx.draw_spring(tn_clean, with_labels=True)
nx.draw_circular(tn_clean, with_labels=True)
graph_attr(tn_clean)
degree, eigen, closeness, between = centrality(tn_clean)
between, closeness, eigen, degree = top_10(degree, eigen, closeness, between)
cor = correlation(between, closeness, eigen, degree)
print(cor)
find_communities(tn_clean)
jc = jaccard_lp(tn_clean)
pa = preferential_lp(tn_clean)
#print(jc)
#print(pa)
# print(tn.degree('Jon'))
print('end')
