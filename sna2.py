import pandas as pd
import networkx as nx
import numpy as np
from networkx.algorithms import community
import matplotlib.pyplot as plt

# read from csv to graph
def load_network():
    tn_edges = pd.read_csv('thrones-network.csv')
    tn_temp = nx.from_pandas_edgelist(tn_edges, source='Node A', target='Node B', edge_attr='Weight')
    tn = nx.to_undirected(tn_temp)
    tn_new = nx.Graph(tn)
    return tn_new

# preprocess for the graph
def preprocess(tn):  # Q1
    remove = [edge for edge in tn.edges().items() if edge[1]['Weight'] < 7]
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
    print('clustering for all: ', clustering_coefficient)
    print(len(clustering_coefficient))
    print('average clustering: ', avg_cluster)
    print('density: ', density)
    print('diameter: ', diameter)
    print('avg path length: ', avg_path_length)

# Q3 - centrality and top 10 characters
def centrality(tn):
    degree_cen = nx.degree_centrality(tn)
    eigen_cen = nx.eigenvector_centrality(tn)
    closeness_cen = nx.closeness_centrality(tn)
    between_cen = nx.betweenness_centrality(tn)
    return degree_cen, eigen_cen, closeness_cen, between_cen

def top_10(degree, eigen, closeness, between):
    top_10_degree = dict(sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10])
    top_10_eigen = dict(sorted(eigen.items(), key=lambda x: x[1], reverse=True)[:10])
    top_10_closeness = dict(sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10])
    top_10_between = dict(sorted(between.items(), key=lambda x: x[1], reverse=True)[:10])
    df_deg = pd.DataFrame.from_dict(top_10_degree, orient='index')
    df_eigen = pd.DataFrame.from_dict(top_10_eigen, orient='index')
    df_close = pd.DataFrame.from_dict(top_10_closeness, orient='index')
    df_between = pd.DataFrame.from_dict(top_10_between, orient='index')
    print('top ten degree centrality', df_deg)
    print('top ten eigen centrality', df_eigen)
    print('top ten closeness centrality', df_close)
    print('top ten between centrality', df_between)
    return top_10_between, top_10_closeness, top_10_eigen, top_10_degree

#Q4 - find correlation between centrality measures, find exceptional
def correlation(between, closeness, eigen, degree):
    between_list = list(between.keys())
    closeness_list = list(closeness.keys())
    eigen_list = list(eigen.keys())
    degree_list = list(degree.keys())
    print('correlation between centrality measures: ',list(set(between_list) & set(closeness_list) & set(eigen_list) & set(degree_list)))

#Q5 - communities
def find_communities(tn):
    communities_generator = community.girvan_newman(tn)
    for i in range(0,2):
        com = tuple(sorted(c) for c in next(communities_generator))
        comm_dict = dict(enumerate(com))
        final = dict()
        for key in comm_dict:
            for item in comm_dict[key]:
                final[item] = key
        final_df = pd.DataFrame.from_dict(final, orient='index')
        print('Partition '+str(i), final_df)
    com = tuple(sorted(c) for c in next(communities_generator))
    comm_dict = dict(enumerate(com))
    partition = dict()
    for key in comm_dict:
        for item in comm_dict[key]:
            partition[item] = key
    final_df = pd.DataFrame.from_dict(partition, orient='index')
    print('Partition ' + str(2), final_df)
    return com
    #statistics todo: statistics
    # nodes = tn.number_of_nodes()
    # first_comm_2 = len(two_communities[0])/nodes
    # second_comm_2 = len(two_communities[1])/nodes

# Q6 - graph with communities and centrality measure
def centrality_communities_graph(tn, partition):
    d = nx.degree_centrality(tn)
    pos = nx.spring_layout(tn)
    colors = ['#c20078', '#8e82fe', '#feb308', '#02c14d']
    for i in range(len(partition)):
        sub = tn.subgraph(partition[i])
        deg_size = [(d[node] * 1500) for node in sub.node]
        #nx.draw_networkx(sub, node_color=colors[i], with_labels=True, pos=pos, node_size=deg_size)
        nx.draw_networkx_nodes(sub, pos, node_size=deg_size, node_color=colors[i])
        nx.draw_networkx_edges(sub, pos, alpha=0.3)
        nx.draw_networkx_labels(sub, pos, font_size=7, font_color='black')
    plt.show()

# Q7 - link prediction
def jaccard_lp(tn):
    pred_jc = nx.jaccard_coefficient(tn)
    pred_dict = {}
    for u, v, p in pred_jc:
        pred_dict[(u, v)] = p
    print(sorted(pred_dict.items(), key=lambda x:x[1], reverse=True)[:10])

def preferential_lp(tn):
    pred_pa = nx.preferential_attachment(tn)
    pred_dict = {}
    for u, v, p in pred_pa:
        pred_dict[(u, v)] = p
    print(sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)[:10])


m_tn = load_network()
#nx.draw(m_tn, with_labels=True)
# print(nx.info(tn_new))
tn_clean = preprocess(m_tn)
# print(nx.info(tn_clean))
#nx.draw_spring(tn_clean, with_labels=True)
#graph_attr(tn_clean)
degree, eigen, closeness, between = centrality(tn_clean)
between, closeness, eigen, degree = top_10(degree, eigen, closeness, between)
correlation(between, closeness, eigen, degree)
#part = find_communities(tn_clean)
#centrality_communities_graph(tn_clean, part)
jaccard_lp(tn_clean)
preferential_lp(tn_clean)
