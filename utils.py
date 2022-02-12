import numpy as np 
def graph_expansion(G, new_edges) -> np.array:
    """
    The function adds nodes to already existing nodes.
    :graph: numpy array
    :new_edges: int 
    """
    num_edges = len(list(G[0]))
    for i in range(num_edges):
        num_edges = len(list(G[0]))
        support_matrix = np.zeros(((num_edges-new_edges*i) + new_edges * (i + 1), (num_edges-new_edges*i) + new_edges * (i + 1)))
        support_matrix[i,:] = 1
        support_matrix[:,i] = 1
        support_matrix[:num_edges, :num_edges] = G
        G = support_matrix
    return G



def slicing(atom_features):
    """
    Creates the new atom features for each nodes.
    :atom_features: numpy array
    """
    new_features = []
    for enum, i in enumerate(atom_features):
        for feature in atom_features[enum]:
            new_features.append(feature)
        new_features.append(0)
    return np.array(new_features).reshape(len(new_features), -1)