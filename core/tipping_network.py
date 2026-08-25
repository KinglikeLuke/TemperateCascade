from networkx import DiGraph
import networkx as nx
from core.coupling import cusp_derivative_coupling # TODO ugly hack
import numpy as np


class tipping_network(DiGraph):

    def __init__( self, incoming_graph_data=None, **attr):
        DiGraph.__init__( self, incoming_graph_data=None, **attr)
        
    def add_element( self, tipping_element ):
        ind = self.number_of_nodes()
        super().add_node( ind, data = tipping_element )
        self._node[ind]['lambda_f'] = tipping_element.dxdt_diag()
        self._node[ind]['lambda_jac'] = tipping_element.jac_diag()

    def update_element(self, old_element_idx, new_element):
        self.nodes[old_element_idx].clear()
        self.nodes[old_element_idx].update(data = new_element)
        self._node[old_element_idx]['lambda_f'] = new_element.dxdt_diag()
        self._node[old_element_idx]['lambda_jac'] = new_element.jac_diag()
        # Because of the way the couplings are implemented, we need to update the couplings with the new element params
        # TODO change the way derivative couplings are implemented
        for edge in list(self.edges(old_element_idx, data=True)):
            coupling = edge[2]['data']
            if isinstance(coupling, cusp_derivative_coupling):
                self.remove_edge(edge[0], edge[1])
                self.add_coupling(edge[0], edge[1], cusp_derivative_coupling(coupling._strength, new_element.get_par()))

    def add_coupling( self, from_id, to_id, coupling):
        super().add_edge( from_id, to_id, data = coupling)
        self[from_id][to_id]['lambda_f'] = coupling.dxdt_cpl()
        self[from_id][to_id]['lambda_jac'] = coupling.jac_cpl()
        self[from_id][to_id]['lambda_jac_diag'] = coupling.jac_diag()

    def set_param( self, node_id, key, val ):
        element = self._node[node_id]['data']
        element.set_par( key, val)
        self._node[node_id]['lambda_f'] = self._node[node_id]['data'].dxdt_diag()
        self._node[node_id]['lambda_jac'] = self._node[node_id]['data'].jac_diag()

    def get_tip_states( self, x):
        tipped = [self._node[i]['data'].tip_state()(x[i]) for i in self.nodes()]
        return np.array( tipped )
    
    def get_number_tipped( self, x):
        return np.count_nonzero( self.get_tip_states( x ) )

    def f( self, t, x):
        f = np.zeros( self.number_of_nodes() )
        for node in self.nodes(data=True):
            ind = node[0]
            x_comp = x[ind]
            f[ind] = node[1]['lambda_f'].__call__( t, x_comp)
        for edge in self.edges(data=True):
            from_id = edge[0]
            to_id = edge[1]
            lmd = edge[2]['lambda_f']
            f[to_id] += lmd.__call__( t, x[from_id], x[to_id])
        return f

    def jac(self, t, x):
        jac = np.zeros((self.number_of_nodes(), self.number_of_nodes()))
        for node in self.nodes(data=True):
            ind = node[0]
            x_comp = x[ind]
            jac[ind,ind] = node[1]['lambda_jac'].__call__( t, x_comp)
        for edge in self.edges(data=True):
            from_id = edge[0]
            to_id = edge[1]
            lmd = edge[2]['lambda_jac']
            jac[to_id, from_id] = lmd.__call__( t, x[from_id], x[to_id] )
            lmd_diag = edge[2]['lambda_jac_diag']
            jac[to_id, to_id] += lmd_diag.__call__( t, x[from_id], x[to_id] )
        return jac

    def compute_impact_matrix(self):
        n = self.number_of_nodes()
        impact_matrix = [[lambda t, x1, x2: 0 for j in range(n)] for i in range(n)]
        for edge in self.edges.data():
            print(edge[2]['data'].bif_impact())
            impact_matrix[edge[1]][edge[0]]=edge[2]['data'].bif_impact()
            # if self.get_node_types()[edge[1]]=='cusp':
            #     impact_matrix[edge[1]][edge[0]]=edge['data'].dxdt_cpl()
            # elif self.get_node_types()[edge[1]]=='hopf':
            #     impact_matrix[edge[1]][edge[0]]=edge['data']
        return impact_matrix

    def get_node_parameters(self, t=0):
        """Returns all parameters of the nodes in a dictionary at time 0 for time dependent variables"""
        node_params = {}
        for node_id, node_data in self.nodes(data=True):
            element = node_data['data']
            par = element.get_par()
            if callable(par['c']):
                par['c'] = par['c'](t)
            node_params[node_id] = par
        return node_params

    def get_out_strengths(self):
        """Returns the connection strengths between nodes in a nested dict, format {from_id: {to_id: value}}"""
        connection_strengths = {}
        for from_id, to_id, edge_data in self.edges(data=True):
            coupling = edge_data['data']
            if from_id not in connection_strengths:
                connection_strengths[from_id] = {}
            connection_strengths[from_id][to_id] = coupling.strength
        return connection_strengths

    def get_in_strengths(self):
        """Returns the connection strengths between nodes in a nested dict, format {to_id: {from_id: value}}"""
        connection_strengths = {}
        for from_id, to_id, edge_data in self.edges(data=True):
            coupling = edge_data['data']
            if to_id not in connection_strengths:
                connection_strengths[to_id] = {}
            connection_strengths[to_id][from_id] = coupling.strength
        return connection_strengths