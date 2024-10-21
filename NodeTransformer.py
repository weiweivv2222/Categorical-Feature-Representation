import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline 
from node2vec import Node2Vec
import networkx as nx
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler# instantiate labelencoder object
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier



#Custom Transformer that extracts columns passed as argument to its constructor 
class NodeTransformer( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self,mode='Extended', n_estimators =5, impurity_reweight = True,
                 max_depth = 5, walk_length = 5, n_walks= 50, window = 5, dimension=2, random_state=0):
        self.walk_length = 5
        self.mode = mode
        self.impurity_reweight = impurity_reweight
        self.n_walks = n_walks
        self.dimension = dimension
        self.window = window
        self.max_depth = max_depth
        self.n_estimators= n_estimators
        self.inner_tree = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.graph_list = None
        self.estimators = None
        self.node2model = None
        self.random_state= random_state
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        self.inner_tree.fit(X,y)
        self.estimators = self.inner_tree.estimators_
        
        Glist_tree, Glist_leaf_graphs= self.create_graph_list(self.inner_tree)
        if self.mode=='Compact':
            self.graph_list = Glist_leaf_graphs
        else:
            self.graph_list=Glist_tree
        #and now the nodes
        
        node2vec_list = []
        for G in  self.graph_list:
            node2vec = Node2Vec(G, dimensions=self.dimension, 
                                walk_length=self.walk_length, 
                                num_walks=self.n_walks)
            node2vec_list.append(node2vec)
        
        
        models_list = []

        for node2v in node2vec_list:
            model = node2v.fit(window=self.window, min_count=1)
            models_list.append(model)
        
        
        self.node2model = models_list
        
        
        
        return self 
    
    
    
   

    def create_graph_list(self,estimators_trees):
        G_list = []
        G_list_leaves =[]

        for estimator in estimators_trees.estimators_:
            G=nx.Graph() #
            G_leaves = nx.Graph() # leaves graph

            n_nodes = estimator.tree_.node_count
            children_left = estimator.tree_.children_left
            children_right = estimator.tree_.children_right
            feature = estimator.tree_.feature
            threshold = estimator.tree_.threshold
            node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
            is_leaves = np.zeros(shape=n_nodes, dtype=bool)
            stack = [(0, -1,0)]  # seed is the root node id and its parent depth

            while len(stack) > 0:
                node_id, parent_depth,parent_id = stack.pop()
                node_depth[node_id] = parent_depth + 1
                G.add_node(str(node_id))
                G.add_edge(str(parent_id),str(node_id))
                # If we have a test node
                if (children_left[node_id] != children_right[node_id]):

                    stack.append((children_left[node_id], parent_depth + 1,node_id))
                    stack.append((children_right[node_id], parent_depth + 1,node_id))
                else:
                    is_leaves[node_id] = True
            #append created graph
            if self.mode=='Compact':
                for i in range(0,len(is_leaves)):
                    if (is_leaves[i]):
                        G_leaves.add_node(str(i))

                        for j in range(0,len(is_leaves)):
                               if (is_leaves[j]):         
                                    if i!=j:
                                        G_leaves.add_node(str(j))
                                        G_leaves.add_edge(str(i),str(j),
                                                         weight=nx.shortest_path_length(G,source=str(i),target=str(j)))

            G_list.append(G)
            G_list_leaves.append(G_leaves)
            #return multi graph
        return G_list,G_list_leaves

    
    
    
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        
        
        reps = []
        for estimator,model in zip(self.inner_tree.estimators_,self.node2model):

            leave_id = estimator.apply(X)
            impurities = estimator.tree_.impurity
            
            vect_rep = []

            for i in leave_id:
                impurity =  1 - impurities[i]
                if self.impurity_reweight:
                    vect_rep.append(model.wv[str(i)]*impurity)
                else:
                    vect_rep.append(model.wv[str(i)])
        
            if len(reps)==0:
                reps.append(vect_rep)
            else:
                reps= np.array(reps).squeeze()
                reps = np.hstack((reps,np.array(vect_rep))) 
                    
        
        return reps
    
    #Method that describes what we need this transformer to do
    def transform_matrix( self, X, y = None ):
        
        reps = []
        #print(np.shape(reps))
        for estimator,model in zip(self.inner_tree.estimators_,self.node2model):

            leave_id = estimator.apply(X)

            vect_rep = []

            for i in leave_id:
                vect_rep.append(model.wv[str(i)])
            #print("reps shape")
            #print(np.shape(reps))
            if len(reps)==0:
                reps.append(vect_rep)
                reps = np.expand_dims(np.squeeze(reps),-1)
    
            else:
               
                #print(np.shape(reps))
                #print("else reps_vec")
                #print(np.shape(vect_rep))
                reps = np.append(reps,np.expand_dims(np.array(vect_rep),-1),axis=2) 
                    
        
        return reps
