import numpy as np
import lightgbm as lgb
import random
import math

def countSplitNodes(tree):
    root = tree['tree_structure']
    def counter(root):
        if 'split_index' not in root:
            return 0
        return 1 + counter(root['left_child']) + counter(root['right_child'])
    ans = counter(root)
    return ans

def getItemByTree(tree, item='split_feature'):
    root = tree.raw['tree_structure']
    split_nodes = tree.split_nodes
    res = np.zeros(split_nodes+tree.raw['num_leaves'], dtype=np.int32)
    if 'value' in item or 'threshold' in item or 'split_gain' in item:
        res = res.astype(np.float64)
    def getFeature(root, res):
        if 'child' in item:
            if 'split_index' in root:
                node = root[item]
                if 'split_index' in node:
                    res[root['split_index']] = node['split_index']
                else:
                    res[root['split_index']] = node['leaf_index'] + split_nodes # need to check
            else:
                res[root['leaf_index'] + split_nodes] = -1
        elif 'value' in item:
            if 'split_index' in root:
                res[root['split_index']] = root['internal_'+item]
            else:
                res[root['leaf_index'] + split_nodes] = root['leaf_'+item]
        else:
            if 'split_index' in root:
                res[root['split_index']] = root[item]
            else:
                res[root['leaf_index'] + split_nodes] = -2
        if 'left_child' in root:
            getFeature(root['left_child'], res)
        if 'right_child' in root:
            getFeature(root['right_child'], res)
    getFeature(root, res)
    return res

def getTreeSplits(model):
    featurelist = []
    threhlist = []
    trees = []
    for idx, tree in enumerate(model['tree_info']):
        trees.append(TreeInterpreter(tree))
        featurelist.append(trees[-1].feature)
        threhlist.append(getItemByTree(trees[-1], 'threshold'))
    return (trees, featurelist, threhlist)


def getChildren(trees):
    listcl = []
    listcr = []
    for idx, tree in enumerate(trees):
        listcl.append(getItemByTree(tree, 'left_child'))
        listcr.append(getItemByTree(tree, 'right_child'))
    return(listcl, listcr)

class TreeInterpreter(object):
    def __init__(self, tree):
        self.raw = tree
        self.split_nodes = countSplitNodes(tree)
        self.node_count = self.split_nodes# + tree['num_leaves']
        self.value = getItemByTree(self, item='value')
        self.feature = getItemByTree(self)
        self.gain = getItemByTree(self, 'split_gain')
        # self.leaf_value = getLeafValue(tree)

class ModelInterpreter(object):
    def __init__(self, model, tree_model='lightgbm'):
        print("Model Interpreting...")
        self.tree_model = tree_model
        model = model.dump_model()
        self.n_features_ = model['max_feature_idx'] + 1
        self.trees, self.featurelist, self.threshlist = getTreeSplits(model)
        self.listcl, self.listcr = getChildren(self.trees)

    def GetTreeSplits(self):
        return (self.trees, self.featurelist, self.threshlist)

    def GetChildren(self):
        return (self.listcl, self.listcr)

    def EqualGroup(self, n_clusters, args):
        vectors = {}
        # n_feature = 256
        for idx,features in enumerate(self.featurelist):
            vectors[idx] = set(features[np.where(features>0)])
        keys = random.sample(vectors.keys(), len(vectors))
        clusterIdx = np.zeros(len(vectors))
        groups = [[] for i in range(n_clusters)]
        trees_per_cluster = len(vectors)//n_clusters
        mod_per_cluster = len(vectors) % n_clusters
        begin = 0
        for idx in range(n_clusters):
            for jdx in range(trees_per_cluster):
                clusterIdx[keys[begin]] = idx
                begin += 1
            if idx < mod_per_cluster:
                clusterIdx[keys[begin]] = idx
                begin += 1
        print([np.where(clusterIdx==i)[0].shape for i in range(n_clusters)])
        return clusterIdx
