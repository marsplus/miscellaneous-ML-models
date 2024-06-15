## Based on the implementation from https://github.com/ddbourgin/numpy-ml/blob/master/numpy_ml/trees/dt.py
## New features:
##
import numpy as np

class Node:
    def __init__(self, left, right, rule):
        self.left = left
        self.right = right
        self.feature = rule[0]
        self.threshold = rule[1]

class Leaf:
    def __init__(self, value):
        self.value = value


class DecisionTree:
    def __init__(
        self,
        classifier=True,
        max_depth=None,
        n_feats=None,
        criterion="entropy",
        seed=None):
        if seed:
            np.random.seed(seed)

        self.depth = 0
        self.root = None
        
        self.max_depth = max_depth if max_depth else np.inf
        self.criterion = criterion
        self.n_feats = n_feats
        self.classifier = classifier

        if classifier and criterion in ('mse'):
            raise ValueError
        if not classifier and criterion in ('entropy', 'gini'):
            raise ValueError

    def fit(self, X, Y):
        """
            Construct a tree data structure over the data.
        """
        X = self._imputation(X)
        self.n_classes = max(Y) + 1
        self.n_feats = X.shape[1] if not self.n_feats else min(X.shape[1, self.n_feats])
        self.root = self._grow(X, Y)

    def predict(self, X):
        """
            The prediciton is equivalent to traverse the tree
            from the root to a leaf.
        """
        return np.array([self._traverse(x, self.root) for x in X])

    def predict_class_probs(self, X):
        assert self.classifier
        return np.array([self._traverse(x, self.root, prob=True) for x in X])
    
    
    def _imputation(self, X, mode='median'):
        """
            Impute missing values with either 1) mean or 2) median
        """
        X_copy = np.copy(X)
        if mode == 'median':
            impute_func = np.nanmedian
        elif mode == 'mean':
            impute_func = np.nanmean
        else:
            raise ValueError
        
        N, M = X_copy.shape
        for col in range(M):
            impute_val = impute_func(X_copy[:, col])
            X_copy[np.isnan(X_copy[:, col]), :] = impute_val
        return X_copy


    def _grow(self, X, Y, cur_depth=0):
        """
            Recursively construct the tree.
        """
        ## base cases for recursion
        # a single class
        if len(set(Y)) == 1:
            if self.classifier:
                prob = np.zeros(self.n_classes)
                prob[Y[0]] = 1.0
            return Leaf(prob) if self.classifier else Leaf(Y[0])

        # max_depth reached
        if cur_depth >= self.max_depth:
            v = np.mean(Y, axis=0)
            if self.classifier:
                v = np.bincount(Y, minlength=self.n_classes) / len(Y)
            return Leaf(v)
        
        cur_depth += 1
        self.depth = max(self.depth, cur_depth)
        N, M = X.shape
        feat_idxs = np.random.choice(M, self.n_feats, replace=False)

        # feat: an ineteger
        # thresh: a threshold used to segment X based on the values of `feat`
        feat, thresh = self._segment(X, Y, feat_idxs)
        l = np.argwhere(X[:, feat] <= thresh).flatten()
        r = np.argwhere(X[:, feat] > thresh).flatten()

        left = self._grow(X[l, :], Y[l], cur_depth)
        right = self._grow(X[r, :], Y[r], cur_depth)
        return Node(left, right, (feat, thresh))


    def _segment(self, X, Y, feat_idxs):
        """
            feat_idxs (Iterable): candidate indices for finding the split point
        """
        best_gain = -np.inf
        split_idx, split_thresh = None, None
        for i in feat_idxs:
            # the values of the i-th feature
            vals = X[:, i]
            # return a sorted array of unique elements
            levels = np.unique(vals)
            # the midpoints of each consecutive unique value
            thresholds = (levels[:-1] + levels[1:]) / 2.0 if len(levels) > 1 else levels

            ## EXPENSIVE
            gains = [self._impurity_gain(Y, th, vals) for th in thresholds]
            max_gain = max(gains)
            if max_gain > best_gain:
                best_gain = max_gain
                split_idx = i
                split_thresh = thresholds[np.argmax(gains)] 
            return (split_idx, split_thresh)


    def _impurity_gain(self, Y, split_thresh, feat_values):
        if self.criterion == 'mse':
            loss = mse
        elif self.criterion == 'entropy':
            loss = entropy
        elif self.criterion == 'gini':
            loss = gini
        else:
            raise ValueError
        
        original_loss = loss(Y)
        l = np.argwhere(feat_values <= split_thresh).flatten()
        r = np.argwhere(feat_values > split_thresh).flatten()
        if len(l) == 0 or len(r) == 0:
            return 0
        
        n = len(Y)
        n_l, n_r = len(l), len(r)
        loss_l, loss_r = loss(Y[l]), loss(Y[r])
        new_loss = (n_l / n) * loss_l + (n_r / n) * loss_r
        diff = original_loss - new_loss
        return diff
        

    def _traverse(self, X, node, prob=False):
        # reach a leaf node
        if isinstance(node, Leaf):
            if self.classifier:
                return node.value if prob else np.argmax(node.value)
            return node.value
        # keep traversing the tree
        if X[node.feature] <= node.threshold:
            return self._traverse(X, node.left, prob)
        return self._traverse(X, node.right, prob)


def mse(y):
    return np.mean((y - np.mean(y)) ** 2)

def entropy(y):
    hist = np.bincount(y)
    prob = hist / np.sum(hist)
    return -np.sum([p * np.log2(p) for p in prob if p > 0])

def gini(y):
    hist = np.bincount(y)
    prob = hist / np.sum(hist)
    return 1 - np.sum(prob**2)

