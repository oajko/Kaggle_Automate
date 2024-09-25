"""
Maybe do condition for probas and ints only. - i.e., make assertion if it's a bunch of ints inputted for say AUC

Classificaion
Multi-class
Regression
"""

from sklearn.metrics import roc_auc_score

class Metric():
    def __init__(self, metric, y_true, y_pred_prob, y_pred_full):
        metric = metric.lower()
        self.y_true = y_true
        self.y_pred_prob = y_pred_prob
        self.y_pred_full = y_pred_full
        self.value = None

        if metric == 'auc':
            self._roc_score()
        else:
            assert ('Metric not known. choose from ...')
    
    def _roc_score(self):
        self.value = roc_auc_score(self.y_true, self.y_pred_prob[:, 1])
        return self.value
    
    def _accuracy_score(self):
        self.value = (self.y_pred_full == self.y_true).astype(int).sum() / len(self.y_true)
        return self.value
    
    def _precision_score(self):
        self.value = (self.y_pred_full[self.y_true == 1] == 1).sum() / ((self.y_pred_full[self.y_true == 1] == 1).sum() + (self.y_pred_full[self.y_true == 0] == 1).sum())
        return self.value
    
    def _recall_score(self):
        self.value = (self.y_pred_full[self.y_true == 1] == 1).sum() / ((self.y_pred_full[self.y_true == 1] == 1).sum() + (self.y_pred_full[self.y_true == 1] == 0).sum())
        return self.value