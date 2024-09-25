from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import pandas as pd
import metrics
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


"""
Function of this program is to get baseline scores for different models.
No sophistication in feature handling, just raw data cleaned for objects.
"""

class Cleaner():
    def __init__(self, split, random_state, skf_splits, categories):
        self.split = split
        self.random_state = random_state
        self.skf_splits = skf_splits
        self.categories = categories
    # Get indexes of splits for tts or skf
    def _data_splits(self):
        if self.split == 'tts':
            train_index = list(train_test_split(
                self.x, self.y, random_state=self.random_state)[0].index)
            split_index = [(train_index, list(self.x.drop(train_index).index))]
        elif self.split == 'skf':
            skf = StratifiedKFold(n_splits=self.skf_splits, random_state=self.random_state, shuffle=True)
            split_index = list(skf.split(self.x, self.y))
        return split_index
    # cleans object data to OHE categories (if user inputs category) and label encodes rest
    def _data_clean(self):
        self.x = pd.get_dummies(self.x, columns=self.categories, prefix=self.categories).astype(int)
        for obj in self.x.select_dtypes(include=object).columns:
            # Filters features for pure str data only - ignores int dtype in feature and nan.
            obj_mask = self.x[(self.x[obj].dtype == object)& (~self.x[obj].isna())][obj]
            label = LabelEncoder().fit(obj_mask.values)
            self.x.loc[obj_mask.index, obj] = label.transform(obj_mask.values)
            self.x[obj] = self.x[obj].astype(float)


class AutoClassificationBaseline(Cleaner):
    def __init__(self, x, y, auto_models = ['xgb'], random_state = None, skf_splits = 5, metric = 'auc', data_split = 'tts', categories = None, const_params = False):
        super().__init__(data_split, random_state, skf_splits, categories)
        self.y = y.dropna()
        self.x = x.loc[self.y.index]
        if type(metric) != 'list':
            self.metric = [metric]
        else:
            self.metric = metric
        self.random_state = random_state
        self.const_params = const_params
        self.splits = self._data_splits()
        self._data_clean()
        # Allows list to pass through for multiple baselines with one line of code
        for model in auto_models:
            if model == 'xgb':
                self.xgb_classification()
            elif model == 'light':
                self.light_classification()
            elif model == 'cat':
                self.cat_classification()
            elif model == 'randf':
                self.randf_classification()
            elif model == 'ada':
                self.ada_classification()
    # Auto baseline for different models 
    def xgb_classification(self):
        avg_metrics = {}
        for train, test in self.splits:
            X_train, y_train = self.x.loc[train], self.y[train]
            X_test, y_test = self.x.loc[test], self.y[test]
            model = XGBClassifier(random_state = self.random_state).fit(X_train, y_train)
            # Allow equal footing with eta and estimators (models have different default params)
            if self.const_params is True:
                model = XGBClassifier(random_state = self.random_state, eta = 0.1, estimators = 600).fit(X_train, y_train)
            else:
                model = XGBClassifier(random_state = self.random_state).fit(X_train, y_train)
            # Loop over several metrics via user input/ choice
            for i in self.metric:
                avg_metrics[i] = avg_metrics.get(i, 0)
                avg_metrics[i] = avg_metrics[i] + metrics.Metric(metric = i, y_true = y_test, y_pred_prob = model.predict_proba(X_test), y_pred_full = model.predict(X_test)).value / len(self.splits)
        print(f'xgb_metrics: {[(p, j) for p, j in avg_metrics.items()]}')

    def light_classification(self):
        avg_metrics = {}
        for train, test in self.splits:
            X_train, y_train = self.x.loc[train], self.y[train]
            X_test, y_test = self.x.loc[test], self.y[test]
            if self.const_params:
                model = LGBMClassifier(random_state = self.random_state, learning_rate = 0.1, n_estimators = 600).fit(X_train, y_train)
            else:
                model = LGBMClassifier(random_state = self.random_state).fit(X_train, y_train)
            for i in self.metric:
                avg_metrics[i] = avg_metrics.get(i, 0)
                avg_metrics[i] = avg_metrics[i] + metrics.Metric(metric = i, y_true = y_test, y_pred_prob = model.predict_proba(X_test), y_pred_full = model.predict(X_test)).value / len(self.splits)
        print(f'xgb_metrics: {[(p, j) for p, j in avg_metrics.items()]}')

    def cat_classification(self):
        avg_metrics = {}
        for train, test in self.splits:
            X_train, y_train = self.x.loc[train], self.y[train]
            X_test, y_test = self.x.loc[test], self.y[test]
            if self.const_params:
                model = CatBoostClassifier(random_state = self.random_state, learning_rate = 0.1, n_estimators = 600).fit(X_train, y_train)
            else:
                model = CatBoostClassifier(random_state = self.random_state).fit(X_train, y_train)
            for i in self.metric:
                avg_metrics[i] = avg_metrics.get(i, 0)
                avg_metrics[i] = avg_metrics[i] + metrics.Metric(metric = i, y_true = y_test, y_pred_prob = model.predict_proba(X_test), y_pred_full = model.predict(X_test)).value / len(self.splits)
        print(f'xgb_metrics: {[(p, j) for p, j in avg_metrics.items()]}')

    def randf_classification(self):
        avg_metrics = {}
        for train, test in self.splits:
            X_train, y_train = self.x.loc[train], self.y[train]
            X_test, y_test = self.x.loc[test], self.y[test]
            model = RandomForestClassifier(random_state=self.random_state).fit(X_train, y_train)
            for i in self.metric:
                avg_metrics[i] = avg_metrics.get(i, 0)
                avg_metrics[i] = avg_metrics[i] + metrics.Metric(metric = i, y_true = y_test, y_pred_prob = model.predict_proba(X_test), y_pred_full = model.predict(X_test)).value / len(self.splits)
        print(f'xgb_metrics: {[(p, j) for p, j in avg_metrics.items()]}')

    def ada_classification(self):
        avg_metrics = {}
        for train, test in self.splits:
            X_train, y_train = self.x.loc[train], self.y[train]
            X_test, y_test = self.x.loc[test], self.y[test]
            model = AdaBoostClassifier(random_state=self.random_state).fit(X_train, y_train)
            for i in self.metric:
                avg_metrics[i] = avg_metrics.get(i, 0)
                avg_metrics[i] = avg_metrics[i] + metrics.Metric(metric = i, y_true = y_test, y_pred_prob = model.predict_proba(X_test), y_pred_full = model.predict(X_test)).value / len(self.splits)
        print(f'xgb_metrics: {[(p, j) for p, j in avg_metrics.items()]}')

