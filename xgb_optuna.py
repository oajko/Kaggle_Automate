import optuna
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

const_params = {'device': 'cuda',
                'nthread': 4,
                'random_state': 42,
                'early_stopping_rounds': 100,
                'n_estimators': 5000,
                'tree_method': 'hist',
                'eval_metric': 'auc',
                'objective': 'binary:logistic'}

def objective(trial):
    params = {**const_params,
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'eta': trial.suggest_float('eta', 0.01, 0.06),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'max_leaves': trial.suggest_int('max_leaves', 8, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 100),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 0, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1)
    }
    # Only get non constant params
    param_used = dict(list(params.items())[8:])
    print(param_used)
    
    X_train, X_test, y_train, y_test = train_test_split(optuna_x, optuna_y, random_state = 42, stratify = optuna_y)
    model = XGBClassifier(**params).fit(X_train, y_train, eval_set = [(X_test, y_test)], verbose = 0)
    
    auc_pred = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    param_list.append((auc_pred, param_used))

    # Using TTS. So, want to get top 8 params for CV (reduce error).
    if len(param_list) > 8:
        param_list.pop(min(range(len(param_list)), key=lambda i: param_list[i][0]))
    
    print(auc_pred, '\n')
    return auc_pred
    
if __name__ == '__main__':
  # Set training and test set for ease of use.
    optuna_x = credit_train.iloc[:, 1:]
    optuna_y = train_df[train_df.sk_id_curr.isin(train_feature.sk_id_curr)].target
    optuna_unlabel = credit_test.iloc[:, 1:]
    param_list = []
    
    study = optuna.create_study(direction = 'maximize')
    study.optimize(objective, n_trials = 200)
    print(*param_list, sep = '\n')
    
    def auto_test_optuna(X, y, param_checker, unlabel = None, do_test = False):
        if do_test is True:
            oofs_submit = np.zeros(shape = (len(unlabel), ))
        
        SPLITS = 5
        check_metric = 0
        best_param = {}
        params_loop = [j for i, j in param_checker]
        for idx, i in enumerate(params_loop):
            params = {**const_params, **i}
            kfold_oofs_metric = 0
            skf = StratifiedKFold(n_splits = SPLITS, random_state = 42, shuffle = True)
            for train, test in skf.split(X, y):
                X_train, y_train = X.iloc[train], y.iloc[train]
                X_test, y_test = X.iloc[test], y.iloc[test]
                model = XGBClassifier(**params).fit(X_train, y_train, eval_set = [(X_test, y_test)], verbose = 0)
                kfold_oofs_metric += roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) / SPLITS

                if do_test is True:
                    oofs_submit += model.predict_proba(unlabel)[:, 1] / SPLITS
                  
            # filter for best param
            if kfold_oofs_metric > check_metric:
                best_param = i
                check_metric = kfold_oofs_metric

            print(f'Metric {idx + 1} got score of: {kfold_oofs_metric}')

      # Submit the file. Change as required
        if do_test is True:
            ans = pd.DataFrame({'SK_ID_CURR': test_df.iloc[:, 0],'TARGET': oofs_submit})
            ans.to_csv('submission.csv', index = False)
            print('Submitted!')
            unlabel = None
      # Recursive to submit the best params
        if unlabel is not None:
            auto_test_optuna(X, y, param_checker = [best_param], unlabel = unlabel, do_test = True)
    
    # To submit unlabel set, we set unlabel into it. Else just don't place unlabel and it only finds best params.
    auto_test_optuna(optuna_x, optuna_y, param_list)
