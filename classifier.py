
import lightgbm as lgb
import seaborn as sns
import logging
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.externals import joblib

from compute_scores import normalized_utility_score

lgb_classifier_params = {'num_leaves': 60,
                         'min_data_in_leaf': 120,
                         'objective': 'binary',
                         'max_depth': -1,
                         'learning_rate': 0.2,
                         'reg_alpha': 0,
                         'reg_lambda': 0,
                         'metric': 'auc',
                         'verbosity': -1,
                         'early_stopping_rounds': 5,
                         'scale_pos_weight': 20,
                         }


def save_model(model, path):
    joblib.dump(model.model, path)


def load_model(path):
    return joblib.load(path)


def save_features_importance(features_importance, features_names, path):
    features_importance = pd.DataFrame(sorted(zip(features_importance, features_names)),
                                       columns=['Value', 'Feature'])

    plt.figure(figsize=(20, 10))
    g = sns.barplot(x="Feature", y="Value", data=features_importance.sort_values(by="Value", ascending=False))
    for item in g.get_xticklabels():
        item.set_rotation(45)
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(path)

class LGBMClassifier:
    def __init__(self, config, writer, eval_set: list):
        self.eval_set = eval_set
        self.feature_importances_ = None

        self.model = lgb.LGBMClassifier(**lgb_classifier_params, n_estimators=20000, n_jobs=-1)

    def fit(self, examples, lenghts, sepsis):
        examples = pd.concat(examples)
        y_train = examples['SepsisLabel'].values
        x_train = examples.drop(['SepsisLabel'], axis=1).values

        eval_set = pd.concat(self.eval_set[0][0])
        y_eval = eval_set['SepsisLabel'].values
        x_eval = eval_set.drop(['SepsisLabel'], axis=1).values

        self.model.fit(x_train, y_train, sample_weight=None, eval_set=[(x_eval, y_eval)], verbose=50,)
                       # eval_metric=eval_metric)
        self.feature_importances_ = self.model.feature_importances_

    def predict(self, examples, num_iteration=None, search_thr=False):
        references = []
        predictions = []
        probas = []
        for example in examples:
            reference = example['SepsisLabel'].values
            x = example.drop(['SepsisLabel'], axis=1).values

            if num_iteration is None:
                num_iteration = self.model.best_iteration_
            proba = self.model.predict_proba(x, num_iteration=num_iteration)[:, 1]

            references.append(reference)
            probas.append(proba)

        if search_thr:
            search_result = threshold_search(references, probas)
            thr = search_result['threshold']
        else:
            thr = 0.5
        logging.info("thr: {}".format(thr))
        print(thr)

        for proba in probas:
            prediction = np.where(proba > thr, 1, 0)
            predictions.append(prediction)

        return predictions, references


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    print(threshold_search)
    for threshold in tqdm.tqdm([i * 0.05 for i in range(20)]):
        score, _, _ = normalized_utility_score(y_true, [y > threshold for y in y_proba])
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result

