import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

##LightGB 모델

# lightgb 또한 같은 방식으로 params를 dict 형태로 저장
# objective, metric을 제외한 세팅은 임의로 지정
# Hyperparameter tuning 관련 kaggle 문서: https://www.kaggle.com/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-761
lgb_params = {
        "objective" : "binary",
        "metric" : "auc",
        "num_leaves" : 1000,
        "learning_rate" : 0.01,
        "bagging_fraction" : 0.8,
        "feature_fraction" : 0.8,
        "bagging_freq" : 5,
        "reg_alpha" : 1.728910519108444,
        "reg_lambda" : 4.9847051755586085,
        "random_state" : 42,
        "bagging_seed" : 2020,
        "verbosity" : -1,
        "max_depth": 18,
        # "boosting":"rf",
        "boosting_type":"gbdt",
        "min_child_samples":100
    }
#lgb의 Dataset 구현
lgb_train = lgb.Dataset(X_train, label=Y_train)
lgb_val = lgb.Dataset(X_val, label=Y_val)
evals_result = {}
model = lgb.train(lgb_params, lgb_train, 2500, valid_sets=[lgb_val], 
                  early_stopping_rounds=50, verbose_eval=50, evals_result=evals_result)

pred_test_y = model.predict(X_val, num_iteration=model.best_iteration)

#LightGB 모델 결과 도출 및 AUROC, AUPRC 그래프
fpr, tpr, thresholds = metrics.roc_curve(Y_val, pred_test_y, pos_label=1)
prec, reca, _ = metrics.precision_recall_curve(Y_val, pred_test_y)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = %0.3f)' % (metrics.auc(fpr, tpr)))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.legend(loc='lower right')
plt.title(disease + ' LightGB ROC curve')
plt.show()
plt.figure(figsize=(8,8))
plt.step(reca, prec, label='AUPRC = %0.3f' % (metrics.average_precision_score(Y_val, pred_test_y)))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.legend(loc='lower right')
plt.title(disease + ' LightGB PRC curve')

plt.show()