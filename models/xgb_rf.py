##XGB RandomForest
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import graphviz

data_dmatrix = xgb.DMatrix(data=X_train, label=Y_train)
#XGB RandomForest의 경우 parameter들을 params 라는 dict형태로 저장
# learning_rate나 num_parallel_tree 등의 hyperparameter들은 XGB RandomForest API 문서 참조
# API link: https://xgboost.readthedocs.io/en/latest/tutorials/rf.html
params = {
  'colsample_bynode':1,
  'learning_rate':1,
  'max_depth':5,
  'num_parallel_tree':200,
  'objective':'binary:logistic',
  'subsample':0.5,
  'eval_metric':'logloss'
  # , 'tree_method': 'gpu_hist'
}
final_round = 5
bst = xgb.train(params, data_dmatrix, num_boost_round=final_round)

# XGB Random Forest 결과 도출 및 AUROC, AUPRC 그래프
dmatrix_val = xgb.DMatrix(data = X_val, label=Y_val)
Y_pred2 = bst.predict(dmatrix_val)
fpr, tpr, thresholds = metrics.roc_curve(Y_val, Y_pred2, pos_label=1)
prec, reca, _ = metrics.precision_recall_curve(Y_val, Y_pred2)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = %0.3f)' % (metrics.auc(fpr, tpr)))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.legend(loc='lower right')
plt.title(disease + ' XGB RandomForest ROC curve')
plt.show()
plt.figure(figsize=(8,8))
plt.step(reca, prec, label='AUPRC = %0.3f' % (metrics.average_precision_score(Y_val, Y_pred2)))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.legend(loc='lower right')
plt.title(disease + ' XGB RandomForest PRC curve')

plt.show()

# tree visualization
xgb.plot_tree(bst, num_trees=5)
plt.rcParams['figure.figsize'] = [40, 15]
plt.show()

# feature importance
# 한 feature가 split 되는데 몇번이나 기여했는지
xgb.plot_importance(bst)
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()