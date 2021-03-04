##XGB RandomForest
import xgboost as xgb

#XGB RandomForest의 경우 parameter들을 params 라는 dict형태로 저장
# learning_rate나 num_parallel_tree 등의 hyperparameter들은 XGB RandomForest API 문서 참조
# API link: https://xgboost.readthedocs.io/en/latest/tutorials/rf.html
params = {
  'colsample_bynode': 0.5,
  'learning_rate': 1,
  'max_depth': 5,
  'num_parallel_tree': 200,
  'objective': 'binary:logistic',
  'subsample': 0.5
  # 'tree_method': 'gpu_hist'
}
bst = xgb.train(params, data_dmatrix, num_boost_round=1)

# XGB Random Forest 결과 도출 및 AUROC, AUPRC 그래프
dmatrix_val = xgb.DMatrix(data = X_val, label=Y_val)
Y_pred2 = bst.predict(dmatrix_val)
fpr, tpr, thresholds = metrics.roc_curve(Y_val, Y_pred2, pos_label=1)
prec, reca, _ = metrics.precision_recall_curve(Y_val, Y_pred2)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = %0.2f)' % (metrics.auc(fpr, tpr)))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.legend(loc='lower right')
plt.title(disease + ' XGB RandomForest ROC curve')
plt.show()
plt.figure(figsize=(8,8))
plt.step(reca, prec, label='AUPRC = %0.2f' % (metrics.average_precision_score(Y_val, Y_pred2)))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.legend(loc='lower right')
plt.title(disease + ' XGB RandomForest PRC curve')

plt.show()