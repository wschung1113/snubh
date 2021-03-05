import xgboost as xgb

## XGB Regression
# XGB 라이브러리는 data와 label이 같이 적혀있는 DMatrix(Data Matrix)를 이용하여 학습된다
data_dmatrix = xgb.DMatrix(data=X_train, label=Y_train)
# XGBRegressor 클래스 instance 생성
# objective 는 우리가 하려는 binary logistic regression임을 보여주고, 나머지는 hyperparameter들
xg_reg = xgb.XGBRegressor(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 100)
xg_reg.fit(X_train, Y_train)
Y_pred = xg_reg.predict(X_val)

#XGB Regression 결과 도출 및 AUROC, AUPRC 그래프
fpr, tpr, thresholds = metrics.roc_curve(Y_val, Y_pred, pos_label=1)
prec, reca, _ = metrics.precision_recall_curve(Y_val, Y_pred)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = %0.2f)' % (metrics.auc(fpr, tpr)))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.legend(loc='lower right')
plt.title(disease + ' XGBReg ROC curve')
plt.show()
plt.figure(figsize=(8,8))
plt.step(reca, prec, label='AUPRC = %0.2f' % (metrics.average_precision_score(Y_val, Y_pred)))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.legend(loc='lower right')
plt.title(disease + ' XGBReg PRC curve')

plt.show()