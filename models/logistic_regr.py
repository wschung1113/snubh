import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Logistic Regression from statsmodel
# 로지스틱 회귀분석 사용
# disp = 0을 지울 경우, 모델의 학습을 볼 수 있음
train_features = list(X_train.columns.values)
X_opt = list(range(len(train_features)))
#pop reference to avoid matrix non-singularity
######################################################################## 성별 바꿀때마다 바꿔주기
# 여성
pop_features = ['BD1_11_1.0', 'BD2_1_1.0', 'BS3_1_1.0', 'HE_STRfh_0', 'HE_IHDfh_0', 'HE_HPfh_0', 'HE_HLfh_0', 'HE_DMfh_0', 'BE3_31_1.0', 'BE5_1_1.0', 'marri_1_1.0', 'house_1.0', 'edu_1.0', 'region_1.0', 'town_t_1.0', 'HE_mens_0.0']
# 남성
# pop_features = ['BD1_11_1.0', 'BD2_1_1.0', 'BS3_1_1.0', 'HE_STRfh_0', 'HE_IHDfh_0', 'HE_HPfh_0', 'HE_HLfh_0', 'HE_DMfh_0', 'BE3_31_1.0', 'BE5_1_1.0', 'marri_1_1.0', 'house_1.0', 'edu_1.0', 'region_1.0', 'town_t_1.0']
for f in pop_features:
    X_opt.remove(train_features.index(f))
regressor = sm.Logit(Y_train, X_train.iloc[:, X_opt]).fit()
while (np.max(regressor.pvalues) > 0.05):
    # p-value가 0.05보다 큰 항목이 있으면 가장 큰 항목부터 backward elimination 시행
    print(X_train.columns[X_opt.pop(np.argmax(regressor.pvalues))])
    regressor = sm.Logit(Y_train, X_train.iloc[:,X_opt]).fit()

# regressor = sm.Logit(Y_train, X_train).fit()

regressor.summary()

Y_pred = regressor.predict(X_val.iloc[:,X_opt])

# Logistic Regression 결과 도출 및 AUROC, AUPRC 그래프
fpr, tpr, thresholds = metrics.roc_curve(Y_val, Y_pred, pos_label=1)
prec, reca, _ = metrics.precision_recall_curve(Y_val, Y_pred)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = %0.2f)' % (metrics.auc(fpr, tpr)))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.legend(loc='lower right')
plt.title(disease + ' LogReg ROC curve')
plt.show()
plt.figure(figsize=(8,8))
plt.step(reca, prec, label='AUPRC = %0.2f' % (metrics.average_precision_score(Y_val, Y_pred)))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.legend(loc='lower right')
plt.title(disease + ' LogReg PRC curve')
plt.show()

# get importance
importance = regressor.params.values
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()