import tensorflow as tf
from tensorflow import keras
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
# from keras.callbacks import EarlyStopping
# from keras.callbacks import ModelCheckpoint

## MLP 모델 구현
# tensorflow의 keras form을 사용하여 모델 구현
# 각 layer들은 임의로 지정한 값
# 모델의 수정 및 레이어 추가, 수정 가능
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
             loss=keras.losses.BinaryCrossentropy(),
             metrics=[keras.metrics.AUC()])

callback=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=2, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False
)

# 모델 training
history = model.fit(
X_train.values,
Y_train.values,
batch_size=64,
epochs=20,
validation_data=(X_val, Y_val),
callbacks=[callback],
# verbose = 1을 하면 트레이닝 내용을 볼 수 있음
verbose=1)

#MLP 모델 결과 도출 및 AUROC, AUPRC 그래프
Y_pred3 = model.predict(X_val.values)

# plot AUROC & AUPRC
fpr, tpr, thresholds = metrics.roc_curve(Y_val, Y_pred3, pos_label=1)
prec, reca, _ = metrics.precision_recall_curve(Y_val, Y_pred3)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = %0.3f)' % (metrics.auc(fpr, tpr)))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.legend(loc='lower right')
plt.title(disease + ' MLP ROC curve')
plt.show()
plt.figure(figsize=(8,8))
plt.step(reca, prec, label='AUPRC = %0.3f' % (metrics.average_precision_score(Y_val, Y_pred3)))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.legend(loc='lower right')
plt.title(disease + ' MLP PRC curve')
plt.show()

# classification report
# Y_pred3_hard = []
# for i in range(len(Y_pred3)):
#     if Y_pred3[i] < 0.5:
#         Y_pred3_hard.append(0)
#     else:
#         Y_pred3_hard.append(1)
# print(metrics.classification_report(Y_val, Y_pred3_hard))