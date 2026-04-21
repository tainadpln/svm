from preprocessing import load_data
import numpy as np
from model import SoftSVM
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC

X_train, y_train = load_data('chest_xray\\train')
X_test, y_test = load_data('chest_xray\\test')

print(f"X train shape: {np.array(X_train).shape}, y train shape: {np.array(y_train).shape}")
print(f"X test shape: {np.array(X_test).shape}, y test shape: {np.array(y_test).shape}")

ssvm = SoftSVM(epochs=200)
ssvm.fit(X_train, y_train)
y_preds_1 = ssvm.predict(X_test)

precision = precision_score(y_test, y_preds_1, pos_label=1)
recall    = recall_score(y_test, y_preds_1, pos_label=1)
f1        = f1_score(y_test, y_preds_1, pos_label=1)

print("Kết quả của SoftSVM")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
# Precision : 0.7220
# Recall    : 0.9923
# F1-score  : 0.8359


svm = LinearSVC(C=1.0, loss='hinge', max_iter=1000)
svm.fit(X_train, y_train)
y_preds = svm.predict(X_test)

precision = precision_score(y_test, y_preds, pos_label=1)
recall    = recall_score(y_test, y_preds, pos_label=1)
f1        = f1_score(y_test, y_preds, pos_label=1)

print("Kết quả của LinearSVC của thư viện sklearn")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
# Precision : 0.7135
# Recall    : 0.9897
# F1-score  : 0.8292

# Đánh giá:
# - Precison của SoftSVM cao hơn nhiều so với của LinearSVC -> ít báo nhầm nhưng bỏ sót nhiều trường hợp bị pneumonia
# - Recall của LinearSVC cao hơn rất nhiều so với SoftSVM -> Bắt được hầu hết các trường hợp bị pneumonia nhưng chấp nhận báo nhầm
# - Về mặt tổng quan thì mô hình LinearSVC tốt hơn (F1: 0.8292 > 0.7889)