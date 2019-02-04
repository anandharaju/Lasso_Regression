from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import numpy as np

dataset = np.loadtxt("D:/00_SFU/00_Graduate_Courses/00_CMPT741_DataMining/Project/2019-741_Data/training_data_preprocessed.csv", delimiter=",")
y = dataset[:,:1]
qid = dataset[:,1]
X = dataset[:,2:]
print("\nDataset Dimensions : ",dataset.shape)

# LASSO SETUP
lasso = Lasso (alpha = 0.215,normalize=True)
lasso_coef = lasso.fit(X,y).coef_
lasso_coef_positive = lasso_coef[lasso_coef > 0]
plt.plot(range(len(lasso_coef_positive)),lasso_coef_positive)
plt.xticks(range(len(lasso_coef_positive)),range(0,58),rotation=60)
plt.ylabel('coefficients')
plt.show()

features_selected = np.where(np.array(lasso_coef) > 0)[0]
print("Features Selected [%d]:" %len(features_selected),features_selected)
X = X[:,features_selected]
print(X.shape)
