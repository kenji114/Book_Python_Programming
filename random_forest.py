import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
header=None)

# print(df_wine)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
min_max = sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
# print('EigenValues %s' % eigen_vals)
# print('¥nEigenVecs ¥n %s' % eigen_vecs)
tot = sum(eigen_vals)
var_exp = [i/tot for i in sorted(eigen_vals, reverse=True)]

#　累積和 分散説明率のグラフを出力する。
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1,14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid',
        label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal Components')
plt.legend(loc='best')
plt.show()

# for i in range(len(eigen_vecs)):
#     print(eigen_vecs)
#     print(eigen_vals[i])
#     print(np.abs(eigen_vals[i]),eigen_vecs[:,i])

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)
# print(eigen_pairs)
print("----------------------------------------------------")
print(eigen_pairs[0][1][:])
print(eigen_pairs[1][1][:])
print("----------------------------------------------------")

w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W: \n', w)

X_train_pca = X_train_std[0].dot(w)

# ２次元に落とされた新たな軸で、特性を把握。
colors = ['r','g','b']
markers = ['s', 'x', 'o']
# for l, c, m in zip()

