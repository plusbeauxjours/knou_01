import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ----------------------
# 1. 데이터 생성
# ----------------------
# 간단히 두 개 클래스 데이터 생성 (50개씩)
class1 = np.random.multivariate_normal([0,0], [[1,0],[0,1]], 50)
class2 = np.random.multivariate_normal([3,3], [[1,0],[0,1]], 50)

# 전체 데이터 (100 x 2)
data = np.vstack((class1, class2))

# ----------------------
# 2. PCA
# ----------------------
# 평균 제거
mean_all = data.mean(axis=0)
data_center = data - mean_all
# 공분산 행렬 & 고유분해
cov_mat = np.cov(data_center.T)
eigvals, eigvecs = np.linalg.eig(cov_mat)
# 가장 큰 고유값에 해당하는 고유벡터 (주성분)
pca_vec = eigvecs[:, np.argmax(eigvals)]

# ----------------------
# 3. LDA
# ----------------------
m1 = class1.mean(axis=0)
m2 = class2.mean(axis=0)

# Within-class scatter
S1 = ((class1 - m1).T).dot(class1 - m1)
S2 = ((class2 - m2).T).dot(class2 - m2)
Sw = S1 + S2

# LDA 방향
w_lda = np.linalg.inv(Sw).dot(m2 - m1)

# ----------------------
# 4. LDA
# ----------------------
plt.scatter(class1[:,0], class1[:,1], c='blue', label='Class1')
plt.scatter(class2[:,0], class2[:,1], c='red', label='Class2')

# PCA 벡터: 초록색 화살표
plt.arrow(0,0, pca_vec[0]*3, pca_vec[1]*3, 
          width=0.05, color='green', label='PCA1')
# LDA 벡터: 마젠타 화살표
plt.arrow(0,0, w_lda[0]*3, w_lda[1]*3, 
          width=0.05, color='magenta', label='LDA')

plt.legend()
plt.title('Simple PCA & LDA Example')
plt.show()