import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def exam_plotmulticlass(Z, Y):
    # ----------------------
    # 0. 다중 클래스 산점도 함수
    # ----------------------
    # Z: (N x 2) 형식의 2차원 투영 데이터
    # Y: (N,) 형식의 클래스 레이블
    classes = np.unique(Y)
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', '*', 'x']
    colors = ['blue', 'red', 'green', 'magenta', 'cyan', 'orange', 
              'purple', 'brown', 'pink', 'gray', 'olive']
    
    for i, c in enumerate(classes):
        idx = (Y == c)
        plt.scatter(Z[idx, 0], Z[idx, 1],
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    label=f'Class {c}',
                    alpha=0.7)
    plt.legend()

def coil20_pca_lda_2d():
    # ----------------------
    # 1. COIL20 데이터 로드
    # ----------------------
    data = loadmat('HW1_COIL20.mat')
    X = data['X']            # (샘플 수 x 특징 차원)
    Y = data['Y'].ravel()    # (샘플 수,)로 변환
    
    # ----------------------
    # 2. PCA로 2차원 특징 추출
    # ----------------------
    # 평균 제거
    Xmean = np.mean(X, axis=0)
    X_centered = X - Xmean
    
    # 공분산 행렬 계산
    C = np.cov(X_centered, rowvar=False)
    
    # 고유분해 및 정렬
    eigvals, eigvecs = np.linalg.eig(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[idx]
    eigvecs_sorted = eigvecs[:, idx]
    
    # 가장 큰 고유값 2개에 해당하는 주성분(2차원)
    W_pca2 = eigvecs_sorted[:, :2]
    Z_pca2 = X_centered @ W_pca2  # (N x 2)
    
    # ----------------------
    # 3. PCA -> 95% 정보보존 후 LDA (2차원)
    # ----------------------
    # 누적 분산율 계산해 95% 넘어가는 차원 찾기
    totalVar = np.sum(eigvals_sorted)
    cumVar = np.cumsum(eigvals_sorted) / totalVar
    dim_95 = np.argmax(cumVar >= 0.95) + 1
    
    # 95% 차원까지 투영
    W_pca95 = eigvecs_sorted[:, :dim_95]
    Z_pca95 = X_centered @ W_pca95
    
    # LDA 수행 (Sw^-1 * Sb)
    nClass = int(np.max(Y))
    meanTotal_pca = np.mean(Z_pca95, axis=0)
    
    Sw = np.zeros((dim_95, dim_95))
    Sb = np.zeros((dim_95, dim_95))
    for c in range(1, nClass+1):
        Xc = Z_pca95[Y == c, :]
        mc = np.mean(Xc, axis=0)
        diffc = Xc - mc
        Sw += diffc.T @ diffc
        nc = Xc.shape[0]
        mdiff = (mc - meanTotal_pca).reshape(-1,1)
        Sb += nc * (mdiff @ mdiff.T)
    
    eigvals_lda, eigvecs_lda = np.linalg.eig(np.linalg.inv(Sw) @ Sb)
    idx_lda = np.argsort(eigvals_lda)[::-1]
    W_lda2 = eigvecs_lda[:, idx_lda[:2]]
    
    Z_lda2 = Z_pca95 @ W_lda2
    
    # ----------------------
    # 4. 시각화
    # ----------------------
    # PCA 2D 결과
    plt.figure()
    exam_plotmulticlass(Z_pca2, Y)
    plt.title('PCA (2D) Result')
    
    # PCA -> 95% -> LDA 2D 결과
    plt.figure()
    exam_plotmulticlass(Z_lda2, Y)
    plt.title('LDA (2D) Result (after PCA 95%)')
    
    plt.show()

if __name__ == '__main__':
    coil20_pca_lda_2d()