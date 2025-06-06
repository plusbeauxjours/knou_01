import numpy as np
import matplotlib.pyplot as plt

# (Optional) 난수 생성의 재현성을 위해 시드(seed)를 설정합니다.
np.random.seed(0)

# 클래스 1의 평균 벡터 μ1 = [0, 0], 클래스 2의 평균 벡터 μ2 = [0, 5]를 정의합니다.
mean1 = [0, 0]
mean2 = [0, 5]

# 두 클래스에 공통으로 사용할 공분산 행렬 Σ를 정의합니다. (Σ = [[10, 2], [2, 1]])
cov = [[10, 2], [2, 1]]

# numpy의 다변량 정규분포 함수(multivariate_normal)를 이용하여 각 클래스의 샘플 데이터를 생성합니다.
# class1_samples: 클래스 1에 대한 100개의 2차원 표본 데이터
class1_samples = np.random.multivariate_normal(mean1, cov, 100)
# class2_samples: 클래스 2에 대한 100개의 2차원 표본 데이터
class2_samples = np.random.multivariate_normal(mean2, cov, 100)

# 산점도를 그리기 위해 새로운 그림(figure)을 생성합니다.
plt.figure(figsize=(6, 6))  # 그림 크기를 6x6 인치로 설정

# 클래스 1 데이터 산점도: 파란색 별표('*') 마커로 표시하고 레이블을 'Class 1'로 지정
plt.scatter(class1_samples[:, 0], class1_samples[:, 1], 
            color='blue', marker='*', label='Class 1')

# 클래스 2 데이터 산점도: 빨간색 원형('o') 마커로 표시하되, facecolors='none'로 설정하여 내부는 채우지 않고 테두리만 표시
plt.scatter(class2_samples[:, 0], class2_samples[:, 1], 
            facecolors='none', edgecolors='red', marker='o', label='Class 2')

# 그래프의 x축과 y축 범위를 고정합니다 (x축: -10 ~ 10, y축: -5 ~ 10).
plt.xlim(-10, 10)
plt.ylim(-5, 10)

# 그래프 제목과 범례를 추가합니다.
plt.title('Sample Data')            # 상단에 제목 'Sample Data' 추가
plt.legend(loc='upper right')       # 우측 상단에 범례(클래스 1과 클래스 2) 추가

# 완성된 산점도를 화면에 표시합니다.
plt.show()