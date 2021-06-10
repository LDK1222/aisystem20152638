from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()


# 훈련/테스트 세트로 나누기  20%는 test할때 사용
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target, test_size=0.2)

# 훈련 세트에서의 특성별 최소값 계산하기

min_on_training = X_train.min(axis=0)

# 특성별 (최대-최소) 범위 계산

range_on_traning = (X_train-min_on_training).max(axis=0)

# 훈련 데이터의 최솟값을 빼고 범위로 나누기

X_train_scaled = (X_train-min_on_training)/range_on_traning

# 테스트 세트에서도 위의 범위를 이용하여 계산한다.

X_test_scaled = (X_test-min_on_training)/range_on_traning

svc = SVC(C=1000) # SVC는 분류를 해주는 머신러닝 기법 svc에 빈모델을 만듬

svc.fit(X_train_scaled,y_train)

print("훈련 세트 정확도 : {:.2f}".format(svc.score(X_train_scaled,y_train)))
print("테스트 세트 정확도 : {:.2f}".format(svc.score(X_test_scaled,y_test)))
