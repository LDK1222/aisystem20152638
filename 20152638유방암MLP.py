from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


cancer = load_breast_cancer()

# 훈련/테스트 세트로 나누기  20%는 test할때 사용
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target, test_size=0.2)

# 난수 초기값 고정
mlp = MLPClassifier(random_state=42)

mlp.fit(X_train,y_train)

print("훈련 세트 정확도 : {:.2f}".format(mlp.score(X_train,y_train)))
print("테스트 세트 정확도 : {:.2f}".format(mlp.score(X_test,y_test)))