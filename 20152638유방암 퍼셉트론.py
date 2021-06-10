from sklearn.linear_model import Perceptron
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

cancer = load_breast_cancer()

# 훈련/테스트 세트로 나누기  20%는 test할때 사용
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target, test_size=0.2)


p = Perceptron(max_iter = 100, eta0= 0.2) #데이터 100번 훑기 학습률은 0.2
p.fit(X_train, y_train)


# y값의 예측값이 들어간다
res = p.predict(X_test)

# 혼동함수 만들기
conf = np.zeros((10,10))
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1
print(conf)


# 합산하여 correct에 넣기
correct=0
for i in range(10):
    correct += conf[i][i]
accuracy = correct/len(res)

print("훈련 세트 정확도 : {:.2f}".format(p.score(X_train,y_train)))
print("테스트 세트 정확도 : {:.2f}".format(p.score(X_test,y_test)))

