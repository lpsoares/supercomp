import numpy as np
import sys
from sklearn.linear_model import LinearRegression

np.random.seed(11)

n = 100000
c = 20

X = np.c_[np.random.random((n, c)),
          np.ones(n) ]

y = X.dot(np.random.random(c+1))

print(X.shape, y.shape, file=sys.stderr)

print(n, c+1)
for i in range(n):
    print(' '.join([str(X[i,j]) for j in range(c+1) ]))

for i in range(n):
    print(y[i])

w = np.zeros(c+1)
print(np.sum( (X.dot(w) - y)**2 ), file=sys.stderr)
lin = LinearRegression(fit_intercept=False)
lin.fit(X, y)
print(lin.coef_, file=sys.stderr)