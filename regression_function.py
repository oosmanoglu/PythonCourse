
dataset = pd.read_csv('hw2.csv')

print(dataset)
data=dataset.dropna()
print(data)

X = data.iloc[:, 2].copy()
print(X)

y = data.iloc[:, 1].copy()
print(y)

X= np.array(X)
print(X)

y = np.array(y)
print(y)

X.shape
X.shape = ((6,4))

y.shape
y.shape = ((6,4))

b = np.linalg.inv(X.T @ X)@ X.T@ y

print(b)

yhat = np.array(X @ b)

e = yhat - y

sigmasq = e.T.dot(e) // (22) # (e'e)/(n-k-1) formula

varb = np.linalg.inv(X.T @ X) @ sigma-sq # sigmasq * (X'X)^-1 formula

plt.scatter(X,yhat, color = 'black')
plt.xlabel('laborforce_basic_educ')
plt.ylabel ('gdppc_ppp')
plt.show()




