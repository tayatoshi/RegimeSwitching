import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import regimeswitching as resim
# import rs as resim
length = 200
trance = np.array([[0.85,0.15],
                   [0.2,0.8]])
tr = np.zeros(length)
y = np.zeros(length)
for i in range(1,length):
    if tr[i-1] == 0:
        tr[i]=np.random.binomial(1,trance[0,1])
    if tr[i-1] == 1:
        tr[i]=np.random.binomial(1,trance[1,1])
c = tr*30 + 30
y[0]=30
for i in range(1,length):
    y[i] = c[i]  + 0.7 * (y[i-1] - c[i-1])+np.random.normal(0,5)
y=y.reshape(length,1)
a0 = np.array([30,60]).reshape(2,1)
p0 = np.ones([2,1,1])
model = resim.Regimeswitching(y=y,a0=a0,p0=p0)
result = model.fit()
for i in range(len(result['marginal_probability'])):
    result['marginal_probability'][i] = result['marginal_probability'][i]/np.sum(result['marginal_probability'][i])
# print(result["marginal_probability"])
print(result['trance'])

plt.figure(figsize=(12,9))
plt.subplot(2,1,1)
plt.plot(y,label="y")
plt.plot(result["marginal_a"][:,0],label="a0")
plt.plot(result["marginal_a"][:,1],label="a1")
plt.title("observation and state variables")
plt.legend()
plt.subplot(2,1,2)
# plt.plot(result["marginal_probability"][:,0],label="Pr(a0)")
plt.plot(result["marginal_probability"][:,1],label="Pr(a1)")
# plt.title("Pr(a_1|omega_t)")
plt.legend()
plt.show()

