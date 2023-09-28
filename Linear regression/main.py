import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
  We need to develop a machine learning model capable of accurately
  predicting real estate prices based on the area-price correlation
'''

area = np.array([1200, 1500, 1700, 2000, 2500, 800, 300]).reshape((-1, 1))
price = np.array([50000, 60000, 70000, 80000, 100000, 30000, 12000])

model = LinearRegression().fit(area, price)

r_sq = model.score(area, price)
print('Coefficient of determination:', r_sq)
print('Slope:', model.coef_)
print('Intercept:', model.intercept_)

area_new = np.array([1800]).reshape((-1, 1))
price_new = model.predict(area_new)
print('Predicted price:', price_new)

plt.scatter(area, price)
plt.plot(area, model.predict(area), color='red')
plt.show()
