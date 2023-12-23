#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sellmeier_equation(wavelength,b1,b2,b3,c1,c2,c3):
    square_n = 1+b1*wavelength**2/(wavelength**2-c1)+b2*wavelength**2/(wavelength**2-c2)+b3*wavelength**2/(wavelength**2-c3)
    return square_n

def original_function(wavelength,a,b,c):
    w = 1000*wavelength
    return (a+b/w+c/w**3.5)**2

SF69 = [1.62594647,0.235927609,1.674346230,0.01216966770,0.0600710405,145.6519080]
NC17 = [1.85307378, 0.000139589850,1666.40582,0.0145009144,0.0613202273,47378.6097]
x_data = np.linspace(0.2, 1.2, 101)
y_data = original_function(x_data,1.6587589,27.1205,20077301)
y_SF69 = sellmeier_equation(x_data, *SF69)

params, covariance = curve_fit(sellmeier_equation, x_data, y_data,maxfev = 10000, p0=SF69)
print(params)
y_fit = sellmeier_equation(x_data, *params)


plt.scatter(x_data, y_data)
plt.plot(x_data, y_fit, color='red')
#plt.plot(x_data, y_SF69, color='b')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %%
