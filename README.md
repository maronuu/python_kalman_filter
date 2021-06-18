# python_kalman_filter
## What is this
Basic Kalman filter algorithms implemented in Python, including 
1) Linear Kalman Filter
2) Extended Kalman Filter
3) Unscented Kalman Filter
It shows derivatives of the Kalman filter can well estimate the states of a system.
Note that time efficiency is not considered so much.


## Examples

### Linear Kalman Filter
```
A = np.array([
        [1.0]
    ])
b = np.array([
    [1.5,],
])
c = np.array([
    [0.36,],
])
Q = 100.0
R = 1.0
xhat_0 = np.array([
    [0.0,],
])
gamma = 0.6879987098
P_0 = np.identity(1) * gamma
N = 50
```
![image](https://user-images.githubusercontent.com/63549742/122604180-8048ac80-d0b0-11eb-9d7a-13300cf7c4e9.png)

### Extended Kalman Filter
```
f = lambda x: 0.2*x + 25*x/(1+x**2) + 10*np.cos(x/10) + 0.01 * np.exp(-x)
h = lambda x: 1/20*(x**2)
b = np.array([[1.0]])
N = 50
Q = np.array([[1.0]])
R = np.array([[3.0]])
xhat_0 = np.array([[0.0]])
gamma = 10.0
P_0 = gamma * np.identity(1)
```
![image](https://user-images.githubusercontent.com/63549742/122604196-85a5f700-d0b0-11eb-9c08-2bcfdad0c7c0.png)

### Unscented Kalman Filter
```
f = lambda x: 0.2*x + 25*x/(1+x**2) + 10*np.cos(x/10) + 0.01 * np.exp(-x)
h = lambda x: 1/20*(x**2)
b = np.array([[1.0]])
N = 50
Q = np.array([[1.0]])
R = np.array([[3.0]])
xhat_0 = np.array([[0.0]])
gamma = 10.0
P_0 = gamma * np.identity(1)
```
![image](https://user-images.githubusercontent.com/63549742/122604201-89397e00-d0b0-11eb-9199-a385212bace3.png)
