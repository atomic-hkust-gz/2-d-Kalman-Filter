import numpy as np
import matplotlib.pyplot as plt

# 状态方程：x = A*x + B*u + Q，其中x是状态向量，u是控制向量，Q是过程噪声
# x [position offset x, y, velocity   vx, vy]
# u [ax, ay]

# kinetic model (position)
# x0 + vx * dt = x1
# y0 + vy * dt = y1

dt = 0.1 # 时间步长
A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

# kinetic model (vecolity)
# vx0 + ax0 * dt = vx1
# vy0 + ay0 * dt = vy1

B = np.array([[0.5*dt**2, 0], [0, 0.5*dt**2], [dt, 0], [0, dt]])

# based on envrionment (like resistance)
Q = np.diag([1, 1, 0.1, 0.1]) # 过程噪声协方差矩阵


# 观测方程：z = H*x + R，其中z是观测向量，R是观测噪声
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) # assume system doesn't have noise
R = np.diag([1, 1]) # 观测噪声协方差矩阵

# 初始化状态向量和协方差矩阵
x = np.array([[0], [0], [0], [0]]) 

P = np.eye(4)

U = np.array([[5],[7]])  #加速度

T = 50  #迭代次数
# 卡尔曼滤波

X_est = np.zeros((4, T)) # estimated status
P_est = np.zeros((4, 4, T)) # the likehood of predicted status
K = np.zeros((4, 2, T)) # porpotion between predicted and measured
Z = np.zeros((2, T))    # measured status
X = np.zeros((4, T))    # ground truth status

X[:,0] = np.array([[10], [50], [10], [0]]).reshape(4,)      #初始位置

for i in range(1,T):

    # generate ground truth status
    X[:,i] = (A.dot(X[:,i-1].reshape(4,1)) + B.dot(U) + np.random.multivariate_normal([0, 0, 0, 0], Q).reshape(4,1)).reshape(4,)
    
    # generate (fake) measured status based on ground truth
    Z[:,i] = (H.dot(X[:,i-1].reshape(4,1)) + np.random.multivariate_normal([0, 0], R).reshape(2,1)).reshape(2,)

for i in range(T):
    # 预测步骤
    x_pred = A.dot(x) + B.dot(U)
    P_pred = A.dot(P).dot(A.T) + Q

    # 更新步骤
    K[:,:,i] = P_pred.dot(H.T).dot(np.linalg.inv(H.dot(P_pred).dot(H.T) + R))
    x = x_pred + K[:,:,i].dot(Z[:,i:i+1] - H.dot(x_pred))
    P = (np.eye(4) - K[:,:,i].dot(H)).dot(P_pred)

    X_est[:,i] = x.reshape(4,)
    P_est[:,:,i] = P

# 绘图
plt.figure(figsize=(10, 5))
plt.subplot(131)
# plt.plot(X[0,:], X[1,:], '-*', label='True')
# plt.plot(Z[0,:], Z[1,:], '.',           label='Observation')
# plt.plot(X_est[0,:], X_est[1,:], '-*', label='Estimate')
plt.plot([i for i in range(50)], X[1,:], '-*', label='True')
plt.plot([i for i in range(50)], Z[1,:], '.',           label='Observation')
plt.plot([i for i in range(50)], X_est[1,:], '-*', label='Estimate')
plt.legend()
plt.grid()
plt.title('Position')
plt.xlabel('x_distance/cm')
plt.ylabel('y_distance/cm')
plt.xlim(0,200)
plt.ylim(0,200)

'''
plt.subplot(132)
plt.plot(X[2,:], X[3,:], label='True')
plt.plot(X_est[2,:], X_est[3,:], label='Estimate')
plt.legend()
plt.title('Velocity')
plt.xlabel('x_vel/(cm/s)')
plt.ylabel('y_vel/(cm/s)')
'''

plt.subplot(132)
plt.plot([i for i in range(T)],X[2,:], label='True')
plt.plot([i for i in range(T)],X_est[2,:], label='Estimate')
plt.legend()
plt.title('x_Velocity')
plt.xlabel('epoch')
plt.ylabel('x_vel/(cm/s)')

plt.subplot(133)
plt.plot([i for i in range(T)],X[3,:], label='True')
plt.plot([i for i in range(T)],X_est[3,:], label='Estimate')
plt.legend()
plt.title('y_Velocity')
plt.xlabel('epoch')
plt.ylabel('y_vel/(cm/s)')

plt.show()