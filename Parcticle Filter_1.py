import numpy as np
import scipy as scipy
import threading
from numpy.random import uniform
import scipy.stats

np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)
import cv2


def drawLines(img, points, r, g, b):
    cv2.polylines(img, [np.int32(points)], isClosed=False, color=(r, g, b))


def drawCross(img, center, r, g, b):
    d = 5
    t = 2
    LINE_AA = cv2.LINE_AA
    color = (r, g, b)
    ctrx = center[0, 0]
    ctry = center[0, 1]
    cv2.line(img, (ctrx - d, ctry - d), (ctrx + d, ctry + d), color, t, LINE_AA)
    cv2.line(img, (ctrx + d, ctry - d), (ctrx - d, ctry + d), color, t, LINE_AA)

def mouseCallback(event, x, y, flags, null):
    global center#中心点
    global trajectory#存储轨迹
    global previous_x#上一次的中心点x坐标
    global previous_y#上一次的中心点y坐标
    global zs
    center = np.array([[x, y]])
    trajectory = np.vstack((trajectory, np.array([x, y])))#新中心点添加入新的轨迹上，按列存储
    # noise=sensorSigma * np.random.randn(1,2) + sensorMu

    if previous_x > 0:
        heading = np.arctan2(np.array([y - previous_y]), np.array([previous_x - x]))#arctan2的作用：按照x2，x1所在象限返回x1/x2的正切值,求得角度

        if heading > 0:
            heading = -(heading - np.pi)
        else:
            heading = -(np.pi + heading)

        distance = np.linalg.norm(np.array([[previous_x, previous_y]]) - np.array([[x, y]]), axis=1)#distance为上一点与下一点的长度，函数意义为求取向量范数

        std = np.array([2, 4])#std为？
        u = np.array([heading, distance])#求heading的意义，求角度
        predict(particles, u, std, dt=1.)
        zs = (np.linalg.norm(landmarks - center, axis=1) + (np.random.randn(NL) * sensor_std_err))#中心点到各landmarks的距离+误差率 randn正态分布随机数
        update(particles, weights, z=zs, R=50, landmarks=landmarks)

        indexes = systematic_resample(weights)#重采样 收集权重大的例子同时保留部分权重小的粒子
        resample_from_index(particles, weights, indexes)

    previous_x = x
    previous_y = y


WIDTH = 800
HEIGHT = 600
WINDOW_NAME = "Particle Filter"

# sensorMu=0
# sensorSigma=3

sensor_std_err = 5#传感误差？


def create_uniform_particles(x_range, y_range, N):
    particles = np.empty((N, 2))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    return particles


def predict(particles, u, std, dt=1.):#更新particles
    N = len(particles)
    dist = (u[1] * dt) + (np.random.randn(N) * std[1])#=distance*1+随机数*4
    particles[:, 0] += np.cos(u[0]) * dist#+=cos(heading)*dist
    particles[:, 1] += np.sin(u[0]) * dist


def update(particles, weights, z, R, landmarks):
    weights.fill(1.)#权重置1
    for i, landmark in enumerate(landmarks):
        distance = np.power((particles[:, 0] - landmark[0]) ** 2 + (particles[:, 1] - landmark[1]) ** 2, 0.5)#求距离
        weights *= scipy.stats.norm(distance, R).pdf(z[i])#正态分布的概率密度函数,distance期望，R标准差，pdf(z[i])为概率密度函数


    weights += 1.e-300  # avoid round-off to zero
    weights /= sum(weights)


def neff(weights):
    return 1. / np.sum(np.square(weights))


def systematic_resample(weights):#重取样
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N#arange等差数列+随机数

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)#累加权重得到新数组
    i, j = 0, 0
    while i < N and j < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes#得到一个整数数组（可用权重


def estimate(particles, weights):
    pos = particles[:, 0:1]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean) ** 2, weights=weights, axis=0)
    return mean, var


def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)#更新权重为百分比权重

def landrandom():
    global landmarks
    global t
    landmarks = np.array(
        [[np.random.randint(0, 800), np.random.randint(0, 600)], [np.random.randint(0, 800), np.random.randint(0, 600)],
         [np.random.randint(0, 800), np.random.randint(0, 600)], [np.random.randint(0, 800), np.random.randint(0, 600)],
         [np.random.randint(0, 800), np.random.randint(0, 600)],
         [np.random.randint(0, 800), np.random.randint(0, 600)]])
    t=threading.Timer(5.0,landrandom)
    t.start()

x_range = np.array([0, 800])
y_range = np.array([0, 600])

# Number of partciles
N = 400
landmarks=np.array([ [144,73], [410,13], [336,175], [718,159], [178,484], [665,464]  ])
NL = len(landmarks)
t = threading.Timer(5.0, landrandom)
t.start()
particles = create_uniform_particles(x_range, y_range, N)

weights = np.array([1.0] * N)

# Create a black image, a window and bind the function to window
img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouseCallback)

center = np.array([[-10, -10]])

trajectory = np.zeros(shape=(0, 2))
robot_pos = np.zeros(shape=(0, 2))
previous_x = -1
previous_y = -1
DELAY_MSEC = 50
count_landmarks = 0

while (1):

    cv2.imshow(WINDOW_NAME, img)
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    drawLines(img, trajectory, 0, 255, 0)
    drawCross(img, center, r=255, g=0, b=0)

    # landmarks
    for landmark in landmarks:
        cv2.circle(img, tuple(landmark), 10, (255, 0, 0), -1)

    # draw_particles:
    for particle in particles:
        cv2.circle(img, tuple((int(particle[0]), int(particle[1]))), 1, (255, 255, 255), -1)

    if cv2.waitKey(DELAY_MSEC) & 0xFF == 27:
        break

    cv2.circle(img, (10, 10), 10, (255, 0, 0), -1)
    cv2.circle(img, (10, 30), 3, (255, 255, 255), -1) 
    drawLines(img, np.array([[10, 55], [25, 55]]), 0, 255, 0)
    cv2.putText(img, "Landmarks", (30, 20), 1, 1.0, (255, 0, 0))
    cv2.putText(img, "Local:"+str(landmarks), (120, 20), 1, 1.0, (255, 255, 255))
    cv2.putText(img, "Particles", (30, 40), 1, 1.0, (255, 255, 255))
    cv2.putText(img, "Robot Trajectory(Ground truth)", (30, 60), 1, 1.0, (0, 255, 0))

cv2.destroyAllWindows()