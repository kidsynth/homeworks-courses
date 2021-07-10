import math as m
import numpy as np
from numpy.linalg import inv
import seaborn as sns
from copy import deepcopy
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

# Начальные условия
sko = 1
mo = 0
alfa = 0.05
u_a = 1.96 

n = 40 
h = 4
V = 0.25
Radius = 90
omega = V/Radius

raspr = 0 # 0 - нормальное, 1 - лапласса

#	1
# Моделирование набора измерений (ОШИБКИ С НОРМАЛЬНЫМ РАСПР.)
t = np.array([(h * i) for i in range(n)])
x0 = 40
y0 = 50
x = np.array([x0 + np.cos(omega * i) *  Radius for i in t ])
y = np.array([y0 - np.sin(omega * i) *  Radius for i in t ])

plt.figure(figsize=(18, 10))
plt.plot(x, y, label = 'Траектория', color = 'black')
plt.ylabel('Y(t)')
plt.xlabel('X(t)')
plt.legend(bbox_to_anchor=(0.4, 0.5), loc='upper left', borderaxespad=0.5)
plt.show()
dalnost = (x**2 + y**2)**0.5

if (raspr == 0 ):
    #Моделирование вектора ошибок нормального распределения
    W_k = np.random.normal(0, sko, n)
else:
    #Моделирование вектора ошибок распределения лапласса
    W_k = np.array(np.random.laplace(0, sko, n))

IZM_R = dalnost + W_k

plt.figure(figsize=(15, 10))
plt.plot(t, dalnost, label = 'Дальность', color = 'black')
plt.plot(t, IZM_R, '.', label = 'Дальность из НИС', color = 'r')
plt.ylabel('Расстояние')
plt.xlabel('Время')
plt.legend(bbox_to_anchor=(0.5, 0.8), loc='upper left', borderaxespad=0.)
plt.show()

#	2
# МНК, линейная регрессия
Vector1 = np.array([sum(IZM_R), sum(t * IZM_R)])
Matrix1 = np.array([[n, sum(np.array(t))], [sum(np.array(t)), sum(np.array(t)**2)]])
Tetta1, Tetta2 = np.linalg.solve(Matrix1, Vector1)
print('Значения коэффициентов тета', Tetta1, Tetta2)
dalnost_approx = Tetta1 + Tetta2 * t

Aa = np.matrix([np.ones(n),t]).transpose() #. матрица А как у влада и ани в отчете про мнк
I = np.eye(n) # единичная матрица
sigma = np.var(W_k)

# тут считает вектор дисперсии для доверительного интервала
def dispersion_count(t_arr):
    Disp = []
    for i in t:
        h_t = np.matrix([1, i]).transpose()
        a = h_t.transpose().dot(inv(Aa.transpose().dot(Aa))).dot(Aa.transpose())
        d_temp = a.dot(sigma).dot(I).dot(a.transpose())
        Disp.append(d_temp.item(0))
    Disp = np.array(Disp)
    return Disp

Disp = dispersion_count(t)

dover_min = dalnost_approx - ur_znach * Disp**0.5
dover_max = dalnost_approx + ur_znach * Disp**0.5

plt.figure(figsize=(15, 10))
plt.plot(t, dalnost, label = 'Дальность (истинное знач.)', color = 'black')
plt.plot(t, IZM_R, '.', label = 'Дальность из НИС', color = 'r')
plt.plot(t, dalnost_approx, label = 'Дальность, оцененная')
plt.plot(t, dover_min, 'y', label = 'Дов. границы')
plt.plot(t, dover_max, 'y')
plt.ylabel('Расстояние')
plt.xlabel('Время')
plt.legend(bbox_to_anchor=(0.5, 0.8), loc='upper left', borderaxespad=0.)
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(t, Disp, label = 'Диспресия оценки дальности', color = 'black')
plt.legend(bbox_to_anchor=(0.5, 0.8), loc='upper left', borderaxespad=0.)
plt.show()

#	3
# МНК, параболическая регрессия
Matrix2 = np.array([[n, sum(t), sum(t**2)], \
             [sum(t), sum(t**2), sum(t**3)], \
             [sum(t**2), sum(t**3), sum (t**4)]])
Vector2 = np.array([sum(IZM_R), sum(IZM_R * t), sum(IZM_R * t**2)])
Tetta_1, Tetta_2, Tetta_3 = np.linalg.solve(Matrix2, Vector2)
print('Значения коэффициентов тета', Tetta_1, Tetta_2, Tetta_3)
dalnost_approx_1 = Tetta_1 + Tetta_2 * t + Tetta_3 * t**2 / 2

Aa = np.matrix([np.ones(n),t, t**2/2]).transpose() #. матрица А как у влада и ани в отчете про мнк
I = np.eye(n) # единичная матрица
sigma = np.var(W_k)

# тут считает вектор дисперсии для доверительного интервала
def dispersion_count(t_arr):
    Disp = []
    for i in t:
        h_t = np.matrix([1, i, i**2/2]).transpose()
        a = h_t.transpose().dot(inv(Aa.transpose().dot(Aa))).dot(Aa.transpose())
        d_temp = a.dot(sigma).dot(I).dot(a.transpose())
        Disp.append(d_temp.item(0))
    Disp = np.array(Disp)
    return Disp
Disp_1 = dispersion_count(t)

dover_min_1 = dalnost_approx_1 - ur_znach * Disp_1**0.5
dover_max_1 = dalnost_approx_1 + ur_znach * Disp_1**0.5

plt.figure(figsize=(15, 10))
plt.plot(t, dalnost, label = 'Дальность (истинное знач.)', color = 'black')
plt.plot(t, IZM_R, '.', label = 'Дальность из НИС', color = 'r')
plt.plot(t, dalnost_approx_1, label = 'Дальность, оцененная')
plt.plot(t, dover_min_1, 'y', label = 'Дов. границы')
plt.plot(t, dover_max_1, 'y')
plt.ylabel('Расстояние')
plt.xlabel('Время')
plt.legend(bbox_to_anchor=(0.5, 0.8), loc='upper left', borderaxespad=0.)
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(t, Disp_1, label = 'Диспресия оценки дальности', color = 'black')
plt.legend(bbox_to_anchor=(0.5, 0.8), loc='upper left', borderaxespad=0.)
plt.show()

#	4
# Проверка гипотезы о равенстве нулю коэффийиента тета
H3def = inv(Aa.transpose().dot(Aa))
Ktheta = 1 * H3def
Dispers = Ktheta[2, 2]
T_ = Tetta_3 / Dispers**0.5
print(T_)
Ul = -1.95996 # границы доверительного интервала
Ur = - Ul # границы доверительного интервала

plt.figure(figsize=(10, 7))
plt.plot([Ul, Ur], [0,0], label = 'Границы принятия гипотезы', color='black')
plt.plot(T_, 0, 'o', color='red',label = 'Значение статистики t' )
plt.legend(bbox_to_anchor=(0.5, 0.8), loc='upper left', borderaxespad=0.)
plt.show()

#	5
# Распределение вектора остатков
Residual = dalnost_approx_1 - IZM_R
rr = deepcopy(Residual) 
rr = sorted(rr)
left = min(rr)
right = max(rr)
bins_1 = [i for i in np.linspace(left, right, 6)]
def gaussian(x, mu, sigma):
     return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- 0.5 * (x - mu)**2 / sigma**0.5)

sns_plot = sns.distplot(Residual, bins=bins_1, kde=False, norm_hist=True, label = 'Распределение остатков')
fig = sns_plot.get_figure()

xs = np.arange(left, right, 0.001)
p2, = plt.plot(xs, [gaussian(x, 0, 1) for x in xs], label='$\sigma = 1$', color = 'black')
plt.legend(bbox_to_anchor=(0.7, 0.9), loc='upper left', borderaxespad=0.)
plt.show()


#	6
# Проверка гипотезы о распределении вектора остатков
import scipy as scp
from scipy.stats import chisquare
import scipy.stats as stats
D_n = 0.895

data = Residual #np.random.normal(mo, sko, n)
normed_data = (data - mo) / sko
Result = stats.kstest(normed_data,'norm')
print('Значение статистики:', Result.statistic, ',', 'Значение уровня значимости:', Result.pvalue)
if Result.statistic < Result.pvalue:
    print(Result.statistic, '<', Result.pvalue)
    print('Гипотеза H0 принимается')
else:
    print(Result.statistic, '>', Result.pvalue)
    print('Гипотеза H0 отклоняется')

plt.figure(figsize=(10, 7))
plt.plot([0, Result.pvalue], [0,0], label = 'Интервал принятия гипотезы', color='black')
plt.plot(Result.statistic, 0, 'o', color='red', label = 'Значение статистики')
plt.legend(bbox_to_anchor=(0.6, 0.9), loc='upper left', borderaxespad=0.)
plt.show()
