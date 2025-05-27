import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
from sympy.abc import alpha
from sympy.printing.pretty.pretty_symbology import line_width


def mse_function(y, t):
    return np.sum((y - t) ** 2).mean()

t = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

y1 = [0.01, 0.03, 0.1, 0.8, 0.1, 0.1, 0.02, 0.1, 0.0, 0.01]
y2 = [0.01, 0.03, 0.1, 0.07, 0.1, 0.1, 0.02, 0.1, 0.8, 0.01]

array = np.arange(0, 1, 0.1)
"""
0 : 시작값
1 : 종료값
0.1 : step
"""

plt.plot(array, t, c='r', linestyle='--', linewidth=2, alpha=0.8)
plt.plot(array, y1, c='g', linestyle='--', linewidth=2, alpha=0.8)
plt.plot(array, y2, c='b', linestyle='--', linewidth=2, alpha=0.8)
"""
- array, t: x축, y축 데이터
- c='r': 빨간색
- linestyle='--': 점선 스타일
- linewidth=2: 선 두께
- alpha=0.8: 투명도 조절
"""
# plt.show()

# y1 의 경우 평균 제곱 오차 계산
mse1 = mse_function(np.array(y1), np.array(t))
print(mse1)
mse2 = mse_function(np.array(y2), np.array(t))
print(mse2)

def cee_function(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

mse1 = cee_function(np.array(y1), np.array(t))
print(mse1)
mse2  = cee_function(np.array(y2), np.array(t))
print(mse2)