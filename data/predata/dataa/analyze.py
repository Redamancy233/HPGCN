import pandas as pd
import seaborn as sns
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np

data = scio.loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
data = np.reshape(data,( -1, data.shape[2]))
cor = np.corrcoef(data.T)
print(cor)
# 设置热力图颜色配色
plt.imshow(cor, cmap='hot', interpolation='nearest')
cb = plt.colorbar()  # 显示颜色条
# cb.ax.tick_params(fontstyle='Times New Roman')
# cb.set_label('CB', fontproperties='Times New Roman')
plt.xticks(font='Times New Roman')
plt.yticks(font='Times New Roman')
plt.tight_layout()
plt.savefig('hotfig.png', dpi=600)
plt.show()
# plt.savefig('hotfig.png')