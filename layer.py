import matplotlib.pyplot as plt


M1=[90.63, 97.59]
M2=[91.05, 98.05]
M3=[90.47, 97.16]
# M1=[90.63, 97.59, 93.43]
# M2=[91.05, 98.05, 94.38]
# M3=[90.47, 97.16, 92.89]

# M1=[90.13, 97.46]
# M2=[91.05, 98.05]
# M3=[90.47, 97.81]
# M1=[90.13, 97.46, 92.16]
# M2=[91.05, 98.05, 94.38]
# M3=[90.47, 97.81, 93.41]


# AddNum = ['IP', 'SA', 'PU']
# totalWidth = 0.6
# barWidth = totalWidth/3
# seriesNums=3
AddNum = ['IP', 'SA']
totalWidth = 0.4
barWidth = totalWidth/2
seriesNums=2


fig = plt.figure(figsize=(6, 4), dpi=600)
# plt.grid()
# plt.bar([x-barWidth for x in range(seriesNums)], height=M1, width=barWidth, label='Learning Rate=0.01', color='darkred')
# plt.bar([x for x in range(seriesNums)], height=M2, width=barWidth, label='Learning Rate=0.001', color='darkblue')
# plt.bar([x+barWidth for x in range(seriesNums)], height=M3, width=barWidth, label='Learning Rate=0.0001', color='yellow')
plt.bar([x-barWidth for x in range(seriesNums)], height=M1, width=barWidth, label='PLGCN with 1 GCN layer', color='darkred', alpha=0.8)
plt.bar([x for x in range(seriesNums)], height=M2, width=barWidth, label='PLGCN with 2 GCN layers', color='darkblue', alpha=0.8)
plt.bar([x+barWidth for x in range(seriesNums)], height=M3, width=barWidth, label='PLGCN with 3 GCN layers', color='yellow', alpha=0.8)
# plt.xticks([x for x in range(seriesNums)], ['IP', 'SA', 'PU'], font='Times New Roman')
plt.xticks([x for x in range(seriesNums)], ['IP', 'SA'], font='Times New Roman')
plt.ylim(80, 100)
plt.yticks(font='Times New Roman')
plt.legend(loc="lower right", prop='Times New Roman')
plt.xlabel('Different HSI dataset', font='Times New Roman')
plt.ylabel('Accuracy(mean of 5 trials) in the final stage', font='Times New Roman')
# plt.title('Testing accuracy of the proposed method on PU dataset.', font='Times New Roman')

# 显示图像
plt.savefig('fig/NEW-GCNLAR-by.png')
plt.show()
print('~')