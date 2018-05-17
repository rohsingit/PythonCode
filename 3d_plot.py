from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

columns = ['A', 'B', 'C']
df_names = ['sale', 'people', 'department']
df = [pd.DataFrame([[20,30,10], [30,20,20], [20,40,40]], columns=columns),
      pd.DataFrame([[2,3,1], [3,2,2], [2,4,4]], columns=columns),
      pd.DataFrame([[1,2,1], [1,1,2], [2,1,1]], columns=columns)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#make sure x and y axis get the right tick labels
plt.xticks([i for i in range(len(columns))], columns)
plt.yticks([i for i in range(len(df_names))], df_names)

#define a list for x positions
xs = list()
for i in range(len(df)):
    for j in range(len(columns)):
         xs.append(i + j * 0.1)

for c1, c in enumerate(['r', 'g', 'b']):
    ys = list()
    for i in range(len(columns)):
        ys.extend(df[c1].ix[:,i:i+1].unstack().tolist())
    cs = [c] * len(xs)
    ax.bar(xs, ys, zs=c1, zdir='y', color=cs, alpha=0.5, width=0.1)

plt.show()