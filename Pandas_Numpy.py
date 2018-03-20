import pandas as pd
import matplotlib.pyplot as plt

path = r"C:\Users\10154861\Desktop\Work\1. Analytics\A. Data\4_yrsData_IPS_Detect_iCMS\Appl_VM_Urbas_Holz_Biomasse\D1_P003_MSC_065140\drive_side\6.csv"
df = pd.read_csv(path)

plt.plot(df['3852'])
# plt.plot(df['3852'][0:10000])

plt.axhline(0.55, color = 'red')
plt.title("CF_32 plot")

plt.show()

