import pandas as pd
import heapq
import matplotlib.pyplot as plt
df=pd.read_csv('x.csv')
#print(df)
N=2
print(pd.DataFrame([df[V32].nlargest(N).values.tolist() for V32 in df.columns], index=df.columns, columns=['{}_largest'.format(i) for i in range(1, N+1)]).T)

print((pd.DataFrame([df[V32].nlargest(N).values.tolist() for V32 in df.columns], index=df.columns, columns=['{}_largest'.format(i) for i in range(1, N+1)]).T).transpose)
