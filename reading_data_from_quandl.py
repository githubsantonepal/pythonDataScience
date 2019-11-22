import pandas as pd
import quandl
df=quandl.get("BATS/EDGA_GOOGL", authtoken="m1nUYTYPUCPt-FJBAQvx")
print(df.head())
