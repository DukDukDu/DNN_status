import pandas as pd


# file_path = '/home/olympus/MingxuanZhang/fatjet/output/123/apply03/out_fjmm.csv'  
# df = pd.read_csv(file_path)

# print(df.columns.tolist())

df = pd.read_csv('/home/olympus/MingxuanZhang/fatjet/high/train123_h.csv') 

column_name = 'is_vhmm'  
# print(df[column_name].to_list())  
 
j = 0

for value in df[column_name]:
    if value == False:
        j = j+1
print(j)