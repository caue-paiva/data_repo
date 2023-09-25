import matplotlib as plt
import pandas as pd
import time
import seaborn as sns

def Set_Sex_binary(df):  #0 is for males and 1 for females 
   Bina_sex = lambda x : int(x == 'F')  
   df['Sex'] =  df['Sex'].apply(Bina_sex)
   return df

def treat_save_data(df):
   df = Set_Sex_binary(df)
   if 'Season' in df.columns:
    df = df.drop(columns='Season')
   df = games_binary(df)
   df = Nan_to_str(df)
   df.to_csv('kaggle_compe\olym_treated2_train.csv', index=False)

def compare_colums(df, MainCol, MinorCol):
   mismatches = 0
   for i in df.index:
      auxMain =   str(df.loc[i, MainCol]).lower()
      auxMinor =  str(df.loc[i, MinorCol]).lower()
      if auxMinor not in auxMain: mismatches+=1
   
   return mismatches

def games_binary(df): #summer = 1 and winter = 0
    game_to_bina= lambda x: int(('summer' in x.lower()))  
    df['Games'] = df['Games'].apply(game_to_bina)
    return df

def Nan_to_str(df):
    df['Medal'].fillna(' ', inplace = True)

#df = pd.read_csv('kaggle_compe\olymp_train.csv')
df = pd.read_csv("kaggle_compe\olym_treated3_train.csv")
#df['Medal'].fillna(' ', inplace = True)
#print(df.isna().sum())





#df = df.drop(columns='Season')

#print(type(df.index))
print(df.head().columns)

for col in df.columns: 
  print( col, "//example ",df[col][0]," // data type  " , type(df[col][0])) 

         #0 Na in games and in years

print(type(df['Medal'][0]))
#treat_save_data(df)

df2 = df.dropna()
#print(df2['Games'].head())

print(df2.isna().sum())

df2.to_csv('kaggle_compe\olym_treated_NoNaN_train.csv', index=False) #most normalized/treated data
#print(df2.head())



    


