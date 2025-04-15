import pandas as pd  

df = pd.read_csv("gym_recommendation.csv")  
print(df["Level"].unique())  # Check all unique levels in the dataset
