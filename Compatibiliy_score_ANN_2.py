import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

#importing data
X = pd.read_csv("Data_Compatibility_score.csv", sep=';')
y = pd.read_csv("Output_Compatibility_score.csv", sep=';')

#spliting datas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


