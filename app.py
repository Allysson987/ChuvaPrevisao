import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
       

import numpy as np
import joblib

class Analise():
    def __init__(self):
        super().__init__()
        self.df=pd.read_csv("Chuva.csv",sep=";")
      
        self.preparacao()
        self.treinar()
        self.treinar_dia()
    def preparacao(self):
        self.dados=self.df.dropna().copy()
        self.dados.columns=self.dados.columns.str.strip()
        
        print(self.dados.columns.tolist())
        
        colNum=["Chuva (mm)", "Umi. Min. (%)", "Umi. Max. (%)", 'Temp. Min. (C)', "Temp. Max. (C)", "Pressao Min. (hPa)", "Pressao Max. (hPa)", "Radiacao (KJ/m²)", 'Pto Orvalho Min. (C)', "Pto Orvalho Max. (C)"]
        for col in colNum:
           
           if self.dados[col].dtype == 'object':
              self.dados[col] = self.dados[col].str.replace(",", ".", regex=False).astype(float)
        self.dados["Data"]=pd.to_datetime(self.dados["Data"],dayfirst=True)
       
        self.dadosDia=self.dados.groupby(self.dados["Data"].dt.date).agg({
        'Chuva (mm)':'sum',
        'Umi. Min. (%)': 'min',
        'Umi. Max. (%)': 'max',
        'Temp. Min. (C)': 'min',
        'Temp. Max. (C)': 'max',
        'Pressao Min. (hPa)': 'min',
        'Pressao Max. (hPa)': 'max',
        'Radiacao (KJ/m²)':'sum',
        'Pto Orvalho Min. (C)': 'sum',
        "Pto Orvalho Max. (C)": 'sum'
        }).reset_index()
        print(self.dadosDia.head())
        print(self.dadosDia.corr(numeric_only=True)["Chuva (mm)"].sort_values(ascending=False))
        
    def treinar_dia(self):
        x_dia = self.dados["Data"].apply(lambda d: d.timetuple().tm_yday)
        x_outros = self.dados[["Umi. Min. (%)", "Umi. Max. (%)", "Pto Orvalho Min. (C)", "Pto Orvalho Max. (C)"]]
    
        x = pd.concat([x_dia.rename("Dia_Ano"), x_outros], axis=1)
        y = self.dados["Chuva (mm)"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        nome="Treino hora"
        
        self.modelos(nome, x_train, y_train,x_test,y_test)
        
    def treinar(self):
        
        x_dia = self.dadosDia["Data"].apply(lambda d: d.timetuple().tm_yday)
        x_outros = self.dadosDia[["Umi. Min. (%)", "Umi. Max. (%)", "Pto Orvalho Min. (C)", "Pto Orvalho Max. (C)"]]
    
        x = pd.concat([x_dia.rename("Dia_Ano"), x_outros], axis=1)
        y = self.dadosDia["Chuva (mm)"]
    
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        nome="Treino dia"
        
        self.modelos(nome, x_train, y_train, x_test,y_test)
        
    def modelos(self, nome, x_train, y_train, x_test, y_test):
       print(nome)
       modelos = {
        "Árvore de Decisão": DecisionTreeRegressor(
            max_depth=2,
            min_samples_leaf=2,
            random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=2,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=30,
            learning_rate=0.5,
            max_depth=5,
            random_state=42
        ),
        "AdaBoost": AdaBoostRegressor(
            n_estimators=200,
            random_state=42
        ),
        "KNN": KNeighborsRegressor(
            n_neighbors=10,
            weights='distance'
        ),
        "SVR": SVR(
            kernel='rbf',
            C=80,
            epsilon=0.5
        )
    }

       scaler = StandardScaler()
       x_train_scaled = scaler.fit_transform(x_train)
       x_test_scaled  = scaler.transform(x_test)
       for nome, modelo in modelos.items():
           print(f"\n🔎 Modelo: {nome}")

           if nome in ["KNN", "SVR"]:
                modelo.fit(x_train_scaled, y_train)
                y_pred = modelo.predict(x_test_scaled)
                
           else:
               modelo.fit(x_train, y_train)
               y_pred = modelo.predict(x_test)

           mae  = mean_absolute_error(y_test, y_pred)
           mse  = mean_squared_error(y_test, y_pred)
           rmse = np.sqrt(mse)
           r2   = r2_score(y_test, y_pred)

           print(f"MAE:  {mae:.3f}")
           print(f"MSE:  {mse:.3f}")
           print(f"RMSE: {rmse:.3f}")
           print(f"R²:   {r2:.3f}")
           if r2 >= 0.65:
               filename = f"{nome.replace(' ', '_')}.joblib"
               joblib.dump(modelo, filename)
               print(f"Modelo salvo como: {filename}")
           
Analise()