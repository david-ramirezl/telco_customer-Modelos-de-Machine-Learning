import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado compatible con Scikit-Learn para Feature Engineering
    sobre el dataset Telco Customer Churn.
    """
    def __init__(self, drop_original=True, drop_addons=False):
        # Permite configurar dinámicamente qué variables eliminar al instanciar la clase
        self.drop_original = drop_original
        self.drop_addons = drop_addons
        
        # Servicios que se suman
        self.services = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        # Variables sin valor predictivo
        self.useless_cols = ['SeniorCitizen', 'gender', 'PhoneService', 'MultipleLines', 'customerID_Hash']
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X = X.copy()
                    
        if all(col in X.columns for col in self.services):
            # Sumar la cantidad de servicios adicionales activos
            X['Total_Internet_Addons'] = X[self.services].apply(lambda row: (row == 'Yes').sum(), axis=1)
            
        # Fricción de Pago
        if 'PaymentMethod' in X.columns:
            X['Is_AutoPay'] = X['PaymentMethod'].str.contains('automatic', case=False, na=False).astype(int)
            
        # Apoyo familiar
        if 'Partner' in X.columns and 'Dependents' in X.columns:
            X['Solo_Resident'] = ((X['Partner'] == 'No') & (X['Dependents'] == 'No')).astype(int)
            
        # Evolución de tarifa
        if 'MonthlyCharges' in X.columns and 'TotalCharges' in X.columns and 'tenure' in X.columns:
            # np.where evita la división por cero cuando tenure es 0
            avg_historical = np.where(X['tenure'] == 0, X['MonthlyCharges'], X['TotalCharges'] / X['tenure'])
            X['Price_Shock'] = X['MonthlyCharges'] - avg_historical
            
        # Interacción de riesgo máximo
        if 'Contract' in X.columns and 'InternetService' in X.columns:
            X['High_Risk_Profile'] = ((X['Contract'] == 'Month-to-month') & 
                                      (X['InternetService'] == 'Fiber optic')).astype(int)
                                      
        # descarte de columnas
        cols_to_drop = [c for c in self.useless_cols if c in X.columns]
        
        if self.drop_original:
            replaced = ['Partner', 'Dependents', 'PaymentMethod']
            cols_to_drop.extend([c for c in replaced if c in X.columns])
            
        if self.drop_addons:
            cols_to_drop.extend([c for c in self.services if c in X.columns])
            
        X = X.drop(columns=cols_to_drop, errors='ignore')
        
        return X