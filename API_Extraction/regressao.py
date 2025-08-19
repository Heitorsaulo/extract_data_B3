# regression.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

def train_random_forest(X_train, y_train, n_estimators=500, random_state=42):
    """Trains a RandomForestRegressor model."""
    model_regression = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    cross_val_scores = cross_val_score(model_regression, X_train, y_train, cv=5)
    print(f"Cross-Validation Scores: {cross_val_scores}")
    print(f"Mean Cross-Validation Score: {np.mean(cross_val_scores):.4f}")

    model_regression.fit(X_train, y_train)
    return model_regression

def train_xgboost(X_train, y_train, n_estimators=75, learning_rate=0.1, max_depth=5, random_state=42):
    """Trains an XGBoost Regressor model."""
    xgboost = xgb.XGBRegressor(n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                random_state=random_state)
    xgboost.fit(X_train, y_train)
    return xgboost

def evaluate_model(model, X_test, y_test, scaler=None):
    """Evaluates the regression model."""
    y_pred = model.predict(X_test)
    if scaler:
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return y_pred, y_test

if __name__ == '__main__':
    # Load data
    merged_info_monthly = pd.read_csv('data/preprocessed_reit_data.csv')

    # Define columns to eliminate
    lista_colunas_eliminar = ['year','month', 'open_mean', 'close_mean','symbol', 'close_min', 'close_max', 'open_min', 'open_max']

    # Prepare data
    imputer = SimpleImputer(strategy='mean')
    symbol_column = merged_info_monthly['symbol']
    dados_treinamento = np.array(imputer.fit_transform(merged_info_monthly.drop(columns=lista_colunas_eliminar)))
    dados_resposta = np.array(merged_info_monthly['close_mean'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(dados_treinamento, dados_resposta, test_size=0.2, random_state=42)

    # Scale data
    X_scaler = MinMaxScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)  # Use transform instead of fit_transform

    # Train and evaluate RandomForestRegressor
    model_rf = train_random_forest(X_train_scaled, y_train)
    y_pred_rf, y_test_rf = evaluate_model(model_rf, X_test_scaled, y_test)

    # Train and evaluate XGBoost
    model_xgb = train_xgboost(X_train, y_train)
    y_pred_xgb, y_test_xgb = evaluate_model(model_xgb, X_test)

    # Create and save results DataFrame
    resultados_rf = pd.DataFrame({'Actual': y_test_rf, 'Predicted': y_pred_rf, 'Model': 'RandomForest'})
    resultados_xgb = pd.DataFrame({'Actual': y_test_xgb, 'Predicted': y_pred_xgb, 'Model': 'XGBoost'})
    resultados = pd.concat([resultados_rf, resultados_xgb])
    resultados['symbol'] = symbol_column.iloc[resultados.index]

    resultados.to_csv('regression_results.csv', index=False)
    print("Regression results saved to 'regression_results.csv'")