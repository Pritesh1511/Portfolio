#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 20:30:55 2023

@author: vedantpatel
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def compute_features(data):
    # RSI calculation
    delta = data['Close'].diff()
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)
    avg_gain = up.rolling(window=14).mean()
    avg_loss = down.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['STD20'] = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['SMA20'] + (data['STD20']*2)
    data['Lower_Band'] = data['SMA20'] - (data['STD20']*2)
    
    # SMA features
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    
    # VWAP
    data['Cum_Volume'] = data['Volume'].cumsum()
    data['Cum_Vol_Price'] = (data['Close'] * data['Volume']).cumsum()
    data['VWAP'] = data['Cum_Vol_Price'] / data['Cum_Volume']
    
    # Lag features
    for i in range(1, 4):
        data[f'Close_Lag_{i}'] = data['Close'].shift(i)
        
    data = data.dropna()
    
    return data

# Load and preprocess SPY data for model training
data = pd.read_excel("SPX.xlsx")
data.columns = data.iloc[0]
data = data[1:]
data = compute_features(data)

# Define the labels for the SPY data
data['Label'] = 'Hold'
buy_conditions = (
    (data['SMA50'] > data['SMA200']) &
    ((data['RSI'] < 30) | (data['Close'] < data['Lower_Band']) | (data['Close'] < data['VWAP']))
)
data.loc[buy_conditions, 'Label'] = 'Buy'
sell_conditions = (
    (data['SMA50'] < data['SMA200']) &
    ((data['RSI'] > 70) | (data['Close'] > data['Upper_Band']) | (data['Close'] > data['VWAP']))
)
data.loc[sell_conditions, 'Label'] = 'Sell'

# Prepare the SPY data for training
features = ['RSI', 'Upper_Band', 'Lower_Band', 'SMA50', 'SMA200', 'VWAP', 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3']
X = data[features].dropna()
y = data['Label'][X.index]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_classifier = grid_search.best_estimator_

# Extract feature importances
feature_importances = best_classifier.feature_importances_

# Create a DataFrame to display the feature importances
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# Load Dow Jones data
data_dow = pd.read_excel("DowJones.xlsx")
data_dow = compute_features(data_dow)

# Define the labels for the Dow Jones data
data_dow['Label'] = 'Hold'
buy_conditions_dow = (
    (data_dow['SMA50'] > data_dow['SMA200']) &
    ((data_dow['RSI'] < 30) | (data_dow['Close'] < data_dow['Lower_Band']) | (data_dow['Close'] < data_dow['VWAP']))
)
data_dow.loc[buy_conditions_dow, 'Label'] = 'Buy'
sell_conditions_dow = (
    (data_dow['SMA50'] < data_dow['SMA200']) &
    ((data_dow['RSI'] > 70) | (data_dow['Close'] > data_dow['Upper_Band']) | (data_dow['Close'] > data_dow['VWAP']))
)
data_dow.loc[sell_conditions_dow, 'Label'] = 'Sell'


# Prepare the Dow Jones data for prediction
X_dow = data_dow[features].dropna()
y_dow = data_dow['Label'][X_dow.index]


# Backtesting strategy on Dow Jones data
initial_capital = 100000
portfolio_value = initial_capital

signals_and_returns = pd.DataFrame(index=X_dow.index)
signals_and_returns['Actual_Return'] = data_dow.loc[X_dow.index, 'Close'].pct_change()

trading_allowed = True

for idx, row in signals_and_returns.iterrows():
    if not trading_allowed:
        break
    predicted_label = best_classifier.predict(X_dow.loc[[idx]])[0]
    signals_and_returns.loc[idx, 'Predicted_Label'] = predicted_label
    

# Backtesting with Risk Management
initial_capital = 100000
portfolio_value = initial_capital
position_multiplier = 0.01  # 1% of portfolio capital per trade
stop_loss_percentage = 0.02  # 2% stop-loss
take_profit_percentage = 0.02  # 2% take-profit
max_drawdown_percentage = 0.10  # 10% maximum drawdown

# Create a DataFrame to store trading signals and returns
signals_and_returns = pd.DataFrame(index=X_dow.index)
signals_and_returns['Actual_Label'] = y_dow
signals_and_returns['Actual_Return'] = data_dow.loc[X_dow.index, 'Close'].pct_change()

trading_allowed = True

for idx, row in signals_and_returns.iterrows():
    if not trading_allowed:
        break
    predicted_label = best_classifier.predict(X_dow.loc[[idx]])[0]
    signals_and_returns.loc[idx, 'Predicted_Label'] = predicted_label
    
    if predicted_label == 'Buy' and portfolio_value > 0:
        position_size = portfolio_value * position_multiplier
        stop_loss_price = data_dow.loc[idx, 'Close'] * (1 - stop_loss_percentage)
        take_profit_price = data_dow.loc[idx, 'Close'] * (1 + take_profit_percentage)

        # Calculate position based on trade conditions
        if data_dow.loc[idx, 'Close'] > stop_loss_price:
            position_size = position_size * (take_profit_price / data_dow.loc[idx, 'Close'])
        else:
            position_size = position_size * (stop_loss_price / data_dow.loc[idx, 'Close'])

        signals_and_returns.loc[idx, 'Position'] = position_size
        signals_and_returns.loc[idx, 'Stop_Loss_Price'] = stop_loss_price
        signals_and_returns.loc[idx, 'Take_Profit_Price'] = take_profit_price
    elif predicted_label == 'Sell':
        position_size = -portfolio_value * position_multiplier
        stop_loss_price = data_dow.loc[idx, 'Close'] * (1 + stop_loss_percentage)  # Adjust for Sell stop loss
        take_profit_price = data_dow.loc[idx, 'Close'] * (1 - take_profit_percentage)  # Adjust for Sell take profit

        # Calculate position based on trade conditions
        if data_dow.loc[idx, 'Close'] > take_profit_price:
            position_size = position_size * (take_profit_price / data_dow.loc[idx, 'Close'])
        else:
            position_size = position_size * (stop_loss_price / data_dow.loc[idx, 'Close'])

        signals_and_returns.loc[idx, 'Position'] = position_size
        signals_and_returns.loc[idx, 'Stop_Loss_Price'] = stop_loss_price
        signals_and_returns.loc[idx, 'Take_Profit_Price'] = take_profit_price
    else:
        signals_and_returns.loc[idx, 'Position'] = 0  # No position taken

    # Calculate portfolio returns
    if 'Position' in signals_and_returns.columns and not np.isnan(signals_and_returns.loc[idx, 'Position']):
        if signals_and_returns.loc[idx, 'Position'] != 0:
            portfolio_return = signals_and_returns.loc[idx, 'Position'] * signals_and_returns.loc[idx, 'Actual_Return']
            if not np.isnan(portfolio_return):  # Check for NaN values
                signals_and_returns.loc[idx, 'Portfolio_Return'] = portfolio_return
                portfolio_value += portfolio_return

        # Check for maximum drawdown
    if portfolio_value < initial_capital * (1 - max_drawdown_percentage):
        trading_allowed = False

# Print final portfolio value and other performance metrics
print("Initial Portfolio Value:", initial_capital)
print("Final Portfolio Value:", portfolio_value)

# Gaphs for the results
# SPY Visualization
plt.figure(figsize=(14,7))
plt.plot(data.index, data['Close'], label="Close Price")
plt.plot(data.index, data['SMA50'], label="SMA50", color="orange")
plt.plot(data.index, data['SMA200'], label="SMA200", color="green")
plt.plot(data.index, data['Upper_Band'], label="Upper Bollinger Band", color="red")
plt.plot(data.index, data['Lower_Band'], label="Lower Bollinger Band", color="blue")
plt.title("SPY Close Price with Indicators")
plt.legend()
plt.show()

# RSI visualization for SPY
plt.figure(figsize=(14,7))
plt.plot(data.index, data['RSI'], label="RSI")
plt.axhline(70, color="red", linestyle="--")  # Overbought line
plt.axhline(30, color="green", linestyle="--")  # Oversold line
plt.title("SPY RSI Indicator")
plt.legend()
plt.show()

# DowJones Visualization
plt.figure(figsize=(14,7))
plt.plot(data_dow.index, data_dow['Close'], label="Close Price")
buy_signals = signals_and_returns[signals_and_returns['Predicted_Label'] == 'Buy']
sell_signals = signals_and_returns[signals_and_returns['Predicted_Label'] == 'Sell']
plt.scatter(buy_signals.index, data_dow.loc[buy_signals.index, 'Close'], label="Buy Signal", marker="^", color="green")
plt.scatter(sell_signals.index, data_dow.loc[sell_signals.index, 'Close'], label="Sell Signal", marker="v", color="red")
plt.title("Dow Jones Close Price with Buy/Sell Signals")
plt.legend()
plt.show()

# Feature Importance Graph
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], align='center')
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.gca().invert_yaxis()  # Display the most important feature at the top
plt.show()

# Measures for the strategy
# SPY Metrics
train_predictions = best_classifier.predict(X_train)
test_predictions = best_classifier.predict(X_test)
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"SPY Training Accuracy: {train_accuracy:.2%}")
print(f"SPY Testing Accuracy: {test_accuracy:.2%}")

# DowJones Backtesting Metrics
actual_labels = signals_and_returns['Actual_Label']
predicted_labels = signals_and_returns['Predicted_Label']
backtest_accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"DowJones Backtesting Accuracy: {backtest_accuracy:.2%}")



