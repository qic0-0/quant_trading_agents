import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Create features if not provided
if features is None:
    features = pd.DataFrame()
    
    # RSI (14-day)
    delta = prices.diff(1)
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean().abs()
    RS = roll_up / roll_down
    RSI = 100.0 - (100.0 / (1.0 + RS))
    features['RSI'] = RSI
    
    # MACD
    ema_12 = prices.ewm(span=12, adjust=False).mean()
    ema_26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    features['MACD'] = macd
    
    # Moving averages
    features['MA_5'] = prices.rolling(window=5).mean()
    features['MA_10'] = prices.rolling(window=10).mean()
    features['MA_20'] = prices.rolling(window=20).mean()
    
    # Momentum
    features['Momentum'] = prices.diff(10)
    
    # Volatility
    features['Volatility'] = prices.rolling(window=14).std()

# Handle NaN values
features = features.dropna()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost classifier
model = xgb.XGBClassifier(objective='binary:logistic', max_depth=5, learning_rate=0.1, n_estimators=100, subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model and the feature scaler
joblib.dump((model, scaler), model_path)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Create the result dictionary
result = {
    'model_type': 'XGBoost',
    'accuracy': accuracy,
    'n_features': X_train.shape[1],
    'feature_names': X_train.columns.tolist(),
    'n_samples': X_train.shape[0]
}