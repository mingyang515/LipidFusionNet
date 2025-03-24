import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# Load data
df = pd.read_csv("../data/finetuning_set_smiles_plus_features.csv")
X = df.drop(columns=["expt_Hela", "expt_Raw", "smiles"])
y = df[["expt_Hela", "expt_Raw"]]

# Segmented data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# anticipate
y_pred = rf.predict(X_test)

# Calculation of assessment indicators
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
pcc = pearsonr(y_test.values.flatten(), y_pred.flatten())[0]
r2 = r2_score(y_test, y_pred)

print(f"Random Forest\nMSE: {mse:.3f}, \nMAE: {mae:.3f}, \nPCC: {pcc:.3f}, \nR²: {r2:.3f}")
