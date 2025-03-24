import pandas as pd
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("../data/finetuning_set_smiles_plus_features.csv")
X = df.drop(columns=["expt_Hela", "expt_Raw", "smiles"])
y = df[["expt_Hela", "expt_Raw"]]

# Segmented data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SVR
svr = MultiOutputRegressor(SVR(kernel="rbf", C=1.0, epsilon=0.2))

# train
svr.fit(X_train, y_train)

# predict
y_pred = svr.predict(X_test)

# Calculation of assessment indicators
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
pcc = pearsonr(y_test.values.flatten(), y_pred.flatten())[0]
r2 = r2_score(y_test, y_pred)

print(f"SVR\nMSE: {mse:.3f}, \nMAE: {mae:.3f}, \nPCC: {pcc:.3f}, \nR²: {r2:.3f}")
