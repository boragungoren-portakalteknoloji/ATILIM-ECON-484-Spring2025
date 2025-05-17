# 1. Simple Linear Regression on the Data Set
import pandas as pd
import statsmodels.api as sm

# 1.1 Load and clean the dataset
df = pd.read_csv('Discovery.csv', delimiter='\t', header=0)
df.columns = ['Y', 'X1', 'X2', 'X3', 'X4', 'X5','X6','X7','X8']

# Convert comma-based decimals to float
for col in df.columns:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# 1.2 Define predictors and target
X = df[['X1', 'X2', 'X3', 'X4', 'X5','X6','X7','X8']]
y = df['Y']

# 1.3 Add constant term for intercept
X = sm.add_constant(X)

# 1.4 Fit the model using Ordinary Least Squares
model = sm.OLS(y, X).fit()

# 1.5 Print detailed summary
print(model.summary())

# 2. Conduct PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("Starting PCA")
# 2.1. Load and clean the dataset
# Skipped because done altready at 1.1
# df = pd.read_csv("Discovery.csv", delimiter=":")
# df.columns = ['Y', 'X1', 'X2', 'X3', 'X4', 'X5']

# 2.2. Convert comma decimals to dot decimals and cast to float
# Skipped because done altready at 1.2
# for col in df.columns:
#     df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# 2.3. Separate features and target
# Skippled because done already at 1.3
# X = df[['X1', 'X2', 'X3', 'X4', 'X5']]
# y = df['Y']

# 2.4 Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2.5. Perform PCA
pca = PCA()
pca.fit(X_scaled)

# 2.6. Extract loadings (components)
loadings = pd.DataFrame(
    pca.components_,
    columns=X.columns,
    index=[f'PC{i+1}' for i in range(X.shape[1])]
)

# 2.7 Select top contributing variables from PC1
# (Choose top 3 variables by absolute value in PC1)
required_number_of_variables = 3
top_vars = loadings.loc['PC1'].abs().sort_values(ascending=False).head(required_number_of_variables).index.tolist()

# 2.8 Create a new DataFrame with Y and selected variables
df_selected = df[['Y'] + top_vars]

# 2.9 Optional: print selected features
print("Selected variables from PCA:", top_vars)
print(df_selected.head())


# 3. Conduct regression with the selected variables
# use df_selected = df[['Y'] + top_vars] instead of df.

# 3.2
X2 = df_selected[['X1', 'X6', 'X8']]
y2 = df_selected['Y']

# 3.3 Add constant term for intercept
X = sm.add_constant(X2)

# 3.4 Fit the model using Ordinary Least Squares
model2 = sm.OLS(y2, X2).fit()

# 1.5 Print detailed summary
print(model2.summary())