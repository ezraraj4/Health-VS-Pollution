############################################################
# Air Pollution and Mental Health Analysis Project
# By: Ezra Raj
# 
# Objectives:
#   1. Investigate how air pollutants (e.g., PM2.5, NO2, CO, O3)
#      may be linked to mental health indicators (e.g., depression, anxiety).
#   2. Analyze seasonal or temperature-related trends and their 
#      potential impact on mental health.
#   3. Build regression models to predict mental health metrics 
#      (e.g., AMI_percent) based on pollution and temperature data.
#   4. Provide recommendations to improve mental health outcomes 
#      based on the findings.
############################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


mental_health_file = 'StatesMentalHealth.csv'
temperature_file = 'StatesTemperature.csv'

pollution_df = pd.read_csv('pollution_data.csv')
mental_health_df = pd.read_csv(mental_health_file)
temperature_df = pd.read_csv(temperature_file)

# Display a few rows of each dataset
print("\n--- Pollution Data ---")
print(pollution_df.head())

print("\n--- Mental Health Data ---")
print(mental_health_df.head())

print("\n--- Temperature Data ---")
print(temperature_df.head())

# -------------------------------------------------------------------
# 2. Merge the Datasets
# -------------------------------------------------------------------

# Convert date column in pollution data 
if 'Date' in pollution_df.columns:
    pollution_df['Date'] = pd.to_datetime(pollution_df['Date'], errors='coerce')

# Standardize the 'State' column in all dataframes
mental_health_df['State'] = mental_health_df['State'].str.strip().str.upper()
pollution_df['State'] = pollution_df['State'].str.strip().str.upper()
temperature_df['State'] = temperature_df['State'].str.strip().str.upper()

state_abbrev_to_name = {
    'AL': 'ALABAMA','AK': 'ALASKA','AZ': 'ARIZONA','AR': 'ARKANSAS','CA': 'CALIFORNIA',
    'CO': 'COLORADO','CT': 'CONNECTICUT','DE': 'DELAWARE','FL': 'FLORIDA','GA': 'GEORGIA',
    'HI': 'HAWAII','ID': 'IDAHO','IL': 'ILLINOIS','IN': 'INDIANA','IA': 'IOWA',
    'KS': 'KANSAS','KY': 'KENTUCKY','LA': 'LOUISIANA','ME': 'MAINE','MD': 'MARYLAND',
    'MA': 'MASSACHUSETTS','MI': 'MICHIGAN','MN': 'MINNESOTA','MS': 'MISSISSIPPI',
    'MO': 'MISSOURI','MT': 'MONTANA','NE': 'NEBRASKA','NV': 'NEVADA','NH': 'NEW HAMPSHIRE',
    'NJ': 'NEW JERSEY','NM': 'NEW MEXICO','NY': 'NEW YORK','NC': 'NORTH CAROLINA',
    'ND': 'NORTH DAKOTA','OH': 'OHIO','OK': 'OKLAHOMA','OR': 'OREGON','PA': 'PENNSYLVANIA',
    'RI': 'RHODE ISLAND','SC': 'SOUTH CAROLINA','SD': 'SOUTH DAKOTA','TN': 'TENNESSEE',
    'TX': 'TEXAS','UT': 'UTAH','VT': 'VERMONT','VA': 'VIRGINIA','WA': 'WASHINGTON',
    'WV': 'WEST VIRGINIA','WI': 'WISCONSIN','WY': 'WYOMING'
}

# Map abbreviations to full state names
if pollution_df['State'].isin(state_abbrev_to_name.keys()).any():
    pollution_df['State'] = pollution_df['State'].map(state_abbrev_to_name)

# Aggregate pollution data by state
pollution_state_avg = pollution_df.groupby('State').mean(numeric_only=True).reset_index()

# Merge the three datasets on 'State'
merged_df = mental_health_df.merge(pollution_state_avg, on='State', how='inner')
merged_df = merged_df.merge(temperature_df, on='State', how='inner')

# Display a few rows of the merged data
print("\n--- Merged Data (Head) ---")
print(merged_df.head())

print("\n--- Merged Data Info ---")
print(merged_df.info())

# -------------------------------------------------------------------
# 3. Exploratory Data Analysis (EDA)
# -------------------------------------------------------------------

# 3a. Correlation Analysis
correlation_matrix = merged_df.corr(numeric_only=True)
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 3b. Seasonal Trend Analysis: Mental Health vs Temperature
temperature_columns = [col for col in merged_df.columns if '202' in col]
mental_health_metric = 'AMI_percent'  # Adjust if you want a different metric

plt.figure(figsize=(15, 8))
for state in merged_df['State'].unique():
    state_data = merged_df[merged_df['State'] == state]
    if not state_data.empty:
        plt.plot(temperature_columns,
                 state_data[temperature_columns].values.flatten(),
                 label=state)

plt.xlabel('Month')
plt.ylabel('Average Temperature')
plt.title('Seasonal Temperature Trends by State')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# -------------------------------------------------------------------
# 4. Regression Modeling to Predict Mental Health Issues
# -------------------------------------------------------------------

# 4a. Prepare Features (X) and Target (y)
# Drop columns that are not numeric or not relevant
columns_to_drop = [
    'State', 'AMI_percent', 'AMI_number', 'SUD_percent', 'SUD_number',
    'Suicide_percent', 'Suicide_number'
]
X = merged_df.drop(columns=columns_to_drop, errors='ignore')
y = merged_df[mental_health_metric]

# Convert all columns in X to numeric, forcing errors to NaN
X_numeric = X.apply(pd.to_numeric, errors='coerce')

# Impute missing values with column means
X_imputed = X_numeric.fillna(X_numeric.mean())

# 4b. Split Data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

# 4c. Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# 4d. Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 4e. Evaluate Models
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)

rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

print("\n--- Model Performance ---")
print(f"Linear Regression MSE: {linear_mse:.4f}, R2: {linear_r2:.4f}")
print(f"Random Forest MSE: {rf_mse:.4f}, R2: {rf_r2:.4f}")

# 4f. Residual Analysis (Linear Model)
residuals = y_test - y_pred_linear
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple')
plt.title('Residuals Distribution (Linear Model)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 4g. Feature Importance (Random Forest)
feature_importances = pd.Series(rf_model.feature_importances_, index=X_imputed.columns)
feature_importances_sorted = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(12, 8))
feature_importances_sorted.head(15).plot(kind='barh', color='teal')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.grid(True)
plt.show()

print("\n--- Top 15 Important Features (Random Forest) ---")
print(feature_importances_sorted.head(15))

plan = """
Recommendations for Improving Mental Health Based on Analysis:
1. Monitor Air Quality:
   - States with high PM2.5 or O3 levels might allocate resources 
     for mental health services or public awareness.
2. Seasonal Preparedness:
   - If temperature extremes correlate with higher mental health issues,
     promote interventions during those months.
3. Further Modeling:
   - Incorporate additional variables (e.g., socioeconomic data, 
     healthcare access) to improve predictive accuracy.
4. Policy and Community Actions:
   - Encourage policies reducing pollutant emissions.
   - Expand community-based mental health programs where pollution is high.
5. Feedback Loop:
   - Gather user feedback on mental health outcomes.
   - Refine the model with more granular data (city-level or zip-code-level).

Next Steps:
- Integrate more data sources (e.g., real-time air quality indexes).
- Explore advanced ML methods (Random Forest, XGBoost).
- Conduct thorough validation with real mental health outcome data.
"""

print(plan)