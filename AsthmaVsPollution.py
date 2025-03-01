############################################################
# Air Pollution and Respiratory Diseases Prediction Project
# By: Ezra Raj
# Objective: 
#   1. Analyze how air pollutants (PM2.5, NO2, CO, O3) 
#      impact respiratory diseases like asthma.
#   2. Detect seasonal trends (via time-series analysis).
#   3. Build regression models to predict asthma-related 
#      hospitalizations or case numbers.
#   4. Provide recommendations for improving asthma outcomes.
############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For regression modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# For time-series (optional advanced approach)
import statsmodels.api as sm

############################################
# 1. LOAD DATA
############################################
pollution_df = pd.read_csv('pollution_data.csv', parse_dates=['Date'])
city_asthma_df = pd.read_csv('AsthmaCityRankings.csv')
state_asthma_df = pd.read_csv('AsthmaStateRankings.csv')
states_temp_df = pd.read_csv('StatesTemperature.csv')

############################################
# 2. INITIAL INSPECTION & CLEANING
############################################
print("\n--- Pollution Data (head) ---")
print(pollution_df.head())
print("\n--- City Asthma Data (head) ---")
print(city_asthma_df.head())
print("\n--- State Asthma Data (head) ---")
print(state_asthma_df.head())
print("\n--- States Temperature Data (head) ---")
print(states_temp_df.head())

# Example cleaning steps:
# - Convert numeric columns that may have commas or strings
# - Remove duplicates or drop missing data as needed

pollution_df['Population Staying at Home'] = (
    pollution_df['Population Staying at Home']
    .replace({',':''}, regex=True)
    .astype(float)
)
pollution_df['Population Not Staying at Home'] = (
    pollution_df['Population Not Staying at Home']
    .replace({',':''}, regex=True)
    .astype(float)
)

# city_asthma_df, state_asthma_df, states_temp_df, etc.

############################################
# 3. EXPLORATORY DATA ANALYSIS (EDA)
############################################

# Basic stats
print("\n--- Pollution Data Description ---")
print(pollution_df.describe())

# Plot distribution of key pollutants
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(pollution_df['pm25_median'], kde=True, ax=axes[0,0])
axes[0,0].set_title('PM2.5 Median Distribution')

sns.histplot(pollution_df['no2_median'], kde=True, ax=axes[0,1])
axes[0,1].set_title('NO2 Median Distribution')

sns.histplot(pollution_df['co_median'], kde=True, ax=axes[1,0])
axes[1,0].set_title('CO Median Distribution')

sns.histplot(pollution_df['o3_median'], kde=True, ax=axes[1,1])
axes[1,1].set_title('O3 Median Distribution')

plt.tight_layout()
plt.show()

############################################
# 4. TIME-SERIES ANALYSIS
############################################
pollution_df['Date'] = pd.to_datetime(pollution_df['Date'])

pollution_df['YearMonth'] = pollution_df['Date'].dt.to_period('M')
monthly_pollution = pollution_df.groupby('YearMonth').agg({
    'pm25_median':'mean',
    'no2_median':'mean',
    'co_median':'mean',
    'o3_median':'mean',
    'temperature_median':'mean'  # if present
}).reset_index()

monthly_pollution['YearMonth'] = monthly_pollution['YearMonth'].astype(str)
monthly_pollution['YearMonth'] = pd.to_datetime(monthly_pollution['YearMonth'])

plt.figure(figsize=(10, 6))
plt.plot(monthly_pollution['YearMonth'], monthly_pollution['pm25_median'], label='PM2.5')
plt.plot(monthly_pollution['YearMonth'], monthly_pollution['o3_median'], label='O3')
plt.xlabel('Date')
plt.ylabel('Pollutant Level (Median)')
plt.title('Monthly Average Pollutant Trends')
plt.legend()
plt.show()

# Seasonal decomposition example (requires a regular date frequency)
# We can demonstrate for PM2.5
monthly_pollution.set_index('YearMonth', inplace=True)
res = sm.tsa.seasonal_decompose(monthly_pollution['pm25_median'], model='additive', period=12)
res.plot()
plt.show()
monthly_pollution.reset_index(inplace=True)

############################################
# 5. MERGE DATA (Pollution + Asthma + Temperature)
############################################
# Suppose city_asthma_df has columns: [City, State, AsthmaRate, ...]
# Suppose state_asthma_df has columns: [State, AsthmaPrevalence, ...]
# Suppose states_temp_df has columns: [State, Date, AvgTemp, ...]
# Adjust as needed.

merged_city_data = pd.merge(
    pollution_df, 
    city_asthma_df, 
    how='left', 
    left_on=['City','State'], 
    right_on=['City','State']
)
print(merged_city_data.head())

# Similarly, merge with state-level data if you want state-level analysis
merged_state_data = pd.merge(
    pollution_df, 
    state_asthma_df, 
    how='left',
    left_on='State',
    right_on='State'
)
print(merged_state_data.head())
# If states_temp_df is by date & state, you can do:
merged_state_data = pd.merge(
    merged_state_data,
    states_temp_df,
    how='left',
    left_on=['State','Date'],
    right_on=['State','Date']
)

print("\n--- Merged City-Level Data (head) ---")
print(merged_city_data.head())

print("\n--- Merged State-Level Data (head) ---")
print(merged_state_data.head())

############################################
# 6. CORRELATION ANALYSIS
############################################
# Let's see how pollutants correlate with asthma metrics
corr_cols = ['pm25_median','no2_median','co_median','o3_median','AsthmaRate']
corr_matrix = merged_city_data[corr_cols].corr()
print("\n--- Correlation Matrix (City Data) ---")
print(corr_matrix)

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation between Pollutants and Asthma Rate')
plt.show()

############################################
# 7. REGRESSION MODEL TO PREDICT ASTHMA CASES
############################################
# We'll do a simple linear regression.

# 7.1 Prepare features & target
features = ['pm25_median','no2_median','co_median','o3_median','temperature_median']
# Filter out rows with missing data
model_data = merged_city_data.dropna(subset=features + ['AsthmaRate'])

X = model_data[features]
y = model_data['AsthmaRate']

# 7.2 Split data into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# 7.3 Train a linear regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# 7.4 Evaluate the model
y_pred = reg_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Regression Model Results ---")
print("Coefficients:", reg_model.coef_)
print("Intercept:", reg_model.intercept_)
print(f"RMSE: {rmse:.2f}")
print(f"R^2: {r2:.2f}")

plan = """
Recommendations for Asthma Management Based on Analysis:
1. Monitor Air Quality: 
   - People with asthma should track daily PM2.5, NO2, O3, and CO levels. 
   - Alert systems can warn on high-pollution days.

2. Seasonal Preparedness:
   - Time-series analysis suggests certain seasons have higher pollution 
     or temperature extremes. 
   - Increase medication adherence and minimize outdoor exposure during 
     these peak periods.

3. Urban Planning & Policy:
   - Encourage local policies reducing vehicle emissions and industrial 
     pollutants in high-risk areas. 
   - Support green infrastructure to improve air quality.

4. Personalized Action Plans:
   - Individuals in cities with high asthma prevalence can use daily 
     pollutant data + weather forecasts to adjust outdoor activities. 
   - Use air purifiers and ensure good ventilation indoors.

5. Further Modeling:
   - Incorporate more advanced machine learning or deep learning 
     (e.g., Random Forest, XGBoost) to improve predictive accuracy. 
   - Use external data (pollen counts, traffic patterns, etc.) 
     to refine asthma risk predictions.
"""

print(plan)
