import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the data
data = pd.read_csv('../../../labs/03/marketing.csv')

print("=" * 80)
print("MARKETING DATA - MULTIVARIABLE LINEAR REGRESSION ANALYSIS")
print("=" * 80)

# Display basic information about the dataset
print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Dataset shape: {data.shape}")
print(f"\nFirst few rows:")
print(data.head(10))
print(f"\nDataset statistics:")
print(data.describe())
print(f"\nMissing values:")
print(data.isnull().sum())

# Separate features and target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n2. DATA SPLIT")
print("-" * 80)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Display model coefficients and intercept
print(f"\n3. MODEL COEFFICIENTS")
print("-" * 80)
print(f"Intercept: {model.intercept_:.4f}")
print(f"\nCoefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")
    print(f"    -> For every $1000 increase in {feature} spending, Sales increase by ${coef:.4f}k")

# Model evaluation
print(f"\n4. MODEL PERFORMANCE")
print("-" * 80)
print(f"Training Set:")
print(f"  R2 Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
print(f"  MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")

print(f"\nTesting Set:")
print(f"  R2 Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
print(f"  MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")

# Interpretation
print(f"\n5. INTERPRETATION")
print("-" * 80)
print(f"The R2 score of {r2_score(y_test, y_pred_test):.4f} means that {r2_score(y_test, y_pred_test)*100:.2f}% of the variance")
print(f"in Sales can be explained by the model using TV, Radio, and Newspaper spending.")

# Create visualizations
fig = plt.figure(figsize=(16, 12))

# 1. Correlation Matrix
plt.subplot(3, 3, 1)
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix\n(Shows relationships between all variables)', fontsize=12, fontweight='bold')

# 2-4. Individual feature impact on Sales (scatter plots with regression line)
features = ['TV', 'Radio', 'Newspaper']
colors = ['#e74c3c', '#3498db', '#2ecc71']

for idx, (feature, color) in enumerate(zip(features, colors), start=2):
    plt.subplot(3, 3, idx)
    plt.scatter(data[feature], data['Sales'], alpha=0.6, color=color, edgecolors='black', linewidth=0.5)

    # Calculate and plot simple linear regression line for visualization
    z = np.polyfit(data[feature], data['Sales'], 1)
    p = np.poly1d(z)
    plt.plot(data[feature], p(data[feature]), color='red', linewidth=2, linestyle='--',
             label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')

    plt.xlabel(f'{feature} Spending ($1000s)', fontsize=10)
    plt.ylabel('Sales ($1000s)', fontsize=10)
    plt.title(f'{feature} Impact on Sales\n(R2 = {data[feature].corr(data["Sales"])**2:.3f})',
              fontsize=11, fontweight='bold')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

# 5. Predicted vs Actual (Training Set)
plt.subplot(3, 3, 5)
plt.scatter(y_train, y_pred_train, alpha=0.6, color='#9b59b6', edgecolors='black', linewidth=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Sales ($1000s)', fontsize=10)
plt.ylabel('Predicted Sales ($1000s)', fontsize=10)
plt.title(f'Training Set: Predicted vs Actual\n(R2 = {r2_score(y_train, y_pred_train):.3f})',
          fontsize=11, fontweight='bold')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# 6. Predicted vs Actual (Testing Set)
plt.subplot(3, 3, 6)
plt.scatter(y_test, y_pred_test, alpha=0.6, color='#e67e22', edgecolors='black', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Sales ($1000s)', fontsize=10)
plt.ylabel('Predicted Sales ($1000s)', fontsize=10)
plt.title(f'Testing Set: Predicted vs Actual\n(R2 = {r2_score(y_test, y_pred_test):.3f})',
          fontsize=11, fontweight='bold')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# 7. Residuals Plot (Training Set)
plt.subplot(3, 3, 7)
residuals_train = y_train - y_pred_train
plt.scatter(y_pred_train, residuals_train, alpha=0.6, color='#16a085', edgecolors='black', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Sales ($1000s)', fontsize=10)
plt.ylabel('Residuals', fontsize=10)
plt.title('Training Set: Residuals Plot\n(Should be randomly scattered around 0)',
          fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)

# 8. Residuals Plot (Testing Set)
plt.subplot(3, 3, 8)
residuals_test = y_test - y_pred_test
plt.scatter(y_pred_test, residuals_test, alpha=0.6, color='#f39c12', edgecolors='black', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Sales ($1000s)', fontsize=10)
plt.ylabel('Residuals', fontsize=10)
plt.title('Testing Set: Residuals Plot\n(Should be randomly scattered around 0)',
          fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)

# 9. Feature Importance (Coefficients Bar Plot)
plt.subplot(3, 3, 9)
coefficients = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
colors_bar = ['#e74c3c' if x > 0 else '#3498db' for x in coefficients]
bars = plt.bar(range(len(coefficients)), coefficients.values, color=colors_bar,
               edgecolor='black', linewidth=1.5)
plt.xticks(range(len(coefficients)), coefficients.index, rotation=45)
plt.ylabel('Coefficient Value', fontsize=10)
plt.title('Feature Importance (Model Coefficients)\n(Higher = More impact on Sales)',
          fontsize=11, fontweight='bold')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

plt.tight_layout()
plt.savefig('marketing_regression_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n6. VISUALIZATIONS")
print("-" * 80)
print("Comprehensive visualization saved as 'marketing_regression_analysis.png'")
print("\nThe visualization includes:")
print("  1. Correlation Matrix - Shows relationships between all variables")
print("  2-4. Individual Feature Impact Plots - How each advertising channel affects sales")
print("  5-6. Predicted vs Actual Plots - Model prediction accuracy")
print("  7-8. Residuals Plots - Model error distribution")
print("  9. Feature Importance - Which features have the strongest impact")

plt.show()

# Additional analysis: Which feature is most important?
print(f"\n7. FEATURE IMPORTANCE RANKING")
print("-" * 80)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"{row['Feature']:10s}: {row['Coefficient']:7.4f} (Impact rank: {list(feature_importance['Feature']).index(row['Feature']) + 1})")

print(f"\nMost influential feature (per dollar): {feature_importance.iloc[0]['Feature']}")
print(f"Least influential feature (per dollar): {feature_importance.iloc[-1]['Feature']}")

# Advanced analysis: Total impact vs Per-Dollar impact
print(f"\n8. TOTAL IMPACT ANALYSIS")
print("-" * 80)
print("Understanding the difference between per-dollar efficiency and total impact:\n")

# Calculate average spending for each channel
avg_spending = data[['TV', 'Radio', 'Newspaper']].mean()
total_impact = model.coef_ * avg_spending.values

impact_analysis = pd.DataFrame({
    'Feature': X.columns,
    'Avg_Spending': avg_spending.values,
    'Coefficient': model.coef_,
    'Total_Impact': total_impact,
    'Correlation_with_Sales': [data[col].corr(data['Sales']) for col in X.columns]
}).sort_values('Total_Impact', ascending=False)

print("Per-Dollar Efficiency (Coefficient):")
print("  -> How much sales increase for every $1000 spent\n")
for idx, row in impact_analysis.sort_values('Coefficient', ascending=False).iterrows():
    print(f"  {row['Feature']:10s}: ${row['Coefficient']:.4f}k sales per $1k spent")

print("\nTotal Impact at Average Spending Levels:")
print("  -> Coefficient * Average Spending = Total contribution to sales\n")
for idx, row in impact_analysis.iterrows():
    print(f"  {row['Feature']:10s}: {row['Coefficient']:.4f} * ${row['Avg_Spending']:.2f}k = ${row['Total_Impact']:.2f}k contribution")

print("\nDirect Correlation with Sales:")
print("  -> How strongly each variable correlates with sales\n")
for idx, row in impact_analysis.sort_values('Correlation_with_Sales', ascending=False).iterrows():
    print(f"  {row['Feature']:10s}: {row['Correlation_with_Sales']:.4f}")

print("\n" + "=" * 40)
print("KEY INSIGHTS")
print("=" * 40)
print(f"\n1. HIGHEST PER-DOLLAR EFFICIENCY: {impact_analysis.sort_values('Coefficient', ascending=False).iloc[0]['Feature']}")
print(f"   Every $1000 spent returns ${impact_analysis.sort_values('Coefficient', ascending=False).iloc[0]['Coefficient']:.4f}k in sales")

print(f"\n2. HIGHEST TOTAL IMPACT: {impact_analysis.iloc[0]['Feature']}")
print(f"   At average spending levels (${impact_analysis.iloc[0]['Avg_Spending']:.2f}k), contributes ${impact_analysis.iloc[0]['Total_Impact']:.2f}k to sales")

print(f"\n3. STRONGEST CORRELATION: {impact_analysis.sort_values('Correlation_with_Sales', ascending=False).iloc[0]['Feature']}")
print(f"   Correlation coefficient: {impact_analysis.sort_values('Correlation_with_Sales', ascending=False).iloc[0]['Correlation_with_Sales']:.4f}")

print("\n4. RECOMMENDATION:")
most_efficient = impact_analysis.sort_values('Coefficient', ascending=False).iloc[0]['Feature']
highest_impact = impact_analysis.iloc[0]['Feature']

if most_efficient == highest_impact:
    print(f"   {most_efficient} is both the most efficient per dollar AND has the highest total impact.")
    print(f"   -> STRONGLY RECOMMENDED to prioritize {most_efficient} advertising!")
else:
    print(f"   {most_efficient} is most efficient per dollar (best ROI)")
    print(f"   {highest_impact} has highest total impact due to higher spending levels")
    print(f"   -> Balance your budget: {most_efficient} for efficiency, {highest_impact} for volume")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
