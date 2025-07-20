import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
print("Label: Loading Academic Performance Dataset")
data = pd.read_csv('Academic.csv')
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(data) 

# Preprocessing
print("\nLabel: Dataset Information Before Preprocessing")
print(data.info())
print("\nLabel: Missing Values in Dataset")
print(data.isnull().sum())

# Handling missing values
numerical_columns = ['Study Hours per Week', 'Attendance Rate', 'High School GPA', 
                    'Extracurricular Activities', 'Library Usage per Week', 
                    'Online Coursework Engagement', 'Sleep Hours per Night']
categorical_columns = ['Major', 'Part-Time Job']
target_column = 'College GPA'

print("\nLabel: Imputing Missing Values")
# Impute numerical columns with mean
num_imputer = SimpleImputer(strategy='mean')
data[numerical_columns] = num_imputer.fit_transform(data[numerical_columns])
print(" - Numerical columns imputed with mean")

# Impute categorical columns with most frequent
cat_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_columns] = cat_imputer.fit_transform(data[categorical_columns])
print(" - Categorical columns imputed with most frequent value")

# Encode categorical variables
print("\nLabel: Encoding Categorical Variables")
label_encoder_major = LabelEncoder()
label_encoder_job = LabelEncoder()
data['Major'] = label_encoder_major.fit_transform(data['Major'])
data['Part-Time Job'] = label_encoder_job.fit_transform(data['Part-Time Job'])
print(" - 'Major' and 'Part-Time Job' encoded using LabelEncoder")

# Cap outliers using IQR method
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

print("\nLabel: Capping Outliers in Numerical Columns")
for col in numerical_columns + [target_column]:
    data = cap_outliers(data, col)
print(" - Outliers capped using IQR method")

# Scale numerical features
print("\nLabel: Scaling Numerical Features")
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
print(" - Numerical features scaled using StandardScaler")

# Exploratory Data Analysis
print("\nLabel: Performing Exploratory Data Analysis (EDA)")
# Correlation heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Label: Correlation Heatmap of Academic Features')
plt.savefig('eda_correlation_heatmap.png')
plt.close()
print(" - Saved: Correlation Heatmap as 'eda_correlation_heatmap.png'")

# Distribution of College GPA
plt.figure(figsize=(10, 6))
sns.histplot(data['College GPA'], bins=30, kde=True)
plt.title('Label: Distribution of College GPA')
plt.xlabel('College GPA')
plt.ylabel('Frequency')
plt.savefig('eda_gpa_distribution.png')
plt.close()
print(" - Saved: GPA Distribution Plot as 'eda_gpa_distribution.png'")

# Scatter plots for key features vs. College GPA
key_features = ['Study Hours per Week', 'Attendance Rate', 'High School GPA']
for feature in key_features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[feature], y=data['College GPA'])
    plt.title(f'Label: {feature} vs. College GPA')
    plt.xlabel(feature)
    plt.ylabel('College GPA')
    filename = f'eda_{feature.lower().replace(" ", "_")}_vs_gpa.png'
    plt.savefig(filename)
    plt.close()
    print(f" - Saved: Scatter Plot of {feature} vs. GPA as '{filename}'")

# Prepare data for modeling
print("\nLabel: Preparing Data for Modeling")
X = data.drop(columns=[target_column])
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(" - Data split into 80% training and 20% testing sets")

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'SVR': SVR(kernel='rbf'),
    'SGD Regressor': SGDRegressor(max_iter=1000, tol=1e-3, random_state=42),
    'MLP Regressor': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42)
}

# Evaluate models
print("\nLabel: Training and Evaluating Models")
results = []
for name, model in models.items():
    print(f" - Training {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    mse = round(mean_squared_error(y_test, y_pred), 2)
    rmse = round(np.sqrt(mse), 2)
    r2 = round(r2_score(y_test, y_pred), 2)
    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2
    })
print(" - All models trained and evaluated")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Print results with formatted decimals
print("\nLabel: Model Performance Metrics (MAE, MSE, RMSE, R² Score)")
pd.set_option('display.float_format', '{:.2f}'.format)
print(results_df)

# Visualize model performance
print("\nLabel: Visualizing Model Performance")
plt.figure(figsize=(16, 6))
# Bar plot for MAE
plt.subplot(1, 4, 1)
sns.barplot(x='Model', y='MAE', data=results_df)
plt.title('Label: Mean Absolute Error (MAE)')
plt.xticks(rotation=45)
# Bar plot for MSE
plt.subplot(1, 4, 2)
sns.barplot(x='Model', y='MSE', data=results_df)
plt.title('Label: Mean Squared Error (MSE)')
plt.xticks(rotation=45)
# Bar plot for RMSE
plt.subplot(1, 4, 3)
sns.barplot(x='Model', y='RMSE', data=results_df)
plt.title('Label: Root Mean Squared Error (RMSE)')
plt.xticks(rotation=45)
# Bar plot for R2 Score
plt.subplot(1, 4, 4)
sns.barplot(x='Model', y='R2 Score', data=results_df)
plt.title('Label: R² Score (Variance Explained)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_performance_comparison_metrics.png')
plt.close()
print(" - Saved: Model Performance Comparison Plot as 'model_performance_comparison_metrics.png'")

# Save results to CSV
results_df.to_csv('model_performance_metrics.csv', index=False)
print("\nLabel: Saving Model Performance Results")
print(" - Saved: Model performance metrics saved to 'model_performance_metrics.csv'")

# Without specific results, we infer that Linear Regression likely has the highest accuracy (highest R² Score, lowest MAE/MSE/RMSE) 
# if the dataset shows linear relationships, due to its simplicity and the preprocessing steps. 
# Decision Tree likely has the lowest accuracy due to overfitting. 
# Other models (SVR, SGD, MLP) fall in between, limited by lack of tuning or complexity mismatch with the dataset. 
# To confirm, you’d need to check the model_performance_metrics.csv file or the printed results_df for exact R² Scores and error metrics. 
# If you can share those results or the correlation heatmap insights, I can provide a more precise analysis!
