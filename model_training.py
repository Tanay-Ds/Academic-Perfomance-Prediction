import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
file_path = 'StudentPerformanceFactors.csv'
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: File {file_path} not found.")
    exit()

# Separate features and target
X = df.drop('Exam_Score', axis=1)
y = df['Exam_Score']

# Identify categorical and numerical columns
# Explicitly define columns to avoid type inference issues
categorical_cols = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 
                   'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 
                   'School_Type', 'Peer_Influence', 'Learning_Disabilities', 
                   'Parental_Education_Level', 'Distance_from_Home', 'Gender']

numerical_cols = ['Hours_Studied/Week', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 
                 'Tutoring_Sessions', 'Physical_Activity']

print(f"Categorical columns: {list(categorical_cols)}")
print(f"Numerical columns: {list(numerical_cols)}")

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create and evaluate the pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R2 Score: {r2}")

# Save model
# Save model
joblib.dump(clf, 'student_performance_model_v2.pkl')
print("Model saved as 'student_performance_model_v2.pkl'")

# Save the column names to help with the app form creation
joblib.dump(list(X.columns), 'feature_names.pkl')
joblib.dump(list(categorical_cols), 'categorical_cols.pkl')
joblib.dump(list(numerical_cols), 'numerical_cols.pkl')

# Also save unique values for categorical columns for dropdowns
unique_values = {col: df[col].dropna().unique().tolist() for col in categorical_cols}
joblib.dump(unique_values, 'categorical_unique_values.pkl')
print("Feature metadata saved.")
