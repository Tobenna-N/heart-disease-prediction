import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_preprocessing import load_and_preprocess_data
import warnings

warnings.filterwarnings("ignore")

def get_user_input(columns, label_encoders, scaler):
    
    user_data = {}
    
    for column in columns:
        if column == 'target':  # Skip the target column
            continue
        
        if column in label_encoders:
            unique_values = list(label_encoders[column].classes_)
            user_input = input(f"Enter {column} (choose from {', '.join(unique_values)}): ")
            # Ensure that the input is valid
            while user_input not in unique_values:
                print(f"Invalid input. Please choose a valid value from {', '.join(unique_values)}.")
                user_input = input(f"Enter {column} (choose from {', '.join(unique_values)}): ")
            user_data[column] = label_encoders[column].transform([user_input])[0]
        else:
            while True:
                try:
                    user_input = float(input(f"Enter {column}: "))
                    user_data[column] = user_input
                    break  # Break loop if the float is valid
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
    
    user_df = pd.DataFrame([user_data])
    user_scaled = scaler.transform(user_df)
    return user_scaled

file_path = r"C:\Users\offic\OneDrive\Desktop\AMOD-5230\project\HeartDiseaseTrain-Test.csv"

# Load and preprocess the data
X_train, X_test, y_train, y_test, columns, label_encoders, scaler = load_and_preprocess_data(file_path)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Get the user input and make a prediction
user_scaled = get_user_input(columns, label_encoders, scaler)
user_prediction = model.predict(user_scaled)

print("Prediction: High risk of heart disease." if user_prediction[0] == 1 else "Prediction: Low risk of heart disease.")
