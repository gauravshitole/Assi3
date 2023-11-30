import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = {
    'company': [
        'google', 'google', 'google', 'google', 'google', 'google',
        'abc pharma', 'abc pharma', 'abc pharma', 'abc pharma',
        'facebook', 'facebook', 'facebook', 'facebook', 'facebook', 'facebook'
    ],
    'job': [
        'sales executive', 'sales executive', 'business manager', 'business manager', 'computer programmer', 'computer programmer',
        'sales executive', 'computer programmer', 'business manager', 'business manager',
        'sales executive', 'sales executive', 'business manager', 'business manager', 'computer programmer', 'computer programmer'
    ],
    'degree': [
        'bachelors', 'masters', 'bachelors', 'masters', 'bachelors', 'masters',
        'masters', 'bachelors', 'bachelors', 'masters',
        'bachelors', 'masters', 'bachelors', 'masters', 'bachelors', 'masters'
    ],
    'salary_more_then_100k': [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Encode categorical variables using Label Encoding
label_encoder = LabelEncoder()
df['company'] = label_encoder.fit_transform(df['company'])
df['job'] = label_encoder.fit_transform(df['job'])
df['degree'] = label_encoder.fit_transform(df['degree'])
# Separate the features (X) and the target variable (y)
X = df.drop('salary_more_then_100k', axis=1)
y = df['salary_more_then_100k']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classification model
decision_tree = DecisionTreeClassifier()

# Fit the model to the training data
decision_tree.fit(X_train, y_train)

# Make predictions on the test data
y_pred = decision_tree.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the evaluation results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)


