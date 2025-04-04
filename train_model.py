import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Sample dataset with 'age', 'gender', and 'favorite_music_genre'
data = {
    'age': [25, 30, 35, 40, 45, 50, 55, 60],
    'gender': ['male', 'female', 'female', 'male', 'male', 'female', 'female', 'male'],
    'favorite_music_genre': ['Rock', 'Pop', 'Jazz', 'Rock', 'Classical', 'Pop', 'Jazz', 'Classical']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Perform one-hot encoding on 'gender' and 'favorite_music_genre'
df = pd.get_dummies(df, columns=['gender', 'favorite_music_genre'], drop_first=True)

# Features (X) and target (y)
X = df.drop(columns=['favorite_music_genre_Rock', 'favorite_music_genre_Jazz', 'favorite_music_genre_Pop'])
y = df['favorite_music_genre_Classical']  # Assume Classical is our target genre for prediction

# Split data into training and testing (for simplicity, we train on the whole dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model to a file using joblib
joblib.dump(model, 'music_genre_predictor_without_encoder.joblib')

print("Model trained and saved successfully!")
