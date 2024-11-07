import os
import json
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime

# Load Cricsheet data
data_dir = 'path/to/cricsheet/data'
match_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
all_matches = []
for file in match_files:
    with open(os.path.join(data_dir, file), 'r') as f:
        match_data = json.load(f)
        all_matches.append(match_data)
df = pd.DataFrame(all_matches)

# Preprocess the data
df = df.dropna(subset=['info.teams', 'innings.1.deliveries'])
df['info.teams'] = df['info.teams'].apply(lambda x: '-'.join(x))
df['info.venue'] = df['info.venue'].astype(str)
df['match_date'] = pd.to_datetime(df['info.dates'].apply(lambda x: x[0]))
df['match_type'] = df['info.tournament'].str.extract(r'(\w+)', expand=False)
df['innings_length'] = df.groupby(['id'])['innings.1.deliveries'].transform('count')

# Define the fantasy points system
fantasy_points = {
    'runs': 1,
    'fours': 2,
    'sixes': 6,
    'wickets': 25,
    'catches': 10,
    'stumpings': 15,
    'runouts': 10
}

def get_dream_team(row):
    # Compute the dream team based on the fantasy points system
    player_points = []
    for i in range(1, 12):
        player = row[f'player{i}']
        runs = row[f'{player}_runs']
        fours = row[f'{player}_fours']
        sixes = row[f'{player}_sixes']
        wickets = row[f'{player}_wickets']
        catches = row[f'{player}_catches']
        stumpings = row[f'{player}_stumpings']
        runouts = row[f'{player}_runouts']
        player_points.append(
            runs * fantasy_points['runs'] +
            fours * fantasy_points['fours'] +
            sixes * fantasy_points['sixes'] +
            wickets * fantasy_points['wickets'] +
            catches * fantasy_points['catches'] +
            stumpings * fantasy_points['stumpings'] +
            runouts * fantasy_points['runouts']
        )
    return np.argsort(player_points)[-11:].tolist()

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Streamlit app
st.title("Model Performance Analysis")

# Input fields
st.subheader("Model Evaluation")
train_start = st.date_input("Training Period Start", value=pd.Timestamp("2000-01-01"))
train_end = st.date_input("Training Period End", value=pd.Timestamp("2024-06-30"))
test_start = st.date_input("Testing Period Start", value=pd.Timestamp("2024-08-01"))
test_end = st.date_input("Testing Period End", value=pd.Timestamp("2024-09-22"))

if st.button("Evaluate Model"):
    # Filter data based on input
    train_mask = (df['match_date'] >= train_start) & (df['match_date'] <= train_end)
    test_mask = (df['match_date'] >= test_start) & (df['match_date'] <= test_end)
    X_train = df.loc[train_mask, ['info.teams', 'info.venue', 'match_date', 'match_type', 'innings_length']]
    y_train = df.loc[train_mask, 'innings.1.deliveries']
    X_test = df.loc[test_mask, ['info.teams', 'info.venue', 'match_date', 'match_type', 'innings_length']]
    y_test = df.loc[test_mask, 'innings.1.deliveries']

    # Train the model
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Compute metrics and format output
    output_data = []
    for i, row in df.loc[test_mask].iterrows():
        predicted_team = np.argsort(y_pred[i])[-11:].tolist()
        dream_team = get_dream_team(row)
        mae = abs(sum(row[f'player{i+1}_points' for i in dream_team]) - sum(y_pred[i][predicted_team]))
        output_data.append({
            'match_date': row['match_date'],
            'team1': row['info.teams'].split('-')[0],
            'team2': row['info.teams'].split('-')[1],
            'predicted_team': ', '.join(map(str, predicted_team)),
            'dream_team': ', '.join(map(str, dream_team)),
            'predicted_points': sum(y_pred[i][predicted_team]),
            'mae': mae
        })

    # Display the model performance results
    st.subheader("Model Performance")
    performance_df = pd.DataFrame(output_data)
    st.dataframe(performance_df)

    # Save the model and training data
    model_path = os.path.join('model_artifacts', f'model_{train_end.strftime("%Y-%m-%d")}.pkl')
    os.makedirs('model_artifacts', exist_ok=True)
    joblib.save(model, model_path)
    st.write(f"Model saved to {model_path}")

    train_data_path = os.path.join('data', 'processed', f'training_data_{train_end.strftime("%Y-%m-%d")}.csv')
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    X_train.to_csv(train_data_path, index=False)
    st.write(f"Training data saved to {train_data_path}")

    # Save the output to a CSV file
    output_path = os.path.join('output', f'model_performance_{train_end.strftime("%Y-%m-%d")}.csv')
    os.makedirs('output', exist_ok=True)
    performance_df.to_csv(output_path, index=False)
    st.write(f"Output saved to {output_path}")