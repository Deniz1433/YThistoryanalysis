import pandas as pd
import numpy as np

# There were somehow is_shorts = True videos that were over 60 seconds?? That's impossible so I set them to False

# Function to convert video length to total seconds
def convert_to_seconds(time_str):
    if pd.isna(time_str):
        return np.nan
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

# Read the CSV file into a DataFrame
file_path = 'output2.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Convert video length to total seconds
df['video_length_seconds'] = df['video_length'].apply(convert_to_seconds)

# Set is_shorts to False for videos longer than 60 seconds
df.loc[df['video_length_seconds'] > 60, 'is_shorts'] = False

# Drop the 'video_length_seconds' column if you don't need it anymore
df.drop('video_length_seconds', axis=1, inplace=True)

# Save the updated DataFrame to a new CSV file
df.to_csv('output2_updated.csv', index=False)
