import pandas as pd

#I made the big mistake of returning 0 seconds instead of NA when I couldn't extract like count
#This script is to set the 0's to NA's so I can avoid using them in the calculations

# Read the CSV file into a DataFrame
input_file = 'output2.csv'
df = pd.read_csv(input_file)

# Replace 0 values in the "like_count" column with 'NA'
df['like_count'] = df['like_count'].replace(0, 'NA')

# Save the modified DataFrame to a new CSV file
output_file = 'output3.csv'
df.to_csv(output_file, index=False)

print(f"Modified data saved to {output_file}")
