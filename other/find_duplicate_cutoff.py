import csv

# I ran my processing scripts on two machines, my laptop and my linux cloud computer
# This script was needed to find the optimal merging point of the processed outputs (which are different halves of all data)

def find_first_difference(video_link, csv_file_path1, csv_file_path2):
    with open(csv_file_path1, 'r', encoding='utf-8') as file1, open(csv_file_path2, 'r', encoding='utf-8') as file2:
        csv_reader1 = csv.DictReader(file1)
        csv_reader2 = csv.DictReader(file2)

        # Find the index of the video link in both files
        idx1 = next((idx for idx, row in enumerate(csv_reader1) if row['video_link'] == video_link), None)
        idx2 = next((idx for idx, row in enumerate(csv_reader2) if row['video_link'] == video_link), None)

        if idx1 is not None and idx2 is not None:
            # Iterate from the found index to find the first difference
            for _ in range(idx1):
                next(csv_reader1)
            for _ in range(idx2):
                next(csv_reader2)

            # Iterate and check rows for equality
            while True:
                try:
                    row1 = next(csv_reader1)
                    row2 = next(csv_reader2)
                    if row1 != row2:
                        return row1, row2
                except StopIteration:
                    # One of the files reached the end
                    return None
        else:
            return None

# Specify the video link
video_link_to_find = 'https://www.youtube.com/watch?v=y8UA2XMqmKU'

# Specify the paths to the CSV files (output2.csv and output3.csv)
csv_file_path_output2 = 'output2.csv'
csv_file_path_output3 = 'output3.csv'

# Find the first difference between rows
first_difference = find_first_difference(video_link_to_find, csv_file_path_output2, csv_file_path_output3)

if first_difference:
    row_output2, row_output3 = first_difference
    print("Rows are different starting from the following indices:")
    print(f"Output2.csv: {row_output2}")
    print(f"Output3.csv: {row_output3}")
else:
    print("No difference found.")
