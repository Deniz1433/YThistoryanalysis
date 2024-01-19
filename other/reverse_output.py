import csv

# I ran my processing scripts on two machines, my laptop and my linux cloud computer
# I ran one in reverse so I had to reverse its output before I merged the two files
# It took 4 hours to process instead of 8 thanks to the 2 machines meeting up at the middle

# Read output2_reversed.csv in reversed order
with open('output2_reversed.csv', 'r', encoding='utf-8') as reverse_csv_file:
    reverse_csv_reader = csv.DictReader(reverse_csv_file)
    reversed_output = list(reversed([row for row in reverse_csv_reader]))

# Write the reversed output to output3.csv
with open('output3.csv', 'w', newline='', encoding='utf-8') as output3_csv_file:
    output3_fieldnames = reversed_output[0].keys()
    output3_csv_writer = csv.DictWriter(output3_csv_file, fieldnames=output3_fieldnames)

    # Write header to the new CSV file
    output3_csv_writer.writeheader()

    # Write rows to the new CSV file
    output3_csv_writer.writerows(reversed_output)
