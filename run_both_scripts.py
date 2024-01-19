import time
import subprocess

# You need to have these files in the same folder as this script:
# partial_watch_history.html
# subscriptions.csv
# convert_to_csv_mp.py
# scrape_data.py


def run_script(command):
    print(f"Running command: {command}")
    
    # Run the script and capture the output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Read and print the output line by line
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    # Wait for the process to finish
    process.wait()

    # Print the exit code
    print(f"Command completed with exit code: {process.returncode}\n")

if __name__ == "__main__":
    print("Running script #1")
    run_script(["python", "convert_to_csv_mp.py"])

    print("Script #1 is done, wait 5 seconds...")
    time.sleep(5)

    print("Running script #2")
    run_script(["python", "scrape_data.py"])

    print("Done")

# Final output is output2.csv
