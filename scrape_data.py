import csv
import requests
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime, timedelta


def parse_iso8601_duration(duration):
    # Parse ISO 8601 duration format (e.g., PT2M9S)
    duration = duration[2:]  # Remove 'PT' prefix
    total_seconds = 0

    # Split duration into parts
    parts = {'H': 3600, 'M': 60, 'S': 1}
    current_part = ''
    for char in duration:
        if char.isnumeric():
            current_part += char
        elif char in parts:
            if current_part:
                total_seconds += int(current_part) * parts[char]
                current_part = ''
        else:
            raise ValueError(f"Unexpected character in duration: {char}")

    return max(0, total_seconds - 1)  # Adjust for 1-second discrepancy

def print_progress(video_count, total_videos, start_time):
    videos_per_second = video_count / (time.time() - start_time)
    remaining_seconds = (total_videos - video_count) / videos_per_second
    remaining_time = timedelta(seconds=remaining_seconds)
    print(f"Processed {video_count} videos in {time.time() - start_time:.2f} seconds. Estimated time remaining: {remaining_time}.")

# Function to scrape YouTube video details
def scrape_youtube_video_details(video_link):
    video_details = {}
    # Send a GET request to the YouTube video page
    response = requests.get(video_link)
    soup = BeautifulSoup(response.text, 'html.parser')


    # Extract view count
    view_count_match = re.search(r'"simpleText":"(\d+\.\d+)[^"]*görüntüleme"', str(soup))
        
    if view_count_match:
        view_count_str = view_count_match.group(1)
        view_count_str = view_count_str.replace('.', '')  # Remove period
        video_details['view_count'] = int(view_count_str)
    else:
        video_details['view_count'] = 'NA'


    # Extract like count
    like_count_match = re.search(r'"iconName":"LIKE","title":"([^"]+)"', str(soup))

    if like_count_match:
        like_count_str = like_count_match.group(1)

        # Remove non-breaking spaces
        like_count_str = like_count_str.replace('\xa0', '')

        # Function to convert like count string to numeric value
        def convert_like_count(like_count_str):
            multiplier = 1
            if 'K' or 'B' in like_count_str:
                multiplier = 1000
            elif 'M' in like_count_str:
                multiplier = 1000000

            # Extract digits
            digits = ''.join(filter(str.isdigit, like_count_str))

            if digits:
                return int(digits) * multiplier
            else:
                return 0  # or any default value you prefer if there are no digits


        like_count_numeric = convert_like_count(like_count_str)
        video_details['like_count'] = like_count_numeric
    else:
        video_details['like_count'] = 'NA'


    # Extract comment count
    comment_count_match = re.search(r'"engagementPanelTitleHeaderRenderer"\s*:\s*{"title"\s*:\s*{"runs"\s*:\s*\[{"text"\s*:\s*"Yorumlar"\s*}\]}\s*,\s*"contextualInfo"\s*:\s*{"runs"\s*:\s*\[{"text"\s*:\s*"([\d,]+)"\s*}\]}', str(soup))


    if comment_count_match:
        comment_count_str = comment_count_match.group(1).replace(',', '')  # Remove commas

        # Check for 'B' and convert to integer accordingly
        if 'B' in comment_count_str:
            comment_count_str = comment_count_str.replace('B', '')
            comment_count = int(float(comment_count_str) * 1000)
        else:
            comment_count = int(comment_count_str)

        video_details['comment_count'] = comment_count
    else:
        video_details['comment_count'] = 'NA'


    # Extract video description
    description_match = re.search(r'"attributedDescription":{"content":"(.+?)"', str(soup))
    if description_match:
        video_details['description'] = description_match.group(1)
        # Calculate description length
        video_details['description_length'] = len(video_details['description'])
    else:
        video_details['description'] = 'NA'
        video_details['description_length'] = 0


    # Extract video tags
    tags_element = soup.find_all('meta', attrs={'property': 'og:video:tag'})
    video_details['tags'] = [tag['content'] for tag in tags_element] if tags_element else []


    # I noticed that you can tell if a video is YouTube shorts or not by checking if the resolution is swapped:
    # for example, a 1080p video would be like width:1920, height:1080, but a 1080p short would be like width:1080, height:1920
    height_match = re.search(r'mimeType":"video/webm; codecs=\\"vp9\\"","bitrate":\d+,"width":(\d+),"height":(\d+)', str(soup))

    if height_match:
        width = height_match.group(1)
        height = height_match.group(2)

        # Assuming common aspect ratios for regular videos
        common_resolutions = [
            (3840, 2160),
            (2560, 1440),
            (1920, 1080),
            (1280, 720),
            (640, 360),
            (256, 144)
        ]

        # Check if the dimensions match common resolutions for regular videos
        is_shorts = (int(width), int(height)) not in common_resolutions

        if is_shorts:
            video_details['is_shorts'] = True
            video_details['video_quality'] = width + 'p'  # Using width for shorts
        else:
            video_details['is_shorts'] = False
            video_details['video_quality'] = height + 'p'  # Using height for regular videos
    else:
        video_details['is_shorts'] = False
        video_details['video_quality'] = 'NA'


    # Extract category
    category_tag = soup.find('meta', itemprop='genre')
    if category_tag:
        category = category_tag.get('content')
        video_details['category'] = category
    else:
         video_details['category'] = 'NA'


    # Extract publish_date_iso8601
    publish_date_tag = soup.find('meta', itemprop='uploadDate')
    if publish_date_tag:
        # Convert ISO 8601 to desired format
        publish_date_iso8601 = publish_date_tag.get('content')
        publish_date_obj = datetime.strptime(publish_date_iso8601, '%Y-%m-%dT%H:%M:%S%z')
        publish_date_formatted = publish_date_obj.strftime('%b %d, %Y, %H:%M:%S')
        video_details['publish_date'] = publish_date_formatted
    else:
        video_details['publish_date'] = "NA"


    # Extract duration
    duration_tag = soup.find('meta', itemprop='duration')
    
    if duration_tag:
        duration_iso8601 = duration_tag.get('content')
        # Convert duration to minutes and seconds
        duration_seconds = parse_iso8601_duration(duration_iso8601)
        duration_minutes = duration_seconds // 60
        duration_seconds %= 60
        video_length = f'{duration_minutes:02d}:{duration_seconds:02d}'
        video_details['video_length'] = video_length
    else:
        video_details['video_length'] = "NA"


    # Extract subscription status using subscriptions.csv file YouTube Takeout includes
    channel_id = row.get('channel_link', '').replace('https://www.youtube.com/channel/', '')
    video_details['subscribed'] = any(sub['Channel Id'] == channel_id for sub in subscriptions)

    print("Processed", video_link)
    return video_details

start_time = time.time()

# Read CSV files, scrape details for each video, and save to a new file
subscriptions_csv_file = 'subscriptions.csv'
with open(subscriptions_csv_file, 'r', encoding='utf-8') as subs_file:
        subscriptions = list(csv.DictReader(subs_file))

with open('output.csv', 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    total_videos = 53296
    
    # Define new column names
    new_columns = ['subscribed', 'video_length', 'view_count', 'like_count', 'comment_count', 'description', 'description_length', 'category', 'tags', 'video_quality', 'is_shorts', 'publish_date']

    # Create a new CSV file for writing
    with open('output2.csv', 'w', newline='', encoding='utf-8') as output_csv:
        # Combine existing and new column names
        fieldnames = csv_reader.fieldnames + new_columns
        csv_writer = csv.DictWriter(output_csv, fieldnames=fieldnames)

        # Write header to the new CSV file
        csv_writer.writeheader()

        # Iterate through each row in the original CSV file
        idx = 0
        for row in csv_reader:
            video_link = row['video_link']
            if video_link:
                video_details = scrape_youtube_video_details(video_link)

                # Update the row with scraped details
                row.update(video_details)

                # Write the updated row to the new CSV file
                csv_writer.writerow(row)

                # Print progress every 10 videos processed
                if (idx + 1) % 10 == 0:
                    print_progress(idx + 1, total_videos, start_time)
                idx = idx + 1
       
        print_progress(total_videos, total_videos, start_time)

print("Scraping and updating completed. Results saved to output2.csv.")
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Total elapsed time: {elapsed_time:.2f} seconds')
