import csv
from bs4 import BeautifulSoup
from datetime import datetime
import re
import time
from multiprocessing import Pool

def get_video_link(soup):
    video_link_tag = soup.find('a', href=re.compile(r'^https://www.youtube.com/watch\?v='))
    return video_link_tag.get('href') if video_link_tag else None

def get_video_title(soup):
    video_title_tag = soup.find('a', href=re.compile(r'^https://www.youtube.com/watch\?v='))
    return " " + video_title_tag.text.strip() if video_title_tag else None

def get_channel_link(soup):
    channel_link_tag = soup.find('a', href=re.compile(r'^https://www.youtube.com/channel/'))
    return channel_link_tag.get('href') if channel_link_tag else None

def get_channel_name(soup):
    channel_name_tag = soup.find('a', href=re.compile(r'^https://www.youtube.com/channel/'))
    return " " + channel_name_tag.text.strip() if channel_name_tag else None

def get_watch_date_time(soup):
    date_time_str = soup.find(string=re.compile(r'^\w{3} \d{1,2}, \d{4}, \d{1,2}:\d{2}:\d{2}'))
    if date_time_str:
        date_time_str = date_time_str.replace('\u202f', '').strip()
        date_time_str = re.sub(r'\s[^\s]+$', '', date_time_str)
        date_time_obj = datetime.strptime(date_time_str, '%b %d, %Y, %I:%M:%S%p')
        military_time_str = date_time_obj.strftime('%b %d, %Y, %H:%M:%S')
        return military_time_str
    else:
        return None

def process_div(div):
    soup = BeautifulSoup(div, 'html.parser')

    video_link = get_video_link(soup)
    video_title = get_video_title(soup)
    channel_link = get_channel_link(soup)
    channel_name = get_channel_name(soup)
    watch_date_time = get_watch_date_time(soup)
    
    return {
        'video_link': video_link,
        'video_title': video_title,
        'channel_link': channel_link,
        'channel_name': channel_name,
        'watch_date_time': watch_date_time,
    }

def process_div_wrapper(args):
    return process_div(args)

def main():
    start_time = time.time()
    html_file_path = 'partial_watch_history.html'

    # Define the chunk size for reading the file
    chunk_size = 1024 * 1024  # 1 MB

    data_list = []

    # Define the number of processes to use
    num_processes = 4

    # My YouTube watch history file is so big I need to process it in chunks
    with open(html_file_path, 'r', encoding='utf-8') as file: 
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break

            soup = BeautifulSoup(chunk, 'html.parser')
            divs = soup.find_all('div', class_='outer-cell')

            with Pool(num_processes) as pool:
                data_list.extend(pool.map(process_div_wrapper, map(str, divs)))

    # Save data to CSV
    csv_file_path = 'output.csv'
    header = ['video_link', 'channel_link', 'video_title', 'channel_name', 'watch_date_time']

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(data_list)

    print(f'Data has been saved to {csv_file_path}')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Total elapsed time: {elapsed_time:.2f} seconds')

if __name__ == "__main__":
    main()

