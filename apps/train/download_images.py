import os
import requests
import time
from tqdm import tqdm
import random

image_folder = "images_sydney_sweeney"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

with open("google_image_urls.txt", "r") as f:
    image_urls = f.readlines()

image_urls = [image_url.strip() for image_url in image_urls]

unique_image_urls = list(set(image_urls))


cookies = {
    'csrftoken': 'f147c2065ecf6e29adfe2727fe6e7a58',
    '_pinterest_sess': 'TWc9PSZoTEJpSHVrUzZDZ0xuZk5oZUJ1ejZScWphV0VXd0pLNjNSeE02UW1rL0t3ZjJIaUhvR2pYTmRVYjhJYVAzN3AvTjRzWHI4Q2VlbVA4TDNjZitBZW96M0V2NnFqM05Cdm5TQ3NlU1QzQWJYQT0mS3c2dEcreWFSclJ0SytXOFdoNEZCQlIzVXJjPQ==',
    '_auth': '0',
    '_routing_id': '"dc0c6999-9a54-4768-99ba-aa5992cfc4d9"',
    'sessionFunnelEventLogged': '1',
    'g_state': '{"i_l":0,"i_ll":1769547453826,"i_b":"DewIy7wqq8Shefh+P0HNTxLObVPs54MU7HrcX2CORhI","i_e":{"enable_itp_optimization":3}}',
}

headers = {
    'accept': 'application/json',
    'accept-language': 'en-US,en;q=0.9',
    'content-type': 'application/json',
    'origin': 'https://www.pinterest.com',
    'priority': 'u=1, i',
    'referer': 'https://www.pinterest.com/',
    'sec-ch-ua': '"Not(A:Brand";v="8", "Chromium";v="144"',
    'sec-ch-ua-full-version-list': '"Not(A:Brand";v="8.0.0.0", "Chromium";v="144.0.7559.97"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-model': '""',
    'sec-ch-ua-platform': '"macOS"',
    'sec-ch-ua-platform-version': '"15.4.1"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36',
    'x-csrftoken': 'f147c2065ecf6e29adfe2727fe6e7a58',
    'x-pinterest-appstate': 'active',
    'x-pinterest-graphql-name': 'UnauthCloseupRelatedPinsFeedPaginationQuery',
    'x-pinterest-pws-handler': 'www/pin/[id].js',
    'x-pinterest-source-url': '/pin/13059-sydney-sweeney-photos-high-res-pictures-getty-images-in-2025--107453141105427087/',
    'x-requested-with': 'XMLHttpRequest',
    # 'cookie': 'csrftoken=f147c2065ecf6e29adfe2727fe6e7a58; _pinterest_sess=TWc9PSZoTEJpSHVrUzZDZ0xuZk5oZUJ1ejZScWphV0VXd0pLNjNSeE02UW1rL0t3ZjJIaUhvR2pYTmRVYjhJYVAzN3AvTjRzWHI4Q2VlbVA4TDNjZitBZW96M0V2NnFqM05Cdm5TQ3NlU1QzQWJYQT0mS3c2dEcreWFSclJ0SytXOFdoNEZCQlIzVXJjPQ==; _auth=0; _routing_id="dc0c6999-9a54-4768-99ba-aa5992cfc4d9"; sessionFunnelEventLogged=1; g_state={"i_l":0,"i_ll":1769547453826,"i_b":"DewIy7wqq8Shefh+P0HNTxLObVPs54MU7HrcX2CORhI","i_e":{"enable_itp_optimization":3}}',
}


def download_image(image_url):
    try:
       
        response = requests.get(image_url, headers=headers, cookies=cookies)
        return response.content
    except Exception as e:
        print(f"Error downloading image {image_url}: {e}")
        return None

for i, image_url in tqdm(enumerate(unique_image_urls), desc="Downloading images"):
    image_url = image_url.strip()
    if not image_url:
        continue

    base_name = os.path.basename(image_url)
    file_path = os.path.join(image_folder, f"{i}_{base_name}")
    if os.path.exists(file_path):
        continue
    image_content = download_image(image_url)
    if image_content:
        with open(file_path, "wb") as f:
            f.write(image_content)
    else:
        print(f"Error downloading image {image_url}")
        
    time.sleep(random.randint(0, 1))