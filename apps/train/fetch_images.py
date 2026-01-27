import requests
import pydash as _
from tqdm import tqdm
import time

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


start_cursor = "'Pz9DZ0FCQUFBQm5BRTIwb29JQUFJQUFBQ1FBZ0FFQUFnQUJnQUFBQUFBfDE0OTE4NTIxMDg4OTYwMzYqR1FMKnwzY2E3ZTdjNzI3MDgzOGZiYzY4YWI2ZjViZTgwYTk2YTJkZDZlY2UwODRiNmIxYjJhODJjYzA4MmQ2MDA0ODFlfE5FV3w='"

for i in tqdm(range(100), desc="Processing pages"):
    json_data = {
    'queryHash': '59af1c03833723bc1034efb436ee06043acb5feb561d94f64c0af517fd388c50',
    'variables': {
        'contextPinIds': [
            '107453141105427087',
        ],
        'count': 12,
        'cursor': 'Pz9DZ0FCQUFBQm5BRS9WajBJQUFJQUFBQU1BZ0FFQUFnQUJnQUFBQUFBfDI5MDc3NTYzMjE5MTYxOTQqR1FMKnwyMTI1NzhjMmQxMzNjMzdlZWI1OWI5ZjBkMzNmOWU3NThkZjMyYzVkN2ViOGQyMmE5ZDJhODBkNjM0OTcxODYyfE5FV3w=',
        'isAuth': False,
        'isCloseupRelatedModulesFeedQuery': True,
        'isDesktop': True,
        'pinId': '107453141105427087',
        'searchQuery': None,
        'shouldPrefetchCloseupPreviewFragment': False,
        'source': None,
        'topLevelSource': None,
        'topLevelSourceDepth': None,
    },
}

    response = requests.post('https://www.pinterest.com/_/graphql/', cookies=cookies, headers=headers, json=json_data)

    json_data = response.json()
    edges = _.get(json_data, 'data.v3RelatedPinsForPinSeoQuery.data.connection.edges')
    if not edges:
        break
    image_urls = []
    for edge in edges:
        image_urls.append(_.get(edge, 'node.images_orig.url'))

    with open("image_urls.txt", "a") as f:
        for image_url in image_urls:
            f.write(image_url + "\n")
            
    start_cursor = _.get(json_data, 'data.v3RelatedPinsForPinSeoQuery.data.connection.pageInfo.endCursor')
    has_next_page = _.get(json_data, 'data.v3RelatedPinsForPinSeoQuery.data.connection.pageInfo.hasNextPage')
    if not has_next_page:
        break
    time.sleep(1)