# Deals Server

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Requests](https://img.shields.io/badge/Requests-2B6CB0?style=for-the-badge&logo=python&logoColor=white)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-59666D?style=for-the-badge&logo=python&logoColor=white)

This application retrieves deal information. It uses the `requests` library to send HTTP requests and `BeautifulSoup` to parse the HTML content. The application also uses the `dotenv` library to manage environment variables.

## Description

The application is designed to search the web using the SerpApi. It takes a query as input and returns the search results. The search function sends a POST request to the SerpApi with the query as payload. The response from the SerpApi is then printed to the console.

## Usage

To use this application, you need to set the `BROWSERLESS_API_KEY` and `SERPER_API_KEY` in your environment variables. These keys are used to authenticate your requests to the SerpApi.

```python
def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
    })

    headers = {
        'x-api-key': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)