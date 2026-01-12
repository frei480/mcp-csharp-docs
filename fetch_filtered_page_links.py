import re
from urllib.parse import urljoin, urlparse
import asyncio
import httpx
from bs4 import BeautifulSoup

http_client = httpx.AsyncClient()
# @mcp.tool()
async def fetch_filtered_page_links(base_url: str, current_page_url: str) -> list[str]:
    """
    Returns a list of relevant internal URLs found on the given page,
    prioritizing API/SDK documentation links.
    """
    resp = await http_client.get(current_page_url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    all_links = []
    base_domain = urlparse(base_url).netloc

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        absolute_url = urljoin(current_page_url, href)
        parsed_url = urlparse(absolute_url)

        # 1. Filter by domain: Only include links within the same domain
        if parsed_url.netloc == base_domain:
            # 2. Further filtering based on path structure (adjust regex for your site)
            # Example: prioritize links that look like API documentation or examples
            # if re.search(r'/(api|sdk|examples)/', parsed_url.path, re.IGNORECASE):
            all_links.append(absolute_url)
            # elif re.search(r'/(classes|methods|properties)/', parsed_url.path, re.IGNORECASE):
            # all_links.append(absolute_url)
            # You might want to add other conditions or a default for general internal links
            # else:
            #     all_links.append(absolute_url) # Include other internal links if needed

    # You could also add logic here to deduplicate and optionally sort/prioritize links
    return list(set(all_links))
if __name__ == "__main__":
    test_base_url = "http://localhost:8080/html/"
    test_page_url = "http://localhost:8080/html/R_Project_TFlexAPI.htm"
    test_page_url = "http://localhost:8080/html/T_TFlex_Model_Model2D_Area.htm"
    links = asyncio.run(fetch_filtered_page_links(test_base_url, test_page_url))
    import pprint
    for link in links:
        pprint.pprint(link)