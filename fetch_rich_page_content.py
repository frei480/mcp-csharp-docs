import re
import httpx
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import  Any


http_client = httpx.AsyncClient(timeout=120.0)

# @mcp.tool()
async def fetch_rich_page_content(url: str) -> dict[str, Any]:

    try:
        resp = await http_client.get(url, timeout=15)
        resp.raise_for_status()
    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}: {exc}")
        return {
            "url": url, "error": str(exc), "title": "Error Page",
            "general_text": [], "code_examples": [], "object_descriptions": [],
            "potential_entity_links": [], "structured_content": []
        }
    except httpx.HTTPStatusError as exc:
        print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
        return {
            "url": url, "error": f"HTTP Error {exc.response.status_code}", "title": "Error Page",
            "general_text": [], "code_examples": [], "object_descriptions": [],
            "potential_entity_links": [], "structured_content": []
        }

    soup = BeautifulSoup(resp.text, "html.parser")

    content: dict[str, Any] = {
        "url": url,
        "title": soup.title.string.strip() if soup.title else "No Title",
        "general_text": [],
        "code_examples": [],
        "object_descriptions": [],
        "potential_entity_links": [],
        "structured_content": [],
        "metadata": {
            "description": soup.find('meta', attrs={'name': 'description'})['content']
                           if soup.find('meta', attrs={'name': 'description'}) else ""
        }
    }

    # --- 1. Извлечение заголовков и общего текста ---
    for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'li', 'td', 'th']):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            if len(text) > 5 and not any(text.lower().startswith(nav_item) for nav_item in ["home", "about", "contact"]):
                content["general_text"].append(text)
            
            if tag.name.startswith('h'):
                content["structured_content"].append({"type": "heading", "level": int(tag.name[1]), "text": text})
            elif tag.name == 'p':
                content["structured_content"].append({"type": "paragraph", "text": text})

    # --- 2. Извлечение блоков кода ---
    for code_container in soup.find_all(['pre', 'code']):
        code = code_container.get_text(separator="\n", strip=True)
        if code:
            is_csharp = False
            csharp_keywords = ["public", "class", "void", "string", "int", "namespace", "using", "return", "this"]
            if any(keyword in code for keyword in csharp_keywords):
                is_csharp = True
            
            if code_container.has_attr('class') and any('csharp' in cls.lower() for cls in code_container['class']):
                is_csharp = True

            if is_csharp:
                context_around_code = []
                previous_sibling = code_container.find_previous_sibling(['p', 'h1', 'h2', 'h3', 'h4', 'div'])
                if previous_sibling and previous_sibling.get_text(strip=True):
                    context_around_code.append(previous_sibling.get_text(strip=True))
                
                content["code_examples"].append({
                    "code": code,
                    "context": " ".join(context_around_code).strip(),
                    "language": "csharp"
                })

    # --- 3. Извлечение описаний объектов ---
    for desc_div in soup.find_all('div', class_=['api-description', 'member-description', 'summary-text']):
        description_text = desc_div.get_text(separator="\n", strip=True)
        if description_text and len(description_text) > 20:
            content["object_descriptions"].append(description_text)
            content["structured_content"].append({"type": "object_description", "text": description_text})

    for heading in soup.find_all(['h2', 'h3', 'h4']):
        heading_text = heading.get_text(strip=True).lower()
        if "summary" in heading_text or "remarks" in heading_text or "description" in heading_text:
            next_sibling = heading.find_next_sibling(['p', 'div'])
            if next_sibling and next_sibling.get_text(strip=True):
                description_text = next_sibling.get_text(separator="\n", strip=True)
                if description_text and len(description_text) > 50:
                    content["object_descriptions"].append(description_text)
                    content["structured_content"].append({"type": "related_description", "heading": heading_text, "text": description_text})

    # --- 4. Специальная логика для таблиц-оглавлений сущностей () ---

    regex = r"<tr[^>]*?>.*?<a href=\"(.*?)\">(.*?)</a></td><td>(.*?)</td></tr>"
    matches = re.finditer(regex, resp.text, re.MULTILINE)
    for match in matches:
        # Добавляем в potential_entity_links
        entity_name = match.group(2).strip()
        entity_url_abs =urljoin(url,  match.group(1))
        description_text = match.group(3).strip()
        content["potential_entity_links"].append({
            "name": entity_name,
            "url": entity_url_abs,
            "summary": description_text if description_text else "Description likely on linked page.",
            "context": "Found in an API entity listing table."
        })

    # Очистка и дедупликация general_text
    content["general_text"] = list(dict.fromkeys(content["general_text"]))

    return content

if __name__ == "__main__":
    test_url = "http://localhost:8080/html/T_TFlex_Model_Model2D_Area.htm"
    test_url = "http://localhost:8080/html/R_Project_TFlexAPI.htm"
    content = asyncio.run(fetch_rich_page_content(test_url))
    import pprint
    pprint.pprint(content)