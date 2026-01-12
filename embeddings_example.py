import asyncio
import logging
import os
import re
from pathlib import Path

# MCP библиотеки
# Библиотеки для БД
import chromadb
import httpx

# --- КОНФИГУРАЦИЯ ---
from config import cfg

DOCS_PATH = cfg.DOCS_PATH
# URL локального AI движка на сервере (например, Ollama или LM Server)
AI_SERVER_URL = cfg.AI_SERVER_URL

# Модели (должны быть установлены на сервере)
RERANK_MODEL_NAME = cfg.RERANK_MODEL
EMBEDDING_MODEL_NAME = cfg.EMBEDDING_MODEL


CHROMA_PERSIST_DIR = cfg.CHROMA_PERSIST_DIR
COLLECTION_NAME = cfg.COLLECTION_NAME
REGISTRY_COLLECTION_NAME = cfg.REGISTRY_COLLECTION_NAME
N_RESULTS = 20
# Инициализация логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

http_client = httpx.AsyncClient(timeout=120.0)
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
registry_collection = chroma_client.get_or_create_collection(
    name=REGISTRY_COLLECTION_NAME
)


async def get_embedding(text: str) -> list[float]:
    if not text.strip():
        return []
    try:
        response = await http_client.post(
            f"{AI_SERVER_URL}/embeddings",
            json={"model": EMBEDDING_MODEL_NAME, "input": text},
        )
        response.raise_for_status()
        data = response.json()
        if "data" in data and len(data["data"]) > 0 and "embedding" in data["data"][0]:
            return data["data"][0]["embedding"]
        logging.error(f"Неожиданный формат ответа от сервера эмбеддингов: {data}")
        return []
    except httpx.HTTPStatusError as e:
        logging.error(
            f"HTTP ошибка при получении эмбеддинга: {e.response.status_code} - {e.response.text}"
        )
        return []
    except Exception as e:
        logging.error(
            f"Ошибка при получении эмбеддинга для текста '{text[:50]}...': {e}",
            exc_info=True,
        )
        return []


async def rerank_documents(query: str, documents: list[str]) -> list[int]:
    if not documents:
        return []
    if not RERANK_MODEL_NAME:
        logging.warning("RERANK_MODEL_NAME не установлен, пропускаем переранжирование.")
        return list(range(len(documents)))  # Возвращаем исходный порядок

    docs_snippets: list[str] = []
    for i, doc in enumerate(documents):
        # Ограничиваем сниппет до 400 символов, чтобы не перегружать reranker
        snippet = doc[:400] + "..." if len(doc) > 400 else doc
        docs_snippets.append(f"[{i}] {snippet}")
    docs_text = "\n".join(docs_snippets)
    logging.info(f"Получены документы: {docs_text}")
    # Более четкий системный промпт
    system_prompt = "You are a document re-ranker. Given a query and a list of document snippets, identify the most relevant document snippets. Return ONLY a comma-separated list of the indices of the top 5 most relevant documents from the provided list. Example: 1, 3, 0, 5, 2"
    user_prompt = (
        f"Query: {query}\n\nDocuments:\n{docs_text}\n\nTop 5 relevant indices:"
    )

    try:
        response = await http_client.post(
            f"{AI_SERVER_URL}/chat/completions",
            json={
                "model": RERANK_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 50,
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()

        indices: list[int] = []
        try:
            # Извлекаем все числа и преобразуем в int
            found_indices = [int(idx) for idx in re.findall(r"\d+", content)]
            # Фильтруем, чтобы индексы были в пределах допустимого диапазона
            indices = [idx for idx in found_indices if 0 <= idx < len(documents)]
            # Уникализируем и сохраняем порядок первого появления
            seen: set[int] = set()
            unique_indices = []
            for idx in indices:
                if idx not in seen:
                    unique_indices.append(idx)
                    seen.add(idx)
            indices = unique_indices[:5]  # Берем только топ-5
        except Exception as parse_e:
            logging.warning(
                f"Ошибка парсинга индексов reranker'а '{content}': {parse_e}. Возвращаем исходный порядок."
            )
            indices = list(
                range(min(5, len(documents)))
            )  # Возвращаем топ-5 по умолчанию

        if not indices and len(documents) > 0:
            # Если reranker ничего не вернул, используем дефолтный порядок
            logging.warning(
                "Reranker не вернул индексы. Возвращаем исходный порядок документов."
            )
            indices = list(range(min(5, len(documents))))

        return indices
    except httpx.HTTPStatusError as e:
        logging.error(
            f"HTTP ошибка при переранжировании: {e.response.status_code} - {e.response.text}"
        )
        return list(range(min(5, len(documents))))  # Возвращаем дефолтный порядок
    except Exception as e:
        logging.error(f"Ошибка при переранжировании документов: {e}", exc_info=True)
        return list(range(min(5, len(documents))))  # Возвращаем дефолтный порядок


def extract_markdown_links(text: str) -> list[str]:
    # Ищет как [текст](файл.md), так и <файл.md>
    pattern = r"(?:\]\(([^)]+\.md)\)|<([^>]+\.md)>)"
    found_links = re.findall(pattern, text)
    # Возвращаем только первый элемент из кортежа, который содержит найденную ссылку
    return [link[0] if link[0] else link[1] for link in found_links]


async def resolve_and_fetch_content(
    link_target: str, source_file_path: str, seen_paths: set[str], max_chars: int = 800
) -> str | None:
    source_path = Path(source_file_path).parent

    # Нормализация link_target: удаляем якоря (#хеш)
    link_target_clean = link_target.split("#")[0]

    target_path = Path(link_target_clean)

    absolute_target: Path
    if not target_path.is_absolute():
        absolute_target = (
            Path(DOCS_PATH) / source_path.relative_to(DOCS_PATH) / target_path
        ).resolve()
    else:
        # Если ссылка абсолютная, убеждаемся, что она находится внутри DOCS_PATH
        if not str(target_path).startswith(str(DOCS_PATH)):
            logging.warning(f"Ссылка вне DOCS_PATH: {target_path}")
            return None
        absolute_target = target_path.resolve()

    try:
        # Проверка на выход за пределы DOCS_PATH после resolve
        if not str(absolute_target).startswith(str(Path(DOCS_PATH).resolve())):
            logging.warning(f"Попытка доступа за пределы DOCS_PATH: {absolute_target}")
            return None

        absolute_target_str = str(absolute_target).replace(os.sep, "/")
        if absolute_target_str in seen_paths:
            return None  # Уже обрабатывали этот файл

        if not absolute_target.exists() or not absolute_target.is_file():
            logging.debug(
                f"Связанный файл не найден или не является файлом: {absolute_target}"
            )
            return None

        if absolute_target.suffix.lower() != ".md":
            logging.debug(f"Связанный файл не Markdown: {absolute_target}")
            return None

        seen_paths.add(absolute_target_str)
        with open(absolute_target, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(
                max_chars * 2
            )  # Читаем немного больше, чтобы потом обрезать красиво
            # Удаляем заголовки, чтобы не дублировать информацию или не сбивать контекст
            # и оставляем только часть текста
            content_cleaned = re.sub(r"#+\s.*", "", content).strip()
            return f"\n\n--- Связанный документ ({absolute_target.name}): ---\n{content_cleaned[:max_chars]}..."
    except Exception as e:
        logging.error(
            f"Ошибка при разрешении/получении связанного контента для {link_target} (из {source_file_path}): {e}"
        )
        return None


async def main():
    query = "Object3DGeomReference - конструктор"
    query_embedding = await get_embedding(query)

    # Проверяем, что эмбеддинг получен
    if not query_embedding:
        logging.error("Не удалось получить эмбеддинг для запроса.")
        return

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=N_RESULTS,  # candidates for reranking
        include=[
            "documents",
            "metadatas",
            "distances",
        ],  # Добавим distances для информации
    )

    # logging.info(f"Полные результаты запроса к ChromaDB: {results}")

    # Предполагается, что 'documents' и 'metadatas' будут списками списков,
    # если query_embeddings был список, но в данном случае он один.
    # Так что results["documents"] будет список документов для первого запроса.

    retrieved_documents = results.get("documents", [[]])[0]  # Содержимое документов
    retrieved_metadatas = results.get("metadatas", [[]])[0]  # Метаданные документов
    retrieved_distances = results.get("distances", [[]])[0]  # Расстояния

    if not retrieved_documents:
        logging.info("Документы не найдены в ChromaDB.")
        return

    logging.info(f"Количество найденных документов: {len(retrieved_documents)}")

    # содержимое документов и их метаданные
    for i, doc_content in enumerate(retrieved_documents):
        meta = retrieved_metadatas[i]
        distance = retrieved_distances[i]
        logging.info(f"--- Документ {i} (Расстояние: {distance:.4f}) ---")
        logging.info(f"Метаданные: {meta}")
        logging.info(f"Содержимое: {doc_content[:500]}...")
        logging.info("--------------------------------------------------")

    reranked_indices = await rerank_documents(query, retrieved_documents)
    final_output = f"Результаты поиска по запросу: '{query}'\n\n"
    # logging.info(f"Переранжированные индексы: {reranked_indices}")
    seen_linked_files = set()

    for rank, original_idx in enumerate(reranked_indices[:5]):
        doc_text = retrieved_documents[original_idx]
        source = retrieved_metadatas[original_idx]["source"]

        final_output += f"--- Результат {rank + 1} (Источник: {source}) ---\n"
        final_output += f"{doc_text}\n"

        # Ищем ссылки в тексте текущего чанка
        links = extract_markdown_links(doc_text)
        if links:
            logging.debug(
                f"Найдено {len(links)} ссылок в результате {rank + 1} из {source}"
            )
            # Подгружаем контекст по ссылкам
            for link in links:
                # Пытаемся загрузить связанный документ (ограничиваем его)
                extra_content = await resolve_and_fetch_content(
                    link,
                    (Path(DOCS_PATH) / source).as_posix(),
                    seen_linked_files,
                    max_chars=400,  # Меньше символов для связанных документов
                )
                if extra_content:
                    final_output += extra_content
                    logging.debug(f"Добавлен связанный контент из {link}")
    with open("final_output.md", "w", encoding="utf-8") as f:
        f.write(final_output)


if __name__ == "__main__":
    asyncio.run(main())
