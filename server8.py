import os
import asyncio
import logging
import httpx
import re
import subprocess
from typing import List, Dict, Set, Optional
from pathlib import Path
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi 
# MCP библиотеки
from mcp.server.fastmcp import FastMCP

# Библиотеки для БД
import chromadb

# Сплиттеры текста
from RecursiveMarkdownTextSplitter import RecursiveMarkdownSplitter
from ApiSdkSplitter import ApiSdkChunkSplitter

# --- КОНФИГУРАЦИЯ ---
from config import cfg

# URL репозитория Git с документацией (HTTPS или SSH)
GIT_REPO_URL = cfg.GIT_REPO_URL
DOCS_PATH = cfg.DOCS_PATH

# URL локального AI движка на сервере (например, Ollama или LM Server)
AI_SERVER_URL = cfg.AI_SERVER_URL

# Модели (должны быть установлены на сервере)
RERANK_MODEL_NAME = cfg.RERANK_MODEL 
EMBEDDING_MODEL_NAME = cfg.EMBEDDING_MODEL

CHUNK_SIZE = cfg.CHUNK_SIZE
CHUNK_OVERLAP = cfg.CHUNK_OVERLAP
CHROMA_PERSIST_DIR = cfg.CHROMA_PERSIST_DIR
COLLECTION_NAME = cfg.COLLECTION_NAME
REGISTRY_COLLECTION_NAME = cfg.REGISTRY_COLLECTION_NAME

# Инициализация логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Инициализация клиентов
http_client = httpx.AsyncClient(timeout=120.0)
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
registry_collection = chroma_client.get_or_create_collection(name=REGISTRY_COLLECTION_NAME)

# --- GIT SYNC ---
def sync_docs_repo():
    """Клонирует или обновляет репозиторий документации."""
    logging.info(f"Синхронизация документации из {GIT_REPO_URL} в {DOCS_PATH}")
    try:
        if not os.path.exists(DOCS_PATH):
            # Клонирование
            logging.info(f"Клонирование репозитория {GIT_REPO_URL} в {DOCS_PATH}...")
            subprocess.run(["git", "clone", "--depth", "1", GIT_REPO_URL, DOCS_PATH], check=True)
            logging.info("Репозиторий клонирован успешно.")
        else:
            # Pull (обновление)
            logging.info(f"Обновление репозитория в {DOCS_PATH}...")
            result = subprocess.run(["git", "pull"], cwd=DOCS_PATH, capture_output=True, text=True, check=True)
            if "Already up to date" not in result.stdout:
                logging.info("Документация обновлена.")
            else:
                logging.info("Документация актуальна.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Ошибка Git при синхронизации: {e.cmd} - {e.stderr}")
        raise # Перевыбрасываем ошибку, чтобы синхронизация не продолжалась с поврежденным репозиторием
    except Exception as e:
        logging.error(f"Непредвиденная ошибка синхронизации: {e}")
        raise

# --- ЛИНКИ ---
def extract_markdown_links(text: str) -> List[str]:
    # Ищет как [текст](файл.md), так и <файл.md>
    pattern = r'(?:\]\(([^)]+\.md)\)|<([^>]+\.md)>)'
    found_links = re.findall(pattern, text)
    # Возвращаем только первый элемент из кортежа, который содержит найденную ссылку
    return [link[0] if link[0] else link[1] for link in found_links]


async def resolve_and_fetch_content(link_target: str, source_file_path: str, seen_paths: Set[str], max_chars: int = 800) -> Optional[str]:
    source_path = Path(source_file_path).parent
    
    # Нормализация link_target: удаляем якоря (#хеш)
    link_target_clean = link_target.split('#')[0]
    
    target_path = Path(link_target_clean)
    
    absolute_target: Path
    if not target_path.is_absolute():
        absolute_target = (Path(DOCS_PATH) / source_path.relative_to(DOCS_PATH) / target_path).resolve()
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
            return None # Уже обрабатывали этот файл
        
        if not absolute_target.exists() or not absolute_target.is_file():
            logging.debug(f"Связанный файл не найден или не является файлом: {absolute_target}")
            return None
        
        if absolute_target.suffix.lower() != '.md':
            logging.debug(f"Связанный файл не Markdown: {absolute_target}")
            return None

        seen_paths.add(absolute_target_str)
        with open(absolute_target, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(max_chars * 2) # Читаем немного больше, чтобы потом обрезать красиво
            # Удаляем заголовки, чтобы не дублировать информацию или не сбивать контекст
            # и оставляем только часть текста
            content_cleaned = re.sub(r'#+\s.*', '', content).strip()
            return f"\n\n--- Связанный документ ({absolute_target.name}): ---\n{content_cleaned[:max_chars]}..."
    except Exception as e:
        logging.error(f"Ошибка при разрешении/получении связанного контента для {link_target} (из {source_file_path}): {e}")
        return None

# --- ИНДЕКСАЦИЯ ---
def choose_splitter(text: str):
    if "| Имя | Описание |" in text or "класс" in text:        
        return ApiSdkChunkSplitter()
    return RecursiveMarkdownSplitter(CHUNK_SIZE)

def sync_index():
    """
    Главная функция синхронизации.
    1. Синхронизирует файлы с Git (pull/clone).
    2. Сканирует файлы на диске.
    3. Сравнивает с ChromaDB и обновляет вектора.
    """
    try:
        # ШАГ 1: Синхронизация файлов с Git
        sync_docs_repo()
    except Exception as e:
        logging.error(f"Не удалось синхронизировать репозиторий: {e}. Пропускаем индексацию.")
        return

    # ШАГ 2: Подготовка к индексации
    logging.info("Запуск индексации...")
    docs_path = Path(DOCS_PATH)
    if not docs_path.exists():
        logging.warning(f"Путь документации {DOCS_PATH} не найден. Невозможно проиндексировать.")
        return

    db_files_map: Dict[str, float] = {}
    try:
        registry_data = registry_collection.get(include=["metadatas"])
        if registry_data and registry_data["ids"]:
            for i, file_path in enumerate(registry_data["ids"]):
                # Защита от отсутствия metadatas
                if registry_data["metadatas"] and registry_data["metadatas"][i]:
                    mtime = registry_data["metadatas"][i].get("mtime", 0)
                    db_files_map[file_path] = mtime
                else:
                    logging.warning(f"Отсутствуют метаданные для файла в реестре: {file_path}")
                    db_files_map[file_path] = 0 # Считаем, что нужно обновить
        logging.info(f"В реестре ChromaDB найдено записей: {len(db_files_map)}")
    except Exception as e:
        logging.error(f"Ошибка чтения реестра из ChromaDB: {e}. Начинаем с чистого реестра.")
        db_files_map = {}

    disk_files_map: Dict[str, float] = {}
    for file_path in docs_path.rglob("*.md"):
        try:
            mtime = os.path.getmtime(file_path)
            # Нормализуем путь для единообразия (использование слешей вместо ос.sep)
            norm_path = str(file_path.relative_to(docs_path)).replace(os.sep, "/")
            disk_files_map[norm_path] = mtime
        except Exception as e:
            logging.warning(f"Ошибка доступа к файлу {file_path}: {e}")

    disk_paths = set(disk_files_map.keys())
    db_paths = set(db_files_map.keys())

    new_files = disk_paths - db_paths
    modified_files = {p for p in disk_paths & db_paths if disk_files_map[p] > db_files_map[p]}
    deleted_files = db_paths - disk_paths

    total_changes = len(new_files) + len(modified_files) + len(deleted_files)
    if total_changes == 0:
        logging.info("Синхронизация не требуется: изменений нет.")
        return

    logging.info(f"Обнаружено изменений: Новых: {len(new_files)}, Изменено: {len(modified_files)}, Удалено: {len(deleted_files)}")

    # ШАГ 3: Обработка изменений
    # Используем синхронный httpx.Client для функции get_sync_embedding
    # в контексте to_thread, чтобы избежать проблем с asyncio
    sync_client = httpx.Client(timeout=120.0)
    
    def get_sync_embedding(text):
        try:
            resp = sync_client.post(
                f"{AI_SERVER_URL}/embeddings",
                json={"model": EMBEDDING_MODEL_NAME, "input": text}
            )
            resp.raise_for_status()
            data = resp.json()
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["embedding"]
            logging.error(f"Ответ от сервера эмбеддингов не содержит 'data' или он пуст: {data}")
            return [0.0] * 768 # Возвращаем вектор нулей при ошибке
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP ошибка при получении эмбеддинга: {e.response.status_code} - {e.response.text}")
            return [0.0] * 768
        except Exception as e:
            logging.error(f"Ошибка при получении эмбеддинга: {e}")
            return [0.0] * 768

    # text_splitter = RecursiveTextSplitter(CHUNK_SIZE, CHUNK_OVERLAP)
    # text_splitter = RecursiveMarkdownSplitter(CHUNK_SIZE)
    

    if deleted_files:
        try:
            # Удаляем из основной коллекции
            collection.delete(where={"source": list(deleted_files)})
            # Удаляем из реестра
            registry_collection.delete(ids=list(deleted_files))
            logging.info(f"Удалено файлов из индекса: {len(deleted_files)}")
        except Exception as e:
            logging.error(f"Ошибка при удалении файлов из ChromaDB: {e}")

    files_to_process = list(new_files) + list(modified_files)
    processed_count = 0

    for file_rel_path_str in files_to_process:
        try:
            local_path = docs_path / Path(file_rel_path_str) # Соединяем базовый путь с относительным
            mtime = disk_files_map[file_rel_path_str]
            
            # Проверяем, что файл существует и читаем его
            if not local_path.exists() or not local_path.is_file():
                logging.warning(f"Файл не найден при обработке: {local_path}. Пропускаем.")
                continue

            with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
              
            text_splitter = choose_splitter(text)
            chunks = text_splitter.split_text(text)
            
            # Если файл был модифицирован, сначала удаляем старые чанки
            if file_rel_path_str in modified_files:
                collection.delete(where={"source": file_rel_path_str})
                logging.debug(f"Удалены старые чанки для модифицированного файла: {file_rel_path_str}")
            
            chunk_ids = []
            chunk_documents = []
            chunk_metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_text = chunk["text"].strip()
                if not chunk_text:# if not chunk.strip(): # Пропускаем пустые чанки
                    continue

                chunk_id = f"{file_rel_path_str.replace('/', '__')}__chunk_{i}" # Создаем уникальный ID
                metadata ={"source": file_rel_path_str,
                           "mtime": mtime,
                           "chunk_index": i,
                           **chunk.get("metadata", {})
                           }
                chunk_ids.append(chunk_id)
                # chunk_documents.append(chunk)
                chunk_documents.append(chunk_text)
                # chunk_metadatas.append({"source": file_rel_path_str, "mtime": mtime, "chunk_index": i})
                chunk_metadatas.append(metadata)
            if chunk_ids:
                embeddings = [get_sync_embedding(t) for t in chunk_documents]
                # Фильтруем пустые эмбеддинги, чтобы избежать ошибок Chroma
                valid_embeddings = [e for e in embeddings if e]
                valid_chunk_ids = [chunk_ids[i] for i, e in enumerate(embeddings) if e]
                valid_chunk_documents = [chunk_documents[i] for i, e in enumerate(embeddings) if e]
                valid_chunk_metadatas = [chunk_metadatas[i] for i, e in enumerate(embeddings) if e]

                if valid_embeddings:
                    collection.add(
                        ids=valid_chunk_ids, 
                        documents=valid_chunk_documents, 
                        embeddings=valid_embeddings, 
                        metadatas=valid_chunk_metadatas
                    )
                    # Обновляем реестр для этого файла
                    registry_collection.upsert(
                        ids=[file_rel_path_str], 
                        documents=[file_rel_path_str], 
                        metadatas=[{"mtime": mtime}]
                    )
                    logging.debug(f"Добавлены/обновлены {len(valid_chunk_ids)} чанков для файла: {file_rel_path_str}")
                else:
                    logging.warning(f"Не удалось сгенерировать эмбеддинги для файла {file_rel_path_str}. Пропускаем его.")
            else:
                logging.info(f"Файл {file_rel_path_str} не содержит извлекаемых чанков после разбиения.")
            
            processed_count += 1
            if processed_count % 10 == 0: # Уменьшаем интервал логирования
                logging.info(f"Обработано {processed_count}/{len(files_to_process)} файлов...")

        except Exception as e:
            logging.error(f"Критическая ошибка обработки файла {file_rel_path_str}: {e}", exc_info=True)

    sync_client.close()
    logging.info("Индексация успешно завершена.")

# --- AI ФУНКЦИИ ---
async def get_embedding(text: str) -> List[float]:
    if not text.strip():
        return []
    try:
        response = await http_client.post(
            f"{AI_SERVER_URL}/embeddings",
            json={"model": EMBEDDING_MODEL_NAME, "input": text}
        )
        response.raise_for_status()
        data = response.json()
        if "data" in data and len(data["data"]) > 0 and "embedding" in data["data"][0]:
            return data["data"][0]["embedding"]
        logging.error(f"Неожиданный формат ответа от сервера эмбеддингов: {data}")
        return [] 
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP ошибка при получении эмбеддинга: {e.response.status_code} - {e.response.text}")
        return []
    except Exception as e:
        logging.error(f"Ошибка при получении эмбеддинга для текста '{text[:50]}...': {e}", exc_info=True)
        return []

async def rerank_documents(query: str, documents: List[str]) -> List[int]:
    if not documents: return []
    if not RERANK_MODEL_NAME:
        logging.warning("RERANK_MODEL_NAME не установлен, пропускаем переранжирование.")
        return list(range(len(documents))) # Возвращаем исходный порядок

    docs_snippets = []
    for i, doc in enumerate(documents):
        # Ограничиваем сниппет до 400 символов, чтобы не перегружать reranker
        snippet = doc[:400] + "..." if len(doc) > 400 else doc
        docs_snippets.append(f"[{i}] {snippet}")
    docs_text = "\n".join(docs_snippets)
    
    # Более четкий системный промпт
    system_prompt = "You are a document re-ranker. Given a query and a list of document snippets, identify the most relevant document snippets. Return ONLY a comma-separated list of the indices of the top 5 most relevant documents from the provided list. Example: 1, 3, 0, 5, 2"
    user_prompt = f"Query: {query}\n\nDocuments:\n{docs_text}\n\nTop 5 relevant indices:"
    
    try:
        response = await http_client.post(
            f"{AI_SERVER_URL}/chat/completions",
            json={
                "model": RERANK_MODEL_NAME, 
                "messages": [
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": user_prompt}
                ], 
                "temperature": 0.1, 
                "max_tokens": 50
            }
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        indices = []
        try:
            # Извлекаем все числа и преобразуем в int
            found_indices = [int(idx) for idx in re.findall(r'\d+', content)]
            # Фильтруем, чтобы индексы были в пределах допустимого диапазона
            indices = [idx for idx in found_indices if 0 <= idx < len(documents)]
            # Уникализируем и сохраняем порядок первого появления
            seen = set()
            unique_indices = []
            for idx in indices:
                if idx not in seen:
                    unique_indices.append(idx)
                    seen.add(idx)
            indices = unique_indices[:5] # Берем только топ-5
        except Exception as parse_e:
            logging.warning(f"Ошибка парсинга индексов reranker'а '{content}': {parse_e}. Возвращаем исходный порядок.")
            indices = list(range(min(5, len(documents)))) # Возвращаем топ-5 по умолчанию
        
        if not indices and len(documents) > 0:
            # Если reranker ничего не вернул, используем дефолтный порядок
            logging.warning("Reranker не вернул индексы. Возвращаем исходный порядок документов.")
            indices = list(range(min(5, len(documents))))
            
        return indices
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP ошибка при переранжировании: {e.response.status_code} - {e.response.text}")
        return list(range(min(5, len(documents)))) # Возвращаем дефолтный порядок
    except Exception as e:
        logging.error(f"Ошибка при переранжировании документов: {e}", exc_info=True)
        return list(range(min(5, len(documents)))) # Возвращаем дефолтный порядок

async def rerank_documents_bm25(query: str, retrieved_documents: list[str]) -> list[int]:
    """
    Переранжирует документы, используя BM25.
    Args:
        query: Исходный запрос пользователя.
        retrieved_documents: Список текстовых содержимых документов, полученных из ChromaDB.
    Returns:
        Список индексов документов, отсортированных по релевантности BM25.
    """
    if not retrieved_documents:
        return []

    logging.info("Начинаем переранжирование документов с помощью BM25.")

    # 1. Токенизация корпуса документов
    # Для более продвинутых сценариев можно использовать NLTK или SpaCy.
    tokenized_corpus = [doc.lower().split(" ") for doc in retrieved_documents]

    # 2. Инициализация BM25Okapi модели
    bm25 = BM25Okapi(tokenized_corpus)

    # 3. Токенизация запроса
    tokenized_query = query.lower().split(" ")

    # 4. Получение BM25 оценок для каждого документа
    doc_scores = bm25.get_scores(tokenized_query)

    # 5. Создание списка пар (оценка BM25, оригинальный_индекс_документа)
    # Это позволит нам отсортировать и сохранить связь с исходным порядком
    # из retrieved_documents.
    indexed_scores = list(enumerate(doc_scores)) # (original_index, score)

    # 6. Сортировка по оценкам BM25 в убывающем порядке
    # sorted() возвращает новый список, исходный не изменяется
    reranked_indexed_scores = sorted(indexed_scores, key=lambda item: item[1], reverse=True)

    # 7. Извлечение только оригинальных индексов, теперь они упорядочены по BM25
    reranked_indices = [idx for idx, score in reranked_indexed_scores]

    logging.info("Переранжирование BM25 завершено.")
    return reranked_indices


# --- MCP SERVER ---
mcp = FastMCP(name="tflex-csharp-docs", host="127.0.0.1", port=8000,)

@mcp.tool(name="open_document", description="Markdown файлы документации C# SDK",)
async def get_document(path: str) -> str:
    # Предотвращаем обход каталогов, даже если Path.resolve() не сработал бы
    if ".." in path or path.startswith('/'):
        raise ValueError("Invalid path: directory traversal attempt detected.")

    file_path = (Path(DOCS_PATH) / path).resolve()

    # Дополнительная проверка, чтобы убедиться, что путь находится внутри DOCS_PATH
    if not str(file_path).startswith(str(Path(DOCS_PATH).resolve())):
        raise ValueError("Access denied: path is outside the documentation directory.")

    if not file_path.exists() or not file_path.is_file():
        logging.warning(f"Файл не найден при чтении ресурса: {file_path}")
        raise ValueError("File not found")

    try:
        logging.info(f"Чтение файла документации: {file_path}")        
        return file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logging.error(f"Ошибка при чтении файла {file_path}: {e}")
        raise ValueError(f"Failed to read file: {e}")

@mcp.tool(name = "search_sdk_docs", description="Поиск в документации C# SDK по заданному запросу.",)
async def handle_call_tool(query: str) -> str:
    if not query: 
        return "Укажите поисковый запрос"

    logging.info(f"Получен поисковый запрос: '{query}'")

    query_embedding = await get_embedding(query)
    if not query_embedding: 
        logging.error("Не удалось получить эмбеддинг для запроса.")
        return "Ошибка при создании эмбеддинга для запроса. Проверьте AI сервер."
        
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=20, # Fetch more candidates for reranking
        include=["documents", "metadatas"]
    )

    if not results or not results["documents"] or not results["documents"][0]:
        logging.info(f"По запросу '{query}' ничего не найдено в ChromaDB.")
        return "По вашему запросу ничего не найдено в документации C# SDK."

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    
    logging.info(f"Найдено {len(docs)} потенциальных документов до переранжирования.")

    # Переранжирование для получения наиболее релевантных
    reranked_indices = await rerank_documents_bm25(query, docs)
        
    final_output = f"Результаты поиска по запросу: '{query}'\n\n"
    
    seen_linked_files = set() # Set для предотвращения дублирования связанных файлов

    # Обрабатываем до 3-5 наиболее релевантных результатов после переранжирования
    # Выбираем не более 5, чтобы не перегружать контекст
    for rank, original_idx in enumerate(reranked_indices[:5]): 
        if original_idx >= len(docs): # Защита на случай некорректного индекса от reranker'а
            continue

        doc_text = docs[original_idx]
        source = metadatas[original_idx]["source"]
        
        final_output += f"--- Результат {rank + 1} (Источник: {source}) ---\n"
        final_output += f"{doc_text}\n"
        
        # Ищем ссылки в тексте текущего чанка
        links = extract_markdown_links(doc_text)
        
        logging.debug(f"Найдено {len(links)} ссылок в результате {rank+1} из {source}")
        # Подгружаем контекст по ссылкам
        for link in links:
            # Пытаемся загрузить связанный документ (ограничиваем его)
            extra_content = await resolve_and_fetch_content(
                link, (Path(DOCS_PATH) / source).as_posix(), seen_linked_files, max_chars=1000 # Меньше символов для связанных документов
            )
            if extra_content:
                final_output += extra_content
                logging.debug(f"Добавлен связанный контент из {link}")
    
        if not reranked_indices:
            final_output += "Хотя документы были найдены, переранжирование не выявило явно релевантных. Представлены общие результаты:\n\n"
            # Если reranker ничего не вернул, берем первые 3 по умолчанию
            for rank, (doc_text, metadata) in enumerate(zip(docs[:3], metadatas[:3])):
                source = metadata["source"]
                final_output += f"--- Общий Результат {rank + 1} (Источник: {source}) ---\n"
                final_output += f"{doc_text}\n"
    return final_output

@mcp.tool()
async def fetch_url_text(url: str) -> str:
    """Download the text from a URL."""
    resp = await http_client.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.get_text(separator="\n", strip=True)

@mcp.tool()
async def fetch_page_links(url: str) -> List[str]:
    """Return a list of all URLs found on the given page."""
    resp = await http_client.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Extract all href attributes from <a> tags
    links = [a['href'] for a in soup.find_all('a', href=True)]
    return links

# --- ЕНДПОИНТ ДЛЯ РУЧНОЙ СИНХРОНИЗАЦИИ ---

# Глобальный флаг для защиты от повторного запуска
is_syncing = False
@mcp.tool(name="sync_docs_index", description="Запуск индексации документации C# SDK.",)
async def run_sync_task():
    """
    Обертка для запуска синхронной функции sync_index в отдельном потоке.    
    """
    global is_syncing

    if is_syncing:
        return "Индексация уже выполняется"

    async def task():
        global is_syncing
        is_syncing = True
        try:
            await asyncio.to_thread(sync_index)
        finally:
            is_syncing = False

    asyncio.create_task(task())
    return "Индексация запущена"


if __name__ == "__main__":
    mcp.run(transport="streamable-http", mount_path="/mcp")
    logging.info("Сервер запускается на http://127.0.0.1:8000")