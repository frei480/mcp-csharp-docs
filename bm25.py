import asyncio
import logging
from pathlib import Path
import chromadb
from rank_bm25 import BM25Okapi 
from config import cfg
from embeddings_example import get_embedding,  extract_markdown_links, resolve_and_fetch_content

DOCS_PATH = cfg.DOCS_PATH
CHROMA_PERSIST_DIR = cfg.CHROMA_PERSIST_DIR
COLLECTION_NAME = cfg.COLLECTION_NAME
REGISTRY_COLLECTION_NAME = cfg.REGISTRY_COLLECTION_NAME
N_RESULTS = 50

chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
registry_collection = chroma_client.get_or_create_collection(name=REGISTRY_COLLECTION_NAME)

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
    # BM25 работает с токенами. Простейшая токенизация - по пробелам.
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


async def main():
    query = "LinearDimension TFlex.Model.Model2D"
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

    retrieved_documents = results.get("documents", [[]])[0]  # Содержимое документов
    retrieved_metadatas = results.get("metadatas", [[]])[0]  # Метаданные документов
    retrieved_distances = results.get("distances", [[]])[0]  # Расстояния

    if not retrieved_documents:
        logging.info("Документы не найдены в ChromaDB.")
        return

    logging.info(f"Количество найденных документов до переранжирования: {len(retrieved_documents)}")

    reranked_indices = await rerank_documents_bm25(query, retrieved_documents)
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
                    max_chars=1000,  # Меньше символов для связанных документов
                )
                if extra_content:
                    final_output += extra_content
                    logging.debug(f"Добавлен связанный контент из {link}")
    with open("final_output.md", "w", encoding="utf-8") as f:
        f.write(final_output)


if __name__ == "__main__":
    asyncio.run(main())