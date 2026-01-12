"""
Модуль для GraphRAG запросов.
Реализует query pipeline: поиск документов → извлечение подграфа → генерация ответа.
"""

import logging

import httpx
from rank_bm25 import BM25Okapi

from config import cfg
from neo4j_graph import Neo4jGraph

logger = logging.getLogger(__name__)


class GraphRAGQuery:
    """Класс для GraphRAG запросов."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        chroma_collection,
        graph: Neo4jGraph | None = None,
    ):
        """
        Инициализация GraphRAG query системы.

        Args:
            http_client: HTTP клиент для запросов к LMStudio
            chroma_collection: Коллекция ChromaDB для векторного поиска
            graph: Экземпляр Neo4jGraph (опционально, создается при необходимости)
        """
        self.http_client = http_client
        self.chroma_collection = chroma_collection
        self.graph = graph or Neo4jGraph()
        self.query_model = cfg.QUERY_MODEL
        self.ai_server_url = cfg.AI_SERVER_URL
        self.embedding_model = cfg.EMBEDDING_MODEL
        self.context_depth = cfg.GRAPHRAG_CONTEXT_DEPTH

        logger.info("GraphRAG query система инициализирована")

    async def get_embedding(self, text: str) -> list[float]:
        """Получение embedding для текста."""
        if not text or not text.strip():
            return []

        try:
            response = await self.http_client.post(
                f"{self.ai_server_url}/embeddings",
                json={"model": self.embedding_model, "input": text},
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

            if (
                "data" in data
                and len(data["data"]) > 0
                and "embedding" in data["data"][0]
            ):
                return data["data"][0]["embedding"]

            logger.error(f"Неожиданный формат ответа от сервера эмбеддингов: {data}")
            return []
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддинга: {e}", exc_info=True)
            return []

    async def retrieve_documents(self, query: str, n_results: int = 20) -> list[dict]:
        """
        Извлечение релевантных документов из ChromaDB.

        Args:
            query: Поисковый запрос
            n_results: Количество результатов для извлечения

        Returns:
            Список документов с метаданными
        """
        logger.info(f"Поиск документов для запроса: '{query}'")

        # Получение embedding запроса
        query_embedding = await self.get_embedding(query)
        if not query_embedding:
            logger.error("Не удалось получить эмбеддинг для запроса")
            return []

        # Поиск в ChromaDB
        try:
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            if not results or not results["documents"] or not results["documents"][0]:
                logger.info("По запросу ничего не найдено в ChromaDB")
                return []

            documents = []
            for i, doc_text in enumerate(results["documents"][0]):
                documents.append(
                    {
                        "text": doc_text,
                        "metadata": results["metadatas"][0][i]
                        if results["metadatas"]
                        else {},
                        "distance": results["distances"][0][i]
                        if results["distances"]
                        else None,
                        "index": i,
                    }
                )

            logger.info(f"Найдено {len(documents)} документов")
            return documents

        except Exception as e:
            logger.error(f"Ошибка при поиске в ChromaDB: {e}", exc_info=True)
            return []

    async def rerank_with_bm25(self, query: str, documents: list[dict]) -> list[int]:
        """
        Переранжирование документов с помощью BM25.

        Args:
            query: Поисковый запрос
            documents: Список документов

        Returns:
            Список индексов документов, отсортированных по релевантности
        """
        if not documents:
            return []

        logger.debug("Переранжирование документов с помощью BM25")

        try:
            # Токенизация документов
            tokenized_corpus = [doc["text"].lower().split(" ") for doc in documents]

            # Инициализация BM25
            bm25 = BM25Okapi(tokenized_corpus)

            # Токенизация запроса
            tokenized_query = query.lower().split(" ")

            # Получение оценок BM25
            doc_scores = bm25.get_scores(tokenized_query)

            # Сортировка по оценкам
            indexed_scores = list(enumerate(doc_scores))
            reranked_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)

            # Извлечение индексов
            reranked_indices = [idx for idx, score in reranked_scores]

            logger.debug("BM25 переранжирование завершено")
            return reranked_indices

        except Exception as e:
            logger.error(f"Ошибка при переранжировании BM25: {e}", exc_info=True)
            return list(range(len(documents)))

    def extract_entity_ids_from_text(self, text: str, graph: Neo4jGraph) -> list[str]:
        """
        Извлечение ID сущностей из текста путем поиска в графе.

        Args:
            text: Текст для анализа
            graph: Экземпляр Neo4jGraph

        Returns:
            Список ID найденных сущностей
        """
        # Простое извлечение: ищем упоминания сущностей по именам
        # В реальном сценарии можно использовать NER модель
        entity_ids = []

        # Поиск упоминаний имен сущностей в тексте
        # Для упрощения: поиск по словам из текста
        words = text.split()
        for word in words:
            # Очистка слова от пунктуации
            clean_word = word.strip(".,;:()[]{}'\"")
            if len(clean_word) > 3:  # Минимальная длина для поиска
                # Поиск сущностей по имени
                entities = graph.search_entities_by_name(clean_word, limit=5)
                for entity in entities:
                    if entity["id"] not in entity_ids:
                        entity_ids.append(entity["id"])

        logger.debug(f"Извлечено {len(entity_ids)} ID сущностей из текста")
        return entity_ids

    def get_subgraph_context(
        self, entity_ids: list[str], depth: int | None = None
    ) -> dict:
        """
        Получение контекста подграфа для указанных сущностей.

        Args:
            entity_ids: Список ID сущностей
            depth: Глубина обхода графа

        Returns:
            Словарь с узлами и связями подграфа
        """
        if not entity_ids:
            return {"nodes": [], "relationships": []}

        depth = depth or self.context_depth

        logger.debug(
            f"Получение подграфа для {len(entity_ids)} сущностей (глубина: {depth})"
        )

        subgraph = self.graph.get_subgraph(entity_ids, depth=depth)

        logger.debug(
            f"Подграф получен: {len(subgraph.get('nodes', []))} узлов, "
            f"{len(subgraph.get('relationships', []))} связей"
        )

        return subgraph

    def format_subgraph_context(self, subgraph: dict) -> str:
        """
        Форматирование подграфа в текстовый контекст для LLM.

        Args:
            subgraph: Словарь с узлами и связями

        Returns:
            Форматированный текстовый контекст
        """
        if not subgraph.get("nodes") and not subgraph.get("relationships"):
            return ""

        context_parts = []

        # Форматирование узлов
        nodes = subgraph.get("nodes", [])
        if nodes:
            context_parts.append("## Сущности в графе знаний:\n")
            for node in nodes[:50]:  # Ограничиваем количество для экономии токенов
                node_str = f"- **{node.get('type', 'Unknown')}**: {node.get('name', 'Unknown')}"
                if node.get("description"):
                    node_str += f" - {node['description'][:200]}"
                context_parts.append(node_str)
            context_parts.append("")

        # Форматирование связей
        relationships = subgraph.get("relationships", [])
        if relationships:
            context_parts.append("## Связи в графе знаний:\n")
            for rel in relationships[:30]:  # Ограничиваем количество
                context_parts.append(
                    f"- {rel.get('source', 'Unknown')} "
                    f"[{rel.get('type', 'RELATED')}] "
                    f"-> {rel.get('target', 'Unknown')}"
                )
            context_parts.append("")

        return "\n".join(context_parts)

    async def generate_answer(
        self, query: str, documents: list[dict], subgraph_context: str
    ) -> str:
        """
        Генерация ответа на основе документов и контекста графа.

        Args:
            query: Поисковый запрос
            documents: Список релевантных документов
            subgraph_context: Контекст подграфа знаний

        Returns:
            Сгенерированный ответ
        """
        logger.info("Генерация ответа с использованием LLM")

        # Форматирование документов
        documents_text = "\n\n".join(
            [
                f"### Документ {i + 1} (Источник: {doc.get('metadata', {}).get('source', 'Unknown')})\n\n{doc['text'][:1000]}"
                for i, doc in enumerate(documents[:5])  # Топ-5 документов
            ]
        )

        # Построение промпта
        system_prompt = """Ты помощник для ответов на вопросы о документации C# SDK на русском языке.
Используй предоставленные документы и контекст графа знаний для формирования точного и полезного ответа.
Если информации недостаточно, честно скажи об этом."""

        user_prompt = f"""Запрос пользователя: {query}

Документы из векторного поиска:
{documents_text}

Контекст графа знаний:
{subgraph_context}

Сформулируй подробный ответ на запрос пользователя, используя информацию из документов и контекста графа знаний."""

        try:
            response = await self.http_client.post(
                f"{self.ai_server_url}/chat/completions",
                json={
                    "model": self.query_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000,
                },
                timeout=120.0,
            )
            response.raise_for_status()

            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()

            logger.info("Ответ сгенерирован успешно")
            return answer

        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}", exc_info=True)
            return f"Ошибка при генерации ответа: {str(e)}"

    async def query(
        self,
        query: str,
        n_results: int = 20,
        use_graph: bool = True,
        generate_answer: bool = True,
    ) -> dict:
        """
        Выполнение полного GraphRAG запроса.

        Args:
            query: Поисковый запрос
            n_results: Количество документов для извлечения
            use_graph: Использовать ли граф знаний для контекста
            generate_answer: Генерировать ли ответ через LLM

        Returns:
            Словарь с результатами запроса
        """
        logger.info(f"Выполнение GraphRAG запроса: '{query}'")

        # Шаг 1: Извлечение документов из ChromaDB
        documents = await self.retrieve_documents(query, n_results=n_results)
        if not documents:
            return {
                "query": query,
                "documents": [],
                "answer": "По вашему запросу ничего не найдено в документации.",
                "success": False,
            }

        # Шаг 2: Переранжирование с BM25
        reranked_indices = await self.rerank_with_bm25(query, documents)
        reranked_documents = [documents[i] for i in reranked_indices[:10]]

        # Шаг 3: Извлечение контекста графа (опционально)
        subgraph_context = ""
        if use_graph:
            # Извлечение ID сущностей из топ-документов
            top_texts = [doc["text"] for doc in reranked_documents[:5]]
            combined_text = " ".join(top_texts)

            entity_ids = self.extract_entity_ids_from_text(combined_text, self.graph)

            if entity_ids:
                subgraph = self.get_subgraph_context(entity_ids)
                subgraph_context = self.format_subgraph_context(subgraph)

        # Шаг 4: Генерация ответа (опционально)
        answer = ""
        if generate_answer:
            answer = await self.generate_answer(
                query=query,
                documents=reranked_documents,
                subgraph_context=subgraph_context,
            )
        else:
            # Простой форматированный вывод документов
            answer = self.format_subgraph_context({"nodes": [], "relationships": []})
            answer += "\n\n## Релевантные документы:\n\n"
            for i, doc in enumerate(reranked_documents[:5]):
                answer += f"### {i + 1}. {doc.get('metadata', {}).get('source', 'Unknown')}\n\n"
                answer += doc["text"][:500] + "\n\n"

        return {
            "query": query,
            "documents": reranked_documents,
            "subgraph_context": subgraph_context,
            "answer": answer,
            "success": True,
            "document_count": len(reranked_documents),
        }

    def close(self):
        """Закрытие соединений."""
        if self.graph:
            self.graph.close()
