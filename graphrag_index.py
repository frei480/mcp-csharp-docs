"""
Модуль для GraphRAG индексации.
Интегрирует двухфазный extraction (entities → relations) в процесс индексации.
"""

import logging
from pathlib import Path
from typing import Any

import httpx

from ApiSdkSplitter import ApiSdkChunkSplitter
from config import cfg
from entity_extraction import EntityExtractor
from neo4j_graph import Neo4jGraph
from RecursiveMarkdownTextSplitter import RecursiveMarkdownSplitter
from relation_extraction import RelationExtractor

logger = logging.getLogger(__name__)


class GraphRAGIndexer:
    """Класс для GraphRAG индексации с двухфазным extraction."""

    def __init__(self, http_client: httpx.AsyncClient):
        """
        Инициализация GraphRAG индексатора.

        Args:
            http_client: HTTP клиент для запросов к LMStudio
        """
        self.http_client = http_client
        self.graph = Neo4jGraph()
        self.entity_extractor = EntityExtractor(http_client)
        self.relation_extractor = RelationExtractor(http_client)

        # Инициализация сплиттеров
        self.markdown_splitter = RecursiveMarkdownSplitter(cfg.CHUNK_SIZE)
        self.api_splitter = ApiSdkChunkSplitter()

        logger.info("GraphRAG индексатор инициализирован")

    def choose_splitter(self, text: str):
        """Выбор подходящего сплиттера на основе содержимого."""
        if "| Имя | Описание |" in text or "класс" in text.lower():
            return self.api_splitter
        return self.markdown_splitter

    async def index_chunk(
        self, chunk_text: str, chunk_id: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Индексация одного чанка с двухфазным extraction.

        Args:
            chunk_text: Текст чанка для индексации
            chunk_id: Уникальный ID чанка
            metadata: Дополнительные метаданные чанка

        Returns:
            Словарь с результатами индексации
        """
        if not chunk_text or not chunk_text.strip():
            return {
                "entities": [],  # type: ignore
                "relations": [],  # type: ignore
                "success": False,
            }

        logger.info(f"Индексация чанка: {chunk_id}")

        try:
            # PHASE 1: Извлечение сущностей
            logger.debug(f"Phase 1: Извлечение сущностей из чанка {chunk_id}")
            entities = await self.entity_extractor.extract_and_store_entities(
                text=chunk_text, chunk_id=chunk_id, graph=self.graph
            )

            logger.info(f"Phase 1 завершена: извлечено {len(entities)} сущностей")

            # PHASE 2: Извлечение связей между сущностями
            relations: list[dict[str, Any]] = []
            if entities:
                logger.debug(f"Phase 2: Извлечение связей из чанка {chunk_id}")
                relations = await self.relation_extractor.extract_and_store_relations(
                    text=chunk_text,
                    entities=entities,
                    chunk_id=chunk_id,
                    graph=self.graph,
                )

                logger.info(f"Phase 2 завершена: извлечено {len(relations)} связей")
            else:
                logger.debug("Phase 2 пропущена: нет сущностей для извлечения связей")

            return {
                "entities": entities,
                "relations": relations,
                "success": True,
                "entity_count": len(entities),
                "relation_count": len(relations),
            }

        except Exception as e:
            logger.error(f"Ошибка при индексации чанка {chunk_id}: {e}", exc_info=True)
            return {
                "entities": [],  # type: ignore
                "relations": [],  # type: ignore
                "success": False,
                "error": str(e),
            }

    async def index_text(
        self, text: str, source_file: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Индексация полного текста (разбиение на чанки + индексация каждого).

        Args:
            text: Полный текст документа
            source_file: Путь к исходному файлу
            metadata: Дополнительные метаданные

        Returns:
            Словарь с результатами индексации
        """
        logger.info(f"Индексация текста из файла: {source_file}")

        # Выбор сплиттера
        splitter = self.choose_splitter(text)

        # Разбиение текста на чанки
        chunks = splitter.split_text(text)

        logger.info(f"Текст разбит на {len(chunks)} чанков")

        # Индексация каждого чанка
        total_entities = 0
        total_relations = 0
        successful_chunks = 0

        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                chunk_text = chunk.get("text", "")
                chunk_metadata: dict[str, Any] = {
                    **(metadata or {}),
                    **chunk.get("metadata", {}),
                }  # type: ignore
            else:
                chunk_text = chunk
                chunk_metadata: dict[str, Any] = metadata or {}  # type: ignore

            if not chunk_text or not chunk_text.strip():
                continue

            # Генерация ID чанка
            chunk_id = f"{source_file.replace('/', '__')}__chunk_{i}"

            # Индексация чанка
            result = await self.index_chunk(
                chunk_text=chunk_text, chunk_id=chunk_id, metadata=chunk_metadata
            )

            if result["success"]:
                successful_chunks += 1
                total_entities += result["entity_count"]
                total_relations += result["relation_count"]

            # Небольшая задержка для избежания перегрузки AI сервера
            # await asyncio.sleep(0.1)

        logger.info(
            f"Индексация завершена: {successful_chunks}/{len(chunks)} чанков, "
            f"{total_entities} сущностей, {total_relations} связей"
        )

        return {
            "source_file": source_file,
            "chunks_processed": len(chunks),
            "chunks_successful": successful_chunks,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "success": successful_chunks > 0,
        }

    async def index_file(
        self, file_path: Path, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Индексация файла документации.

        Args:
            file_path: Путь к файлу
            metadata: Дополнительные метаданные

        Returns:
            Словарь с результатами индексации
        """
        try:
            # Чтение файла
            if not file_path.exists() or not file_path.is_file():
                logger.warning(f"Файл не найден: {file_path}")
                return {"success": False, "error": "File not found"}

            if file_path.suffix.lower() != ".md":
                logger.debug(f"Файл не является Markdown: {file_path}")
                return {"success": False, "error": "Not a markdown file"}

            text = file_path.read_text(encoding="utf-8", errors="ignore")

            # Нормализация пути для использования в качестве идентификатора
            source_file = str(file_path.relative_to(Path(cfg.DOCS_PATH))).replace(
                "\\", "/"
            )

            # Индексация текста
            result = await self.index_text(
                text=text, source_file=source_file, metadata=metadata
            )

            return result

        except Exception as e:
            logger.error(f"Ошибка при индексации файла {file_path}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def close(self):
        """Закрытие соединений."""
        if self.graph:
            self.graph.close()
            logger.info("GraphRAG индексатор закрыт")
