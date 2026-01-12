"""
Модуль для работы с Neo4j графом знаний.
Управляет узлами (сущностями) и связями в графе документации C# SDK.
"""

import logging
from typing import Any

from neo4j import GraphDatabase

from config import cfg

logger = logging.getLogger(__name__)


class Neo4jGraph:
    """Класс для работы с Neo4j графом знаний."""

    # Определение типов сущностей
    ENTITY_TYPES = [
        "Class",
        "Interface",
        "Method",
        "Field",
        "Property",
        "Enum",
        "DTO",
        "Exception",
        "CodeExample",
        "Constructor",
    ]

    # Определение типов связей
    RELATION_TYPES = [
        "CONTAINS",
        "INHERITS",
        "IMPLEMENTS",
        "ACCEPTS",
        "RETURNS",
        "THROWS",
        "DEPENDS_ON",
        "USED_IN",
    ]

    def __init__(self):
        """Инициализация подключения к Neo4j."""
        self.driver = GraphDatabase.driver(
            cfg.NEO4J_URI, auth=(cfg.NEO4J_USER, cfg.NEO4J_PASSWORD)
        )
        self._create_constraints()
        logger.info(f"Подключение к Neo4j установлено: {cfg.NEO4J_URI}")

    def close(self):
        """Закрытие подключения к Neo4j."""
        if self.driver:
            self.driver.close()
            logger.info("Подключение к Neo4j закрыто")

    def _create_constraints(self):
        """Создание ограничений и индексов для узлов."""
        with self.driver.session(database=cfg.NEO4J_DATABASE) as session:
            # Создаем уникальные ограничения для каждого типа сущности
            for entity_type in self.ENTITY_TYPES:
                session.run(f"""
                    CREATE CONSTRAINT IF NOT EXISTS FOR (e:{entity_type})
                    REQUIRE e.id IS UNIQUE
                """)
                # Индекс по имени для быстрого поиска
                session.run(f"""
                    CREATE INDEX IF NOT EXISTS FOR (e:{entity_type})
                    ON (e.name)
                """)

            logger.info("Ограничения и индексы Neo4j созданы")

    def upsert_entity(
        self,
        entity_type: str,
        entity_id: str,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_chunk: str | None = None,
    ) -> bool:
        """
        Создание или обновление сущности в графе.

        Args:
            entity_type: Тип сущности (Class, Method, и т.д.)
            entity_id: Уникальный идентификатор сущности
            name: Имя сущности
            description: Описание сущности
            metadata: Дополнительные метаданные
            source_chunk: ID чанка, из которого извлечена сущность

        Returns:
            True если успешно
        """
        if entity_type not in self.ENTITY_TYPES:
            logger.warning(f"Неизвестный тип сущности: {entity_type}")
            return False

        with self.driver.session(database=cfg.NEO4J_DATABASE) as session:
            props = {"id": entity_id, "name": name, "entity_type": entity_type}

            if description:
                props["description"] = description

            if metadata:
                for key, value in metadata.items():
                    if value is not None:
                        props[key] = value

            if source_chunk:
                props["source_chunk"] = source_chunk

            query = f"""
                MERGE (e:{entity_type} {{id: $id}})
                ON CREATE SET e += $props, e.created_at = timestamp()
                ON MATCH SET e += $props, e.updated_at = timestamp()
                RETURN e.id as id
            """

            result = session.run(query, id=entity_id, props=props)
            record = result.single()

            if record:
                logger.debug(f"Сущность создана/обновлена: {entity_type} {entity_id}")
                return True
            return False

    def upsert_relation(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        source_type: str | None = None,
        target_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Создание или обновление связи между сущностями.

        Args:
            source_id: ID исходной сущности
            relation_type: Тип связи (CONTAINS, INHERITS, и т.д.)
            target_id: ID целевой сущности
            source_type: Тип исходной сущности (опционально, для оптимизации)
            target_type: Тип целевой сущности (опционально, для оптимизации)
            metadata: Дополнительные метаданные связи

        Returns:
            True если успешно
        """
        if relation_type not in self.RELATION_TYPES:
            logger.warning(f"Неизвестный тип связи: {relation_type}")
            return False

        with self.driver.session(database=cfg.NEO4J_DATABASE) as session:
            # Если типы не указаны, находим их
            if not source_type or not target_type:
                source_info = session.run(
                    "MATCH (n {id: $id}) RETURN labels(n) as labels LIMIT 1",
                    id=source_id,
                ).single()
                target_info = session.run(
                    "MATCH (n {id: $id}) RETURN labels(n) as labels LIMIT 1",
                    id=target_id,
                ).single()

                if source_info:
                    source_type = (
                        source_info["labels"][0] if source_info["labels"] else None
                    )
                if target_info:
                    target_type = (
                        target_info["labels"][0] if target_info["labels"] else None
                    )

            if not source_type or not target_type:
                logger.warning(
                    f"Не удалось найти типы для связи: {source_id} -> {target_id}"
                )
                return False

            props = {}
            if metadata:
                props.update(metadata)

            query = f"""
                MATCH (source:{source_type} {{id: $source_id}})
                MATCH (target:{target_type} {{id: $target_id}})
                MERGE (source)-[r:{relation_type}]->(target)
                ON CREATE SET r += $props, r.created_at = timestamp()
                ON MATCH SET r += $props, r.updated_at = timestamp()
                RETURN r
            """

            result = session.run(
                query, source_id=source_id, target_id=target_id, props=props
            )
            record = result.single()

            if record:
                logger.debug(
                    f"Связь создана/обновлена: {source_id} -[{relation_type}]-> {target_id}"
                )
                return True
            return False

    def get_subgraph(
        self,
        entity_ids: list[str],
        depth: int = 2,
        relation_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Получение подграфа вокруг указанных сущностей.

        Args:
            entity_ids: Список ID сущностей для начала обхода
            depth: Глубина обхода графа
            relation_types: Типы связей для включения (None = все)

        Returns:
            Словарь с узлами и связями подграфа
        """
        with self.driver.session(database=cfg.NEO4J_DATABASE) as session:
            if relation_types:
                rel_filter = ":" + "|".join(relation_types)
            else:
                rel_filter = ""

            # Получаем узлы с учетом глубины обхода
            query_nodes = f"""
                MATCH (start)
                WHERE start.id IN $entity_ids
                OPTIONAL MATCH path = (start)-[*1..{depth}{rel_filter}]-(connected)
                WITH DISTINCT nodes(path) as path_nodes
                UNWIND path_nodes as node
                WITH DISTINCT node
                WHERE node IS NOT NULL
                RETURN collect(DISTINCT {{
                    id: node.id,
                    name: node.name,
                    type: labels(node)[0],
                    description: node.description
                }}) as nodes
                LIMIT 1000
            """

            result_nodes = session.run(query_nodes, entity_ids=entity_ids)
            record_nodes = result_nodes.single()

            nodes = []
            if record_nodes and record_nodes["nodes"]:
                nodes = [n for n in record_nodes["nodes"] if n and n.get("id")]

            # Получаем связи отдельно - упрощенный запрос
            rel_query = f"""
                MATCH (start)-[r{rel_filter}]->(target)
                WHERE start.id IN $entity_ids OR target.id IN $entity_ids
                RETURN DISTINCT
                    start.id as source,
                    target.id as target,
                    type(r) as type
                LIMIT 500
            """

            rel_result = session.run(rel_query, entity_ids=entity_ids)
            relationships = [
                {"source": rel["source"], "target": rel["target"], "type": rel["type"]}
                for rel in rel_result
                if rel.get("source") and rel.get("target")
            ]

            return {"nodes": nodes or [], "relationships": relationships or []}

    def search_entities_by_name(
        self,
        name_pattern: str,
        entity_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Поиск сущностей по имени.

        Args:
            name_pattern: Паттерн для поиска (поддержка LIKE)
            entity_types: Типы сущностей для фильтрации
            limit: Максимальное количество результатов

        Returns:
            Список найденных сущностей
        """
        with self.driver.session(database=cfg.NEO4J_DATABASE) as session:
            type_filter = ""
            if entity_types:
                type_filter = f":{':'.join(entity_types)}"

            query = f"""
                MATCH (e{type_filter})
                WHERE e.name CONTAINS $pattern
                RETURN e.id as id, e.name as name, 
                       labels(e)[0] as type, e.description as description
                LIMIT $limit
            """

            result = session.run(query, pattern=name_pattern, limit=limit)
            return [
                {
                    "id": record["id"],
                    "name": record["name"],
                    "type": record["type"],
                    "description": record.get("description"),
                }
                for record in result
            ]

    def get_entity_context(self, entity_id: str, depth: int = 2) -> dict[str, Any]:
        """
        Получение контекста сущности (соседние узлы и связи).

        Args:
            entity_id: ID сущности
            depth: Глубина обхода

        Returns:
            Контекст сущности
        """
        subgraph = self.get_subgraph([entity_id], depth=depth)
        return subgraph

    def clear_graph(self):
        """Очистка всего графа (использовать с осторожностью!)."""
        with self.driver.session(database=cfg.NEO4J_DATABASE) as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Весь граф Neo4j очищен")
