"""
Модуль для извлечения связей между сущностями (Phase 2) из текста документации.
Оптимизирован для русского текста и малых моделей.
"""

import json
import logging
import re
from typing import Any

import httpx

from config import cfg
from neo4j_graph import Neo4jGraph

logger = logging.getLogger(__name__)


class RelationExtractor:
    """Класс для извлечения связей между сущностями."""

    # Определение типов связей с русскими описаниями
    RELATION_DEFINITIONS = {
        "CONTAINS": "Содержит - сущность содержит другую сущность (например, класс содержит методы)",
        "INHERITS": "Наследует - сущность наследуется от другой (класс наследует класс)",
        "IMPLEMENTS": "Реализует - сущность реализует интерфейс",
        "ACCEPTS": "Принимает - метод принимает параметр определенного типа",
        "RETURNS": "Возвращает - метод возвращает значение определенного типа",
        "THROWS": "Выбрасывает - метод выбрасывает исключение",
        "DEPENDS_ON": "Зависит от - сущность зависит от другой сущности",
        "USED_IN": "Используется в - сущность используется в другом контексте",
    }

    def __init__(self, http_client: httpx.AsyncClient):
        """
        Инициализация извлекателя связей.

        Args:
            http_client: HTTP клиент для запросов к LMStudio
        """
        self.http_client = http_client
        self.model = cfg.RELATION_EXTRACTION_MODEL
        self.ai_server_url = cfg.AI_SERVER_URL
        self.max_relations = cfg.MAX_RELATIONS_PER_CHUNK

    def _build_relation_prompt(self, text: str, entities: list[dict]) -> str:
        """
        Построение промпта для извлечения связей.
        Оптимизирован для русских текстов и малых моделей.
        """
        relation_types_list = "\n".join(
            [f"- {rtype}: {desc}" for rtype, desc in self.RELATION_DEFINITIONS.items()]
        )

        # Форматирование списка сущностей для промпта
        entities_list = "\n".join(
            [
                f"  - {e['type']}: {e['name']} (ID: {e['id']})"
                for e in entities[:20]  # Ограничиваем для экономии токенов
            ]
        )

        prompt = f"""Ты извлекаешь связи между сущностями из документации C# SDK на русском языке.

Типы связей:
{relation_types_list}

Известные сущности в тексте:
{entities_list}

Инструкции:
1. Найди все связи между упомянутыми сущностями
2. Для каждой связи определи её тип и направление (от source к target)
3. Верни результат ТОЛЬКО в формате JSON

Формат ответа (строгий JSON, без markdown, без комментариев):
{{
  "relations": [
    {{
      "source_id": "ID исходной сущности",
      "target_id": "ID целевой сущности",
      "type": "CONTAINS|INHERITS|IMPLEMENTS|ACCEPTS|RETURNS|THROWS|DEPENDS_ON|USED_IN",
      "description": "Краткое описание связи (опционально)",
      "metadata": {{}}
    }}
  ]
}}

Важно:
- source_id и target_id должны соответствовать ID из списка известных сущностей
- Если сущность не найдена в списке, используй её имя для поиска
- Направление связи: source -> target (например, Class CONTAINS Method)

Текст для анализа:
{text[:3000]}

Ответ (только JSON):"""

        return prompt

    async def extract_relations(
        self, text: str, entities: list[dict[str, Any]], chunk_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Извлечение связей между сущностями из текста.

        Args:
            text: Текст для анализа
            entities: Список известных сущностей
            chunk_id: ID чанка (для метаданных)

        Returns:
            Список извлеченных связей
        """
        if not text or not text.strip():
            return []

        if not entities:
            logger.debug("Нет сущностей для извлечения связей")
            return []

        prompt = self._build_relation_prompt(text, entities)

        try:
            response = await self.http_client.post(
                f"{self.ai_server_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Ты помощник для извлечения связей между сущностями. Отвечай ТОЛЬКО валидным JSON без markdown разметки.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "response_format": {"type": "json_object"}
                    if "json" in self.model.lower() or "gemma" in self.model.lower()
                    else None,
                },
                timeout=60.0,
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()

            # Очистка markdown обёрток, если есть
            content = re.sub(r"^```json\s*", "", content, flags=re.MULTILINE)
            content = re.sub(r"^```\s*$", "", content, flags=re.MULTILINE)
            content = content.strip()

            # Парсинг JSON
            try:
                data = json.loads(content)
                relations = data.get("relations", [])

                # Создание словаря сущностей для быстрого поиска по ID и имени
                entity_map_by_id = {e["id"]: e for e in entities}
                entity_map_by_name = {e["name"].lower(): e for e in entities}

                # Валидация и нормализация
                validated_relations = []
                seen = set()

                for relation in relations[: self.max_relations]:
                    relation_type = relation.get("type", "").strip()
                    source_id = relation.get("source_id", "").strip()
                    target_id = relation.get("target_id", "").strip()

                    # Пробуем найти по ID, если не найден - по имени
                    source_entity = entity_map_by_id.get(source_id)
                    if not source_entity and source_id:
                        # Пробуем найти по имени (case-insensitive)
                        source_entity = entity_map_by_name.get(source_id.lower())
                        if source_entity:
                            source_id = source_entity["id"]

                    target_entity = entity_map_by_id.get(target_id)
                    if not target_entity and target_id:
                        target_entity = entity_map_by_name.get(target_id.lower())
                        if target_entity:
                            target_id = target_entity["id"]

                    if not relation_type or not source_id or not target_id:
                        continue

                    # Проверка типа связи
                    if relation_type not in self.RELATION_DEFINITIONS:
                        logger.debug(f"Пропущен неизвестный тип связи: {relation_type}")
                        continue

                    # Проверка, что обе сущности найдены
                    if not source_entity or not target_entity:
                        logger.debug(
                            f"Не найдены сущности для связи: {source_id} -> {target_id}"
                        )
                        continue

                    # Дедупликация
                    relation_key = f"{source_id}:{relation_type}:{target_id}"
                    if relation_key in seen:
                        continue
                    seen.add(relation_key)

                    validated_relations.append(
                        {
                            "source_id": source_id,
                            "target_id": target_id,
                            "type": relation_type,
                            "source_type": source_entity["type"],
                            "target_type": target_entity["type"],
                            "description": relation.get("description", "").strip(),
                            "metadata": relation.get("metadata", {}),
                            "source_chunk": chunk_id,
                        }
                    )

                logger.info(f"Извлечено {len(validated_relations)} связей из чанка")
                return validated_relations

            except json.JSONDecodeError as e:
                logger.error(
                    f"Ошибка парсинга JSON ответа: {e}\nОтвет: {content[:500]}"
                )
                return []

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP ошибка при извлечении связей: {e.response.status_code} - {e.response.text}"
            )
            return []
        except Exception as e:
            logger.error(f"Ошибка при извлечении связей: {e}", exc_info=True)
            return []

    async def extract_and_store_relations(
        self,
        text: str,
        entities: list[dict[str, Any]],
        chunk_id: str,
        graph: Neo4jGraph,
    ) -> list[dict[str, Any]]:
        """
        Извлечение связей и сохранение их в Neo4j.

        Args:
            text: Текст для анализа
            entities: Список известных сущностей
            chunk_id: ID чанка
            graph: Экземпляр Neo4jGraph

        Returns:
            Список извлеченных и сохраненных связей
        """
        relations = await self.extract_relations(text, entities, chunk_id)

        stored_relations = []
        for relation in relations:
            success = graph.upsert_relation(
                source_id=relation["source_id"],
                relation_type=relation["type"],
                target_id=relation["target_id"],
                source_type=relation["source_type"],
                target_type=relation["target_type"],
                metadata={
                    **relation.get("metadata", {}),
                    "description": relation.get("description"),
                    "source_chunk": chunk_id,
                },
            )

            if success:
                stored_relations.append(relation)

        logger.info(f"Сохранено {len(stored_relations)} связей в Neo4j")
        return stored_relations
