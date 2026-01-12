"""
Модуль для извлечения сущностей (Phase 1) из текста документации.
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


class EntityExtractor:
    """Класс для извлечения сущностей из текста."""

    # Определение типов сущностей с русскими описаниями
    ENTITY_DEFINITIONS = {
        "Class": "Класс C# - определяет структуру объектов",
        "Interface": "Интерфейс C# - контракт для реализации",
        "Method": "Метод - функция класса или интерфейса",
        "Field": "Поле класса - переменная-член",
        "Property": "Свойство класса - геттер/сеттер",
        "Enum": "Перечисление - набор именованных констант",
        "DTO": "Data Transfer Object - объект для передачи данных",
        "Exception": "Исключение - класс для обработки ошибок",
        "CodeExample": "Пример кода - фрагмент кода для демонстрации",
        "Constructor": "Конструктор - метод инициализации объекта",
    }

    def __init__(self, http_client: httpx.AsyncClient):
        """
        Инициализация извлекателя сущностей.

        Args:
            http_client: HTTP клиент для запросов к LMStudio
        """
        self.http_client = http_client
        self.model = cfg.ENTITY_EXTRACTION_MODEL
        self.ai_server_url = cfg.AI_SERVER_URL
        self.max_entities = cfg.MAX_ENTITIES_PER_CHUNK

    def _build_entity_prompt(self, text: str) -> str:
        """
        Построение промпта для извлечения сущностей.
        Оптимизирован для русских текстов и малых моделей.
        """
        entity_types_list = "\n".join(
            [f"- {etype}: {desc}" for etype, desc in self.ENTITY_DEFINITIONS.items()]
        )

        prompt = f"""Ты извлекаешь сущности из документации C# SDK на русском языке.

Типы сущностей:
{entity_types_list}

Инструкции:
1. Найди все упомянутые в тексте сущности
2. Для каждой сущности определи её тип
3. Извлеки имя и краткое описание (если есть)
4. Верни результат ТОЛЬКО в формате JSON

Формат ответа (строгий JSON, без markdown, без комментариев):
{{
  "entities": [
    {{
      "type": "Class|Interface|Method|Field|Property|Enum|DTO|Exception|CodeExample|Constructor",
      "name": "ИмяСущности",
      "description": "Краткое описание",
      "metadata": {{}}
    }}
  ]
}}

Текст для анализа:
{text[:3000]}

Ответ (только JSON):"""

        return prompt

    async def extract_entities(
        self, text: str, chunk_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Извлечение сущностей из текста.

        Args:
            text: Текст для анализа
            chunk_id: ID чанка (для метаданных)

        Returns:
            Список извлеченных сущностей
        """
        if not text or not text.strip():
            return []

        prompt = self._build_entity_prompt(text)

        try:
            response = await self.http_client.post(
                f"{self.ai_server_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Ты помощник для извлечения структурированной информации. Отвечай ТОЛЬКО валидным JSON без markdown разметки.",
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
                entities = data.get("entities", [])

                # Валидация и нормализация
                validated_entities = []
                seen: set[str] = set()

                for entity in entities[: self.max_entities]:
                    entity_type = entity.get("type", "").strip()
                    name = entity.get("name", "").strip()

                    if not entity_type or not name:
                        continue

                    # Проверка типа
                    if entity_type not in self.ENTITY_DEFINITIONS:
                        logger.debug(
                            f"Пропущен неизвестный тип сущности: {entity_type}"
                        )
                        continue

                    # Дедупликация
                    entity_key = f"{entity_type}:{name.lower()}"
                    if entity_key in seen:
                        continue
                    seen.add(entity_key)

                    # Создание уникального ID
                    entity_id = self._generate_entity_id(entity_type, name, chunk_id)

                    validated_entities.append(
                        {
                            "type": entity_type,
                            "id": entity_id,
                            "name": name,
                            "description": entity.get("description", "").strip(),
                            "metadata": entity.get("metadata", {}),
                            "source_chunk": chunk_id,
                        }
                    )

                logger.info(f"Извлечено {len(validated_entities)} сущностей из чанка")
                return validated_entities

            except json.JSONDecodeError as e:
                logger.error(
                    f"Ошибка парсинга JSON ответа: {e}\nОтвет: {content[:500]}"
                )
                return []

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP ошибка при извлечении сущностей: {e.response.status_code} - {e.response.text}"
            )
            return []
        except Exception as e:
            logger.error(f"Ошибка при извлечении сущностей: {e}", exc_info=True)
            return []

    def _generate_entity_id(
        self, entity_type: str, name: str, chunk_id: str | None = None
    ) -> str:
        """
        Генерация уникального ID для сущности.

        Args:
            entity_type: Тип сущности
            name: Имя сущности
            chunk_id: ID чанка (для контекста)

        Returns:
            Уникальный ID сущности
        """
        # Нормализация имени для ID
        normalized_name = re.sub(r"[^\w\s]", "", name)
        normalized_name = re.sub(r"\s+", "_", normalized_name)
        normalized_name = normalized_name.lower()

        # Простой ID: тип:имя
        entity_id = f"{entity_type}:{normalized_name}"

        # Если нужен более уникальный ID, добавляем хеш или часть chunk_id
        if chunk_id:
            chunk_hash = hash(chunk_id) % 10000
            entity_id = f"{entity_type}:{normalized_name}:{chunk_hash}"

        return entity_id

    async def extract_and_store_entities(
        self, text: str, chunk_id: str, graph: Neo4jGraph
    ) -> list[dict[str, Any]]:
        """
        Извлечение сущностей и сохранение их в Neo4j.

        Args:
            text: Текст для анализа
            chunk_id: ID чанка
            graph: Экземпляр Neo4jGraph

        Returns:
            Список извлеченных и сохраненных сущностей
        """
        entities = await self.extract_entities(text, chunk_id)

        stored_entities = []
        for entity in entities:
            success = graph.upsert_entity(
                entity_type=entity["type"],
                entity_id=entity["id"],
                name=entity["name"],
                description=entity.get("description"),
                metadata=entity.get("metadata", {}),
                source_chunk=chunk_id,
            )

            if success:
                stored_entities.append(entity)

        logger.info(f"Сохранено {len(stored_entities)} сущностей в Neo4j")
        return stored_entities
