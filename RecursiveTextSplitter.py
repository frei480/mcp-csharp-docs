# --- КЛАСС: RECURSIVE TEXT SPLITTER ---

import re
from typing import List


class RecursiveTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Более специфичные разделители в порядке убывания "грубости"
        self.separators = [
            "\n\n",  # Абзацы
            "\n",    # Строки
            " ",     # Слова
            "",      # Посимвольно (крайний случай)
            # Дополнительные разделители для предложений, если нужно:
            # r"(?<=\.)", r"(?<=\?)", r"(?<=!)" 
        ]

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        # Убедимся, что chunk_size > chunk_overlap
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        final_chunks = []
        current_text = text

        for separator in self.separators:
            if not current_text:
                break # Если текст уже полностью разбит, выходим

            # Если текущий разделитель - это пустая строка, то разбиваем по символам
            if separator == "":
                final_chunks.extend(self._split_by_char(current_text))
                current_text = ""
                break # После посимвольного разбиения дальнейшие разделители не нужны

            # Разделяем по текущему разделителю
            parts = current_text.split(separator)

            # Временно собираем чанки, чтобы потом объединить их с перекрытием
            temp_chunks_for_separator = []
            current_chunk_builder = []
            current_len = 0

            for part in parts:
                if not part.strip() and len(current_chunk_builder) == 0:
                    continue # Пропускаем пустые части в начале или между разделителями

                # Длина текущей части (+ длина разделителя, если это не последняя часть)
                part_with_sep_len = len(part) + len(separator)

                if current_len + part_with_sep_len <= self.chunk_size:
                    current_chunk_builder.append(part)
                    current_len += part_with_sep_len
                else:
                    # Текущая часть не помещается, сохраняем текущий чанк
                    if current_chunk_builder:
                        temp_chunks_for_separator.append(separator.join(current_chunk_builder).strip())

                    # Если текущая часть сама по себе слишком велика, 
                    # обрабатываем её рекурсивно или посимвольно (если это последний разделитель)
                    if part_with_sep_len > self.chunk_size:
                        if separator == self.separators[-1]: # Если это самый "мелкий" разделитель, то уже по символам
                             temp_chunks_for_separator.extend(self._split_by_char(part))
                        else:
                            # Для более "грубых" разделителей, просто добавляем, если она все равно большая
                            # и она будет дальше обработана следующим разделителем.
                            # Это может привести к чанкам > chunk_size временно, которые будут разбиты далее.
                            temp_chunks_for_separator.append(part.strip())

                        current_chunk_builder = []
                        current_len = 0
                    else:
                        # Начинаем новый чанк с текущей части
                        current_chunk_builder = [part]
                        current_len = part_with_sep_len

            # Добавляем последний чанк, если он есть
            if current_chunk_builder:
                temp_chunks_for_separator.append(separator.join(current_chunk_builder).strip())

            # После обработки всех частей для текущего разделителя, объединяем их с перекрытием
            # и подготавливаем для следующего шага
            processed_chunks_for_this_separator = []
            for j, chunk in enumerate(temp_chunks_for_separator):
                if not chunk: continue

                # Добавляем перекрытие из предыдущего чанка, если это не первый чанк
                if j > 0 and self.chunk_overlap > 0:
                    prev_chunk = temp_chunks_for_separator[j-1]
                    overlap = prev_chunk[-self.chunk_overlap:]
                    # Избегаем дублирования, если overlap уже в начале chunk
                    if not chunk.startswith(overlap):
                        chunk = overlap + chunk

                processed_chunks_for_this_separator.append(chunk)


            return self._split_recursive_robust(text, self.separators)


    def _split_recursive_robust(self, text: str, separators: List[str]) -> List[str]:
        if not text:
            return []

        # Базовый случай: если больше нет разделителей, или текст достаточно мал
        if not separators or len(text) <= self.chunk_size:
            return self._split_by_char(text) if len(text) > self.chunk_size else [text]

        current_separator = separators[0]
        remaining_separators = separators[1:]

        # Если текущий разделитель пустой, то это последний шаг, разбиваем посимвольно
        if not current_separator:
            return self._split_by_char(text)

        # Если текущего разделителя нет в тексте, переходим к следующему
        if not re.search(re.escape(current_separator), text): # Экранируем разделитель для regex
            return self._split_recursive_robust(text, remaining_separators)

        final_chunks = []
        current_chunk_content = []
        current_len = 0

        splits = text.split(current_separator)

        for i, split_part in enumerate(splits):
            part_len = len(split_part)
            # Если это не последняя часть, учитываем длину разделителя
            if i < len(splits) - 1:
                part_len += len(current_separator)

            if current_len + part_len <= self.chunk_size:
                current_chunk_content.append(split_part)
                current_len += part_len
            else:
                # Текущий чанк заполнен, сохраняем его
                if current_chunk_content:
                    full_chunk = current_separator.join(current_chunk_content).strip()
                    if full_chunk: # Добавляем только непустые чанки
                        final_chunks.append(full_chunk)

                    # Подготавливаем перекрытие для следующего чанка
                    if self.chunk_overlap > 0 and len(full_chunk) > self.chunk_overlap:
                        overlap_for_next = full_chunk[-self.chunk_overlap:]
                    else:
                        overlap_for_next = ""
                else:
                    overlap_for_next = "" # Нет предыдущего чанка для перекрытия

                # Если текущая часть сама по себе слишком велика, рекурсивно разбиваем её
                if part_len > self.chunk_size:
                    sub_chunks = self._split_recursive_robust(split_part, remaining_separators)
                    final_chunks.extend(sub_chunks)
                    current_chunk_content = [] # Сбрасываем, так как часть уже обработана
                    current_len = 0
                else:
                    # Начинаем новый чанк с текущей части, добавляя перекрытие
                    current_chunk_content = [overlap_for_next, split_part] if overlap_for_next else [split_part]
                    current_len = len(overlap_for_next) + part_len
                    # Если было перекрытие, нужно учесть потенциальный разделитель между ним и новой частью
                    if overlap_for_next and split_part:
                         current_len += len(current_separator)

        # Добавляем последний оставшийся чанк
        if current_chunk_content:
            full_chunk = current_separator.join(current_chunk_content).strip()
            if full_chunk:
                final_chunks.append(full_chunk)

        # Фильтрация пустых строк, которые могут появиться
        return [chunk for chunk in final_chunks if chunk]

    def _split_by_char(self, text: str) -> List[str]:
        chunks = []
        if not text:
            return chunks

        # Если текст меньше chunk_size, возвращаем его целиком
        if len(text) <= self.chunk_size:
            return [text]

        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(text), step):
            chunk = text[i : i + self.chunk_size]
            chunks.append(chunk)
        return [chunk for chunk in chunks if chunk] # Удаляем пустые чанки, если они есть