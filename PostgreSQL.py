import psycopg2
import re
import os
from typing import Set, Dict
import docx
from nltk.corpus import stopwords
import nltk
import pymorphy2



# Конфигурация подключения к БД (можно ввести свои данные)
# Используется только для тестирования
DB_CONFIG = {
    'host': '',
    'port': '', # здесь просто цифры без ''
    'database': '',
    'user': '',
    'password': ''
}

# Скачивание стоп-слов для русского языка (только при первом запуске)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


# Создание таблиц для хранения тезисов и категорий
def create_tables(db_config):
    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Удаляются таблицы, если они существуют
        cursor.execute("""
            DROP TABLE IF EXISTS thesis_authors CASCADE;
            DROP TABLE IF EXISTS theses CASCADE;
            DROP TABLE IF EXISTS categories CASCADE;
            DROP TABLE IF EXISTS source_files CASCADE;
            DROP TABLE IF EXISTS authors CASCADE;
        """)

        # Создание таблицы исходных файлов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS source_files (
                id SERIAL PRIMARY KEY,
                file_content BYTEA,
                file_name VARCHAR(500) NOT NULL,
                file_type VARCHAR(50)
            );
        """)

        # Создание таблицы авторов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS authors (
                id SERIAL PRIMARY KEY,
                author VARCHAR(100) UNIQUE
            );
        """)

        # Создание таблицы категорий
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id SERIAL PRIMARY KEY,
                raw_text VARCHAR(500) UNIQUE NOT NULL,
                source_file_id INTEGER REFERENCES source_files(id) ON DELETE CASCADE,
                cleaned_text VARCHAR(500)
            );
        """)

        # Создание таблицы тезисов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS theses (
                id SERIAL PRIMARY KEY,
                raw_text TEXT NOT NULL,
                source_file_id INTEGER REFERENCES source_files(id) ON DELETE CASCADE,
                cleaned_text TEXT,
                chosen_category_id INTEGER REFERENCES categories(id) ON DELETE SET NULL,
                probability_vector JSONB,
                search_vector tsvector
            );
        """)

        # Связующая таблица для связи многие-ко-многим между тезисами и авторами
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS thesis_authors (
                thesis_id INTEGER REFERENCES theses(id) ON DELETE CASCADE,
                author_id INTEGER REFERENCES authors(id) ON DELETE CASCADE,
                PRIMARY KEY (thesis_id, author_id)
            );
        """)

        conn.commit()
        print("Таблицы успешно созданы")

    except Exception as e:
        print(f"Ошибка при создании таблиц: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            cursor.close()
            conn.close()


#region Загрузка файлов в БД

def text_upload_from_file(text_type: int, file_paths, db_config):
    conn = None
    try:
        # Подключение к БД
        conn = psycopg2.connect(**db_config)

        # Определение типа файла по расширению
        for file_path in file_paths:
            if file_path.endswith('.txt'):
                source_file_id = save_file_to_db(file_path, 'txt', conn)
                if text_type == 1:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        process_authors_file(content, 1, conn)
                elif text_type == 2:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        process_categories_file(content, 1, source_file_id, conn)
                elif text_type == 3:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        process_theses_file(content, 1, source_file_id, conn)

            elif file_path.endswith('.docx'):
                source_file_id = save_file_to_db(file_path, 'docx',conn)
                if text_type == 1:
                    content = docx.Document(file_path)
                    process_authors_file(content, 2, conn)
                elif text_type == 2:
                    content = docx.Document(file_path)
                    process_categories_file(content, 2, source_file_id,conn)
                elif text_type == 3:
                    content = docx.Document(file_path)
                    process_theses_file(content, 2, source_file_id, conn)
            else:
                file_type = 'unknown'
                save_file_to_db(file_path, 'unknown', conn)
    finally:
        if conn:
            conn.close()


# region Загрузка авторов из файла

# Загрузка авторов в БД
def load_authors_to_db(authors, conn):
    try:
        cursor = conn.cursor()

        # Загрузка каждого автора в таблицу
        for author in authors:
            cursor.execute("""
                INSERT INTO authors (author) 
                VALUES (%s)
            """, (author,))
        conn.commit()
        print(f"Успешно загружено {len(authors)} авторов")
        cursor.close()

    except Exception as e:
        print(f"Ошибка при загрузке авторов: {e}")
        conn.rollback()
        raise


# Извлечение авторов из файла и загрузка их в БД
def process_authors_file(content, file_type: int, conn):
    should_close = False
    if conn is None:
        conn = psycopg2.connect(**DB_CONFIG)
        should_close = True

    authors = []
    # Извлечение авторов из файла
    try:
        if file_type == 2:
            lines = [p.text for p in content.paragraphs] # docx.Document список строк из параграфов
        else:
            lines = content.split('\n') # txt split по переносам строк

        for line in lines:
            line = line.strip()
            if not line:  # Пропус пустых строк
                continue
            if line.isdigit(): # Проверка, не является ли строка просто числом
                continue
            # Убираются числа в конце строки
            line = re.sub(r'\s+\d+$', '', line)
            # Если строка содержит только число, то она пропускается
            if line.isdigit():
                continue
            # Паттерн 1: Фамилия И.О.
            pattern_with_initials = r'^[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.$'
            # Паттерн 2: Фамилия И.
            pattern_one_initial = r'^[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.$'
            # Паттерн 3: Фамилия Имя Отчество
            pattern_full_name = r'^[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+$'
            # Паттерн 4: Только фамилия
            pattern_only_surname = r'^[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]$'
            if re.match(pattern_with_initials, line):
                authors.append(line)
            elif re.match(pattern_one_initial, line):
                authors.append(line)
            elif re.match(pattern_full_name, line):
                authors.append(line)
            elif re.match(pattern_only_surname, line):
                authors.append(line)

        # Удаление дубликатов, сохраняя порядок
        unique_authors = []
        seen = set()
        for author in authors:
            if author not in seen:
                seen.add(author)
                unique_authors.append(author)
        print(f"Найдено уникальных авторов: {len(unique_authors)}")

        if not unique_authors:
            print("Авторы не найдены в файле")
            return

        # Загрузка в БД
        load_authors_to_db(unique_authors, conn)
        print(f"Обработка файла авторов завершена")

    except Exception as e:
        print(f"Ошибка при обработке: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if should_close and conn:
            conn.close()

# endregion


#region Загрузка категорий из файла и загрузка в БД

# Загрузка категорий в БД
def load_categories_to_db(categories, source_file_id: int, conn):
    try:
        cursor = conn.cursor()

        # Загрузка каждого тезиса в таблицу
        for categ in categories:
            cursor.execute("""
                INSERT INTO categories (raw_text, source_file_id) 
                VALUES (%s, %s)
            """, (categ, source_file_id))
        conn.commit()
        print(f"Успешно загружено {len(categories)} категорий")
        cursor.close()

    except Exception as e:
        print(f"Ошибка при загрузке категорий: {e}")
        conn.rollback()
        raise


# Извлечение категорий из файла и загрузка их в БД
def process_categories_file(content, file_type: int, source_file_id: int, conn):
    should_close = False
    if conn is None:
        conn = psycopg2.connect(**DB_CONFIG)
        should_close = True

    categories = []
    try:
        if file_type == 2:
            current_block = []
            for p in content.paragraphs:
                text = p.text.strip()
                if text:
                    current_block.append(text)
                else:
                    if current_block:
                        categories.append('\n'.join(current_block))
                        current_block = []
            if current_block:
                categories.append('\n'.join(current_block))
        else:
            # Извлечение категорий из файла
            normalized = content.replace('\r\n', '\n').replace('\r', '\n')
            blocks = re.split(r'\n\s*\n', normalized)
            # Очистка каждого блока от лишних пробелов и переносов строк
            for block in blocks:
                block = block.strip()
                if block:
                    categories.append(block)

        print(f"Найдено категорий: {len(categories)}")

        if not categories:
            print("Категории не найдены в файле")
            return

        # Загрузка в БД
        load_categories_to_db(categories, source_file_id, conn)

        print(f"Обработка файла с категориями завершена")

    except Exception as e:
        print(f"Ошибка при обработке: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if should_close and conn:
            conn.close()

#endregion


#region Загрузка тезисов из файла Word и загрузка в БД

# Поиск авторов в тексте тезиса
def find_authors_in_thesis(thesis_text: str, authors_dict: Dict[str, int]) -> Set[int]:
    found_authors = set()

    for author_name, author_id in authors_dict.items():
        # Разные варианты написания инициалов
        variations = [
            author_name,  # "Фамилия И.О."
            author_name.replace('. ', '.'),  # "Фамилия И. О."
            author_name.replace('. ', '')  # "Фамилия ИО"
        ]

        for variation in variations:
            if variation in thesis_text:
                found_authors.add(author_id)
                break

    return found_authors


# Загрузка тезисов в БД
def load_theses_to_db(theses, source_file_id: int, conn, authors_dict: Dict[str, int] = None):
    try:
        cursor = conn.cursor()

        # Загрузка каждого тезиса в таблицу
        for thesis in theses:
            cursor.execute("""
                INSERT INTO theses (raw_text, source_file_id) 
                VALUES (%s, %s)
                RETURNING id
            """, (thesis, source_file_id))

            # Получение ID текущего тезиса
            thesis_id = cursor.fetchone()[0]

            # Если есть словарь авторов, ищутся авторы в тезисе
            if authors_dict:
                found_authors = find_authors_in_thesis(thesis, authors_dict)
                # Создание связи между тезисом и найденными авторами
                for author_id in found_authors:
                    cursor.execute("""
                        INSERT INTO thesis_authors (thesis_id, author_id)
                        VALUES (%s, %s)
                        ON CONFLICT DO NOTHING
                    """, (thesis_id, author_id))

        conn.commit()
        print(f"Успешно загружено {len(theses)} тезисов")
        cursor.close()

    except Exception as e:
        print(f"Ошибка при загрузке тезисов: {e}")
        conn.rollback()
        raise


# Извлечение авторов из файла и загрузка их в БД """
def process_theses_file(content, file_type: int, source_file_id: int, conn):
    should_close = False
    if conn is None:
        conn = psycopg2.connect(**DB_CONFIG)
        should_close = True

    try:
        # Получение всех авторов из БД
        authors_dict = get_all_authors(conn)
        print(f"Загружено авторов для поиска: {len(authors_dict)}")

        theses = []
        current_thesis = []

        # Извлечение тезисов из файла docx
        if file_type == 2:
            # Группировка параграфов в тезисы
            for p in content.paragraphs:
                text = p.text.strip()
                if text:  # Непустой параграф
                    current_thesis.append(text)
                else:  # Пустая строка - разделитель тезисов
                    if current_thesis:
                        # Объединение параграфов текущего тезиса
                        thesis_text = '\n'.join(current_thesis)
                        theses.append(thesis_text)
                        current_thesis = []
        else:
            # Извлечение из файла txt
            for line in content.split('\n'):
                text = line.strip()
                if text:
                    current_thesis.append(text)
                else:
                    if current_thesis:
                        thesis_text = '\n'.join(current_thesis)
                        theses.append(thesis_text)
                        current_thesis = []


        # Добавление последнего тезиса, если он есть
        if current_thesis:
            thesis_text = '\n'.join(current_thesis)
            theses.append(thesis_text)

        if not theses:
            print("Тезисы не найдены в файле")
            return


        # Загрузка в БД
        load_theses_to_db(theses, source_file_id, conn, authors_dict)
        print(f"Обработка файла с тезисами завершена")

    except Exception as e:
        print(f"Ошибка при обработке: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if should_close and conn:
            conn.close()

#endregion



# Получение всех авторов из БД в виде словаря
def get_all_authors(conn) -> Dict[str, int]:
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, author FROM authors")
        authors = cursor.fetchall()
        cursor.close()
        return {author_name: author_id for author_id, author_name in authors}

    except Exception as e:
        print(f"Ошибка при получении авторов: {e}")
        return {}


# Сохранение файла в базу данных и возвращение его ID
def save_file_to_db(file_path: str, file_type: str, conn) -> int:
    try:
        cursor = conn.cursor()

        # Считывание файла в бинарном виде
        with open(file_path, 'rb') as file:
            file_content = file.read()

        # Получение имени файла
        file_name = os.path.basename(file_path)
        # Проверка, не сохранен ли уже этот файл
        cursor.execute("""
            SELECT id FROM source_files WHERE file_name = %s
        """, (file_name,))

        existing = cursor.fetchone()
        if existing:
            print(f"Файл {file_name} уже существует в БД (ID: {existing[0]})")
            return existing[0]

        # Сохранение файла
        cursor.execute("""
            INSERT INTO source_files (file_name, file_content, file_type)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (file_name, psycopg2.Binary(file_content), file_type))

        file_id = cursor.fetchone()[0]
        conn.commit()

        print(f"Файл {file_name} сохранен в БД (ID: {file_id})")
        cursor.close()
        return file_id

    except Exception as e:
        print(f"Ошибка при сохранении файла {file_path}: {e}")
        conn.rollback()
        raise

#endregion


#region Обработка тезисов/категорий (лемматизация, удаление стоп-слов, приведение к нижнему регистру)

# Класс для предобработки текста тезисов
class TextProcessor:
    def __init__(self):
        # Инициализация морфологического анализатора
        self.morph = pymorphy2.MorphAnalyzer()
        # Загрузка стоп-слов для русского языка
        self.stop_words = set(stopwords.words('russian'))
        # Добавление дополнительных стоп-слов
        self.stop_words.update(['это', 'так', 'все', 'них', 'быть', 'весь'])

    # Приведение к нижнему регистру, удаление спецсимволов, лишних пробелов и цифр
    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        # Приведение к нижнему регистру
        text = text.lower()
        # Удаление спецсимволов, буквы, цифры и пробелы остаются
        text = re.sub(r'[^\w\s]', ' ', text)
        # Удаление цифр
        text = re.sub(r'\d+', '', text)
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()

        return text


    # Лемматизация текста
    def lemmatize_text(self, text: str, remove_stopwords: bool = True) -> str:
        if not text:
            return ""

        # Разделение на слова
        words = text.split()
        lemmatized_words = []
        for word in words:
            # Пропуск коротких слов
            if len(word) < 2:
                continue

            # Удаление стоп-слов
            if remove_stopwords and word in self.stop_words:
                continue

            # Приведение к нормальной форме
            lemma = self.morph.parse(word)[0].normal_form
            lemmatized_words.append(lemma)

        return ' '.join(lemmatized_words)


    # Обработка текста
    def process_text(self, text: str, remove_stopwords: bool = True) -> str:
        if not text:
            return ""

        # Очистка текста
        cleaned = self.clean_text(text)
        # Лемматизация
        lemmatized = self.lemmatize_text(cleaned, remove_stopwords)

        return lemmatized


# Получение и загрузка текстов в БД
def process_texts_from_db(db_config, process_theses: bool = True, remove_stopwords: bool = True) -> Dict:
    # process_theses: Если True, то обрабатывает тезисы, если False, то категории
    processor = TextProcessor()
    conn = None

    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        if process_theses:
            cursor.execute("""
                SELECT id, raw_text 
                FROM theses 
                WHERE raw_text IS NOT NULL
            """)
        else:
            cursor.execute("""
                SELECT id, raw_text 
                FROM categories
                WHERE raw_text IS NOT NULL
            """)

        texts = cursor.fetchall()
        print(f"Найдено текстов для обработки: {len(texts)}")

        # Обработка текстов
        batch_updates = []
        for text_id, raw_text in texts:
            try:
                cleaned = processor.process_text(raw_text, remove_stopwords)
                if cleaned:  # Если после обработки текст не пустой, он сохраняется
                    batch_updates.append((cleaned, text_id))

            except Exception as e:
                print(f"Ошибка при обработке тезиса {text_id}: {e}")

        # Обновление базы данных
        if batch_updates:
            if process_theses:
                cursor.executemany("""
                    UPDATE theses 
                    SET cleaned_text = %s 
                    WHERE id = %s
                """, batch_updates)
                conn.commit()
            else:
                cursor.executemany("""
                    UPDATE categories
                    SET cleaned_text = %s 
                    WHERE id = %s
                """, batch_updates)
                conn.commit()

            print(f"Обработано {len(texts)}")
        cursor.close()

    except Exception as e:
        print(f"Ошибка при обработке: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

#endregion


#region Тестирование

"""
# Назначает категорию тезисам в указанном диапазоне ID
def assign_categories_to_theses_by_range(start_id: int, end_id: int, category_id: int, conn=None):
    should_close = False
    if conn is None:
        conn = psycopg2.connect(**DB_CONFIG)
        should_close = True

    try:
        cursor = conn.cursor()

        # Проверка, существует ли категория
        cursor.execute("SELECT id FROM categories WHERE id = %s", (category_id,))
        if not cursor.fetchone():
            print(f"Ошибка: Категория с ID {category_id} не найдена")
            return

        # Обновление тезисов в указанном диапазоне
        cursor.execute("""
"""         UPDATE theses 
            SET chosen_category_id = %s 
            WHERE id BETWEEN %s AND %s
            RETURNING id
        """#, (category_id, start_id, end_id))
"""
        updated_ids = cursor.fetchall()
        conn.commit()

        print(f"Категория ID {category_id} назначена {len(updated_ids)} тезисам (ID: {start_id} - {end_id})")

        cursor.close()

    except Exception as e:
        print(f"Ошибка при назначении категорий: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if should_close and conn:
            conn.close()


# Считывает данные с файла и вызывает функцию для назначения категории тезисам в указанном диапазоне ID
def assign_categories_from_file(mapping_file_path: str):
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)

        # Считывание данных из файла
        with open(mapping_file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()

                if len(parts) == 3:
                    # Диапазон: start_id end_id category_id
                    start_id, end_id, category_id = map(int, parts)
                    assign_categories_to_theses_by_range(start_id, end_id, category_id, conn)
                elif len(parts) == 2:
                    # Один тезис: thesis_id category_id
                    thesis_id, category_id = map(int, parts)
                    assign_categories_to_theses_by_range(thesis_id, thesis_id, category_id, conn)
                else:
                    print(f"Ошибка в строке {line_num}: неверный формат '{line}'")

    except FileNotFoundError:
        print(f"Файл {mapping_file_path} не найден")
    except Exception as e:
        print(f"Ошибка: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
"""
#endregion


#Тестирование
"""
if __name__ == "__main__":
    # Создание таблиц
    create_tables()

    authors_file = []
    categories_file = []
    theses_files = []

    # Пути к файлам
    authors_file.append(
        "C:/")

    categories_file.append(
        "C:/")

    theses_files.append(
        "C:/")

    # Загрузка авторов из файла
    text_upload_from_file(1, authors_file)

    # Загрузка категорий из файла
    text_upload_from_file(2, categories_file)

    # Загрузка тезисов из файлов
    text_upload_from_file(3, theses_files)

    # Предобработка текстов
    processed = process_texts_from_db(True)
    processed = process_texts_from_db(False)


    # Искусственное присвоение категорий для теста
    #assign_categories_from_file("C:/Users/Hjg/Desktop/Универ/4 курс/Диплом/Программа/meow_abstract2025.txt")
"""