import psycopg2
import numpy as np
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
from typing import List, Dict, Tuple
import warnings
import nltk
from nltk.corpus import wordnet
import random

warnings.filterwarnings('ignore')

# Конфигурация подключения к БД (Необходимо ввести свои данные)
DB_CONFIG = {
    'host': '',
    'port': '', # здесь просто цифры без ''
    'database': '',
    'user': '',
    'password': ''
}

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')


# Классификатор тезисов на основе CatBoost с векторами релевантности
class ThesisClassifier:

    def __init__(self, model_path: str = None):
        self.model = None
        self.tfidf_vectorizer = None
        self.category_mapping = None
        self.reverse_mapping = None

        # Путь к сохраненной модели
        if model_path:
            self.load_model(model_path)


    # Загрузка данных из БД для обучения
    def load_data_from_db(self) -> Tuple[List[str], List[int]]:
        conn = None
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Загружаются тезисы с назначенными категориями
            query = """
                SELECT t.cleaned_text, t.id, c.id as category_id, c.raw_text as category_name
                FROM theses t
                JOIN categories c ON t.chosen_category_id = c.id
                WHERE t.cleaned_text IS NOT NULL 
                  AND t.cleaned_text != ''
                  AND t.chosen_category_id IS NOT NULL
                ORDER BY c.id
            """

            cursor.execute(query)
            results = cursor.fetchall()

            if not results:
                print("Не найдено размеченных тезисов для обучения")
                return [], []

            # Создаётся маппинг категорий
            categories = {}
            for _, _, cat_id, cat_name in results:
                if cat_id not in categories:
                    categories[cat_id] = cat_name

            # Сортировка категорий для создания индексов
            sorted_categories = sorted(categories.items())
            self.category_mapping = {cat_id: idx for idx, (cat_id, _) in enumerate(sorted_categories)}
            self.reverse_mapping = {idx: cat_id for cat_id, idx in self.category_mapping.items()}

            # Сбор данных
            texts = []
            labels = []

            for text, thesis_id, cat_id, cat_name in results:
                texts.append(text)
                labels.append(self.category_mapping[cat_id])


            print(f"Загружено данных: {len(texts)} тезисов")
            print(f"Количество категорий: {len(set(labels))}")

            # Вывод распределения по категориям
            unique, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique, counts):
                cat_id = self.reverse_mapping[label]
                cat_name = categories[cat_id][:50]  # Обрезаем длинные названия
                print(f" Категория {cat_id} ({cat_name}...): {count} тезисов")

            cursor.close()
            return texts, labels

        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            return [], []
        finally:
            if conn:
                conn.close()


    # Создание TF-IDF векторов признаков
    def create_tfidf_features(self, texts: List[str], fit: bool = True) -> np.ndarray:
        if fit:
            # Создание и обучение TF-IDF векторизатора
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,  # Максимальное количество признаков
                min_df=2,  # Минимальная частота документа
                max_df=0.95,  # Максимальная частота документа
                ngram_range=(1, 2),  # Униграммы и биграммы
                sublinear_tf=True  # Использование sublinear TF
            )
            features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            if self.tfidf_vectorizer is None:
                raise ValueError("Векторизатор не обучен. Сначала вызовите fit=True")
            features = self.tfidf_vectorizer.transform(texts)

        return features.toarray()


    # Создание статистических признаков на основе текста
    def create_statistical_features(self, texts: List[str]) -> np.ndarray:
        features = []

        for text in texts:
            words = text.split()
            word_count = len(words)
            char_count = len(text)
            avg_word_length = char_count / word_count if word_count > 0 else 0

            # Количество уникальных слов
            unique_words = len(set(words))

            features.append([
                word_count,
                char_count,
                avg_word_length,
                unique_words,
                unique_words / word_count if word_count > 0 else 0,  # Лексическое разнообразие
                len(set([w for w in words if len(w) > 6]))  # Количество длинных слов (>6 букв)
            ])

        return np.array(features)


    # Создание векторов релевантности между текстами тезисов и категориями
    def create_relevance_vectors(self, texts: List[str], category_texts: Dict[int, str]) -> np.ndarray:
        n_texts = len(texts)
        n_categories = len(category_texts)

        # Создание TF-IDF векторов для тезисов
        tfidf_theses = self.tfidf_vectorizer.transform(texts).toarray()
        # Создание TF-IDF векторов для категорий
        category_list = [category_texts[cat_id] for cat_id in sorted(category_texts.keys())]
        tfidf_categories = self.tfidf_vectorizer.transform(category_list).toarray()

        # Вычисление косинусного сходства между каждым тезисом и каждой категорией
        relevance_vectors = []
        for i in range(n_texts):
            # Нормализация вектора тезиса
            thesis_vec = tfidf_theses[i]
            thesis_norm = np.linalg.norm(thesis_vec)
            if thesis_norm == 0:
                thesis_norm = 1

            # Вычисление сходства со всеми категориями
            similarities = []
            for j in range(n_categories):
                category_vec = tfidf_categories[j]
                category_norm = np.linalg.norm(category_vec)
                if category_norm == 0:
                    category_norm = 1

                # Косинусное сходство
                similarity = np.dot(thesis_vec, category_vec) / (thesis_norm * category_norm)
                similarities.append(similarity)

            relevance_vectors.append(similarities)

        return np.array(relevance_vectors)


    # Подготовка всех признаков для классификации
    def prepare_features(self, texts: List[str], category_texts: Dict[int, str]) -> np.ndarray:
        # TF-IDF признаки
        tfidf_features = self.create_tfidf_features(texts, fit=(self.tfidf_vectorizer is None))
        # Статистические признаки
        stat_features = self.create_statistical_features(texts)
        # Векторы релевантности
        if category_texts:
            relevance_features = self.create_relevance_vectors(texts, category_texts)
            # Объединение всех признаков
            features = np.hstack([tfidf_features, stat_features, relevance_features])
        else:
            features = np.hstack([tfidf_features, stat_features])

        print(f"Создано признаков: {features.shape[1]}")
        return features


    # Обучение модели CatBoost
    def train(self,
              test_size: float = 0.2, # Размер тестовой выборки
              verbose: bool = True, # Вывод подробной информации
              catboost_params: Dict = None,
              min_samples_per_category: int = 3,  # Минимальное количество примеров в категории
              imbalance_threshold: float = 2.0) -> Dict: # Дополнительные параметры

        # Загрузка данных
        texts, labels = self.load_data_from_db()

        if not texts:
            raise ValueError("Нет данных для обучения")

        texts, labels = self.augment_minority_classes(
            texts,
            labels,
            min_samples=min_samples_per_category,
            imbalance_threshold=imbalance_threshold
        )

        # Загрузка текстов категорий для векторов релевантности
        category_texts = self._load_category_texts()

        # Подготовка признаков
        X = self.prepare_features(texts, category_texts)
        y = np.array(labels)

        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y
        )

        # Параметры CatBoost по умолчанию
        default_params = {
            'iterations': 500,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'MultiClass',
            'eval_metric': 'Accuracy',
            'verbose': verbose,
            'early_stopping_rounds': 200,
            'l2_leaf_reg': 3,
            'border_count': 128
        }
        if catboost_params:
            default_params.update(catboost_params)

        # Создание и обучение модели
        self.model = CatBoostClassifier(**default_params)

        # Для валидации используется часть обучающей выборки
        fit_params = {
            'eval_set': [(X_test, y_test)],
            'plot': not verbose  # Отключает графики в консольном режиме
        }

        self.model.fit(X_train, y_train, **fit_params)

        # Оценка модели
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        metrics = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': self.model.get_feature_importance().tolist(),
            'best_iteration': self.model.get_best_iteration(),
            'best_score': self.model.get_best_score()
        }

        print(f"\n{'=' * 50}")
        print(f"Точность модели: {accuracy:.4f}")
        print(f"{'=' * 50}")
        print("\nОтчет по классификации:")
        print(metrics['classification_report'])

        return metrics


    # Увеличение количества примеров для малочисленных классов через аугментацию текста
    def augment_minority_classes(self, texts: List[str], labels: List[int], min_samples: int = 5,
                                 imbalance_threshold: float = 2.0) -> Tuple[List[str], List[int]]:
        # Подсчет примеров по классам
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Нахождение максимального количества примеров в категории
        max_count = max(counts)
        min_count = min(counts)

        print(f"\nАнализ дисбаланса классов:")
        print(f"  Максимальное количество примеров: {max_count}")
        print(f"  Минимальное количество примеров: {min_count}")
        print(f"  Коэффициент дисбаланса: {max_count / min_count:.2f}")

        augmented_texts = list(texts)
        augmented_labels = list(labels)

        total_augmented = 0

        for label in unique_labels:
            current_count = counts[list(unique_labels).index(label)]

            # Определение целевого количества примеров для этого класса
            target_count = current_count

            # Проверка дисбаланса относительно самой заполненной категории
            if max_count / current_count > imbalance_threshold:
                # Если дисбаланс превышает порог, увеличиваем до max_count / imbalance_threshold
                target_count = max(int(max_count / imbalance_threshold), target_count)
                print(f"  Категория {label}: дисбаланс {max_count / current_count:.2f} > {imbalance_threshold}")

            # Применяется абсолютный минимум, если он больше
            target_count = max(target_count, min_samples)

            # Если нужно увеличить количество примеров
            if current_count < target_count:
                # Найти все примеры этого класса
                class_texts = [texts[i] for i, l in enumerate(labels) if l == label]
                samples_needed = target_count - current_count

                cat_id = self.reverse_mapping[label]
                print(f"\n  Аугментация категории {cat_id} (label {label}):")
                print(f"    Текущее количество: {current_count}")
                print(f"    Целевое количество: {target_count}")
                print(f"    Нужно добавить: {samples_needed} примеров")

                augmented_for_class = 0
                attempts = 0
                max_attempts = samples_needed * 3  # Ограничение на количество попыток

                while augmented_for_class < samples_needed and attempts < max_attempts:
                    # Выбрать случайный текст из этого класса
                    original_text = random.choice(class_texts)
                    words = original_text.split()

                    # Разные стратегии аугментации
                    augmentation_type = random.choice(['synonyms', 'shuffle', 'delete', 'both'])

                    if augmentation_type == 'synonyms':
                        # Замена синонимами случайных слов
                        augmented_words = []
                        for word in words:
                            if len(word) > 3 and random.random() < 0.3:  # 30% шанс замены
                                synonyms = []
                                for syn in wordnet.synsets(word):
                                    for lemma in syn.lemmas():
                                        if lemma.name() != word and '_' not in lemma.name():
                                            synonyms.append(lemma.name().replace('_', ' '))
                                if synonyms:
                                    augmented_words.append(random.choice(synonyms))
                                else:
                                    augmented_words.append(word)
                            else:
                                augmented_words.append(word)
                        augmented_text = ' '.join(augmented_words)

                    elif augmentation_type == 'shuffle' and len(words) > 5:
                        # Перемешивание порядка слов (только для части текста)
                        split_point = len(words) // 3
                        middle_section = words[split_point:-split_point]
                        random.shuffle(middle_section)
                        augmented_words = words[:split_point] + middle_section + words[-split_point:]
                        augmented_text = ' '.join(augmented_words)

                    elif augmentation_type == 'delete' and len(words) > 10:
                        # Удаление случайных слов
                        keep_probability = 0.8
                        augmented_words = [word for word in words if random.random() < keep_probability]
                        augmented_text = ' '.join(augmented_words) if augmented_words else original_text

                    else:  # 'both' или fallback
                        # Комбинация синонимов и перемешивания
                        augmented_words = []
                        for word in words:
                            if len(word) > 3 and random.random() < 0.2:
                                synonyms = []
                                for syn in wordnet.synsets(word):
                                    for lemma in syn.lemmas():
                                        if lemma.name() != word and '_' not in lemma.name():
                                            synonyms.append(lemma.name().replace('_', ' '))
                                if synonyms:
                                    augmented_words.append(random.choice(synonyms))
                                else:
                                    augmented_words.append(word)
                            else:
                                augmented_words.append(word)

                        if len(augmented_words) > 5 and random.random() < 0.5:
                            # Частичное перемешивание
                            start_idx = random.randint(0, len(augmented_words) - 3)
                            end_idx = min(start_idx + random.randint(2, 4), len(augmented_words))
                            section = augmented_words[start_idx:end_idx]
                            random.shuffle(section)
                            augmented_words[start_idx:end_idx] = section

                        augmented_text = ' '.join(augmented_words)

                    # Проверка, что аугментированный текст отличается от оригинала
                    if augmented_text != original_text and len(augmented_text.split()) >= 3:
                        augmented_texts.append(augmented_text)
                        augmented_labels.append(label)
                        augmented_for_class += 1
                        total_augmented += 1

                    attempts += 1

                if augmented_for_class < samples_needed:
                    print(
                        f"    Предупреждение: удалось добавить только {augmented_for_class} из {samples_needed} примеров")

        print(f"\nИтоги аугментации:")
        print(f"  Всего добавлено примеров: {total_augmented}")
        print(f"  Исходное количество: {len(texts)}")
        print(f"  Итоговое количество: {len(augmented_texts)}")

        # Вывод финального распределения
        final_unique, final_counts = np.unique(augmented_labels, return_counts=True)
        final_max = max(final_counts)
        final_min = min(final_counts)
        print(f"  Финальный коэффициент дисбаланса: {final_max / final_min:.2f}")

        return augmented_texts, augmented_labels


    # Загрузка текстов категорий из БД
    def _load_category_texts(self) -> Dict[int, str]:
        conn = None
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, cleaned_text 
                FROM categories 
                WHERE cleaned_text IS NOT NULL AND cleaned_text != ''
            """)

            results = cursor.fetchall()
            category_texts = {cat_id: text for cat_id, text in results}

            cursor.close()
            return category_texts

        except Exception as e:
            print(f"Ошибка при загрузке категорий: {e}")
            return {}
        finally:
            if conn:
                conn.close()


    # Сохранение модели и векторизатора
    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("Нет модели для сохранения")

        model_data = {
            'model': self.model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'category_mapping': self.category_mapping,
            'reverse_mapping': self.reverse_mapping
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Модель сохранена в {path}")


    # Загрузка модели и векторизатора
    def load_model(self, path: str):
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.category_mapping = model_data['category_mapping']
        self.reverse_mapping = model_data['reverse_mapping']

        print(f"Модель загружена из {path}")



def main():
    # Создание и обучение классификатора
    classifier = ThesisClassifier()

    # Параметры для CatBoost
    catboost_params = {
        'iterations': 300,
        'learning_rate': 0.05,
        'depth': 5,
        'l2_leaf_reg': 5,
        'border_count': 128
    }

    # Обучение модели
    print("Начало обучения модели...")
    metrics = classifier.train(test_size=0.2, verbose=True, catboost_params=catboost_params)

    # Сохраняем модель
    classifier.save_model("thesis_classifier_catboost.pkl")


if __name__ == "__main__":
    main()