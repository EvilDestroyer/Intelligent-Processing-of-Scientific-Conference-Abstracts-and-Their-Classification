import psycopg2
import numpy as np
import pickle
import json
from typing import List, Dict, Tuple, Optional
import warnings
import time

warnings.filterwarnings('ignore')

# Конфигурация подключения к БД (можно ввести свои данные)
# Используется только для тестирования
DB_CONFIG = {
    'host': '',
    'port': '', # здесь просто цифры без ''
    'database': '',
    'user': '',
    'password': ''
}

# Класс для классификации тезисов с использованием обученной модели
class ThesisClassifier:
    def __init__(self, model_path: str, db_config):
        self.db_config = db_config
        self.model = None
        self.tfidf_vectorizer = None
        self.category_mapping = None
        self.reverse_mapping = None

        self.timing_stats = {
            'load_time': 0.0,
            'prediction_time': 0.0,
            'db_update_time': 0.0,
            'rebalance_time': 0.0,
            'total_time': 0.0
        }

        self._load_model(model_path)


    # Загрузка модели из файла
    def _load_model(self, model_path: str):
        start_time = time.time()
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.category_mapping = model_data['category_mapping']
            self.reverse_mapping = model_data['reverse_mapping']

            print(f"Модель успешно загружена из {model_path}")
            print(f"Количество категорий: {len(self.category_mapping)}")

        except Exception as e:
            raise ValueError(f"Ошибка при загрузке модели: {e}")
        finally:
            self.timing_stats['load_time'] = time.time() - start_time
            print(f"Время загрузки модели: {self.timing_stats['load_time']:.2f} сек")


    # Загрузка текстов категорий из БД
    def _load_category_texts(self) -> Dict[int, str]:
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, cleaned_text 
                FROM categories 
                WHERE cleaned_text IS NOT NULL AND cleaned_text != ''
            """)

            results = cursor.fetchall()
            return {cat_id: text for cat_id, text in results}

        except Exception as e:
            print(f"Ошибка при загрузке категорий: {e}")
            return {}
        finally:
            if conn:
                conn.close()


    # Создание TF-IDF векторов признаков
    def _create_tfidf_features(self, texts: List[str]) -> np.ndarray:
        features = self.tfidf_vectorizer.transform(texts)
        return features.toarray()


    # Создание статистических признаков
    def _create_statistical_features(self, texts: List[str]) -> np.ndarray:
        features = []

        for text in texts:
            words = text.split()
            word_count = len(words)
            char_count = len(text)
            avg_word_length = char_count / word_count if word_count > 0 else 0
            unique_words = len(set(words))

            features.append([
                word_count,
                char_count,
                avg_word_length,
                unique_words,
                unique_words / word_count if word_count > 0 else 0,
                len(set([w for w in words if len(w) > 6]))
            ])

        return np.array(features)


    # Создание векторов релевантности
    def _create_relevance_vectors(self, texts: List[str], category_texts: Dict[int, str]) -> np.ndarray:
        n_texts = len(texts)
        n_categories = len(category_texts)

        tfidf_theses = self.tfidf_vectorizer.transform(texts).toarray()
        category_list = [category_texts[cat_id] for cat_id in sorted(category_texts.keys())]
        tfidf_categories = self.tfidf_vectorizer.transform(category_list).toarray()

        relevance_vectors = []

        for i in range(n_texts):
            thesis_vec = tfidf_theses[i]
            thesis_norm = np.linalg.norm(thesis_vec) or 1

            similarities = []
            for j in range(n_categories):
                category_vec = tfidf_categories[j]
                category_norm = np.linalg.norm(category_vec) or 1
                similarity = np.dot(thesis_vec, category_vec) / (thesis_norm * category_norm)
                similarities.append(similarity)

            relevance_vectors.append(similarities)

        return np.array(relevance_vectors)


    # Подготовка всех признаков для классификации
    def _prepare_features(self, texts: List[str]) -> np.ndarray:
        # Загружаем тексты категорий
        category_texts = self._load_category_texts()

        # TF-IDF признаки
        tfidf_features = self._create_tfidf_features(texts)

        # Статистические признаки
        stat_features = self._create_statistical_features(texts)

        # Векторы релевантности
        if category_texts:
            relevance_features = self._create_relevance_vectors(texts, category_texts)
            features = np.hstack([tfidf_features, stat_features, relevance_features])
        else:
            features = np.hstack([tfidf_features, stat_features])

        return features


    # Предсказание категорий для новых текстов
    def predict(self, texts: List[str], return_proba: bool = False) -> np.ndarray:
        if self.model is None:
            raise ValueError("Модель не загружена")

        # Подготовка признаков
        X = self._prepare_features(texts)

        if return_proba:
            predictions = self.model.predict_proba(X)
            result = []
            for probs in predictions:
                cat_probs = {self.reverse_mapping[i]: prob for i, prob in enumerate(probs)}
                result.append(cat_probs)
            return np.array(result)
        else:
            predictions = self.model.predict(X)
            return np.array([self.reverse_mapping[pred] for pred in predictions])


    # Обновление предсказаний для тезисов в БД
    def update_thesis_predictions(self, threshold: float = 0.0):
        start_time = time.time()
        prediction_start = time.time()
        db_start = time.time()
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Получение тезисов без назначенной категории
            query = """
                SELECT id, cleaned_text 
                FROM theses 
                WHERE cleaned_text IS NOT NULL 
                  AND cleaned_text != ''
                  AND (chosen_category_id IS NULL OR chosen_category_id = 0)
            """

            cursor.execute(query)
            theses_to_classify = cursor.fetchall()

            if not theses_to_classify:
                print("Нет тезисов для классификации")
                return

            print(f"Классификация {len(theses_to_classify)} тезисов...")

            # Классификация всех тезисов
            texts = [thesis[1] for thesis in theses_to_classify]
            pred_start = time.time()
            probabilities = self.predict(texts, return_proba=True)
            prediction_time = time.time() - pred_start
            self.timing_stats['prediction_time'] = prediction_time

            print(f"Время предсказания: {prediction_time:.2f} сек")
            print(f"Среднее время на тезис: {prediction_time / len(texts) * 1000:.2f} мс")


            # Обновление БД
            db_update_start = time.time()
            updated_count = 0
            for (thesis_id, _), probs in zip(theses_to_classify, probabilities):
                # Сохранение вектора вероятностей в JSON
                prob_vector_json = json.dumps(probs, ensure_ascii=False)

                # Если максимальная вероятность выше порога, назначается категория
                max_prob_category = max(probs, key=probs.get)
                max_prob = probs[max_prob_category]

                if max_prob >= threshold:
                    cursor.execute("""
                        UPDATE theses 
                        SET probability_vector = %s::jsonb,
                            chosen_category_id = %s
                        WHERE id = %s
                    """, (prob_vector_json, max_prob_category, thesis_id))
                else:
                    cursor.execute("""
                        UPDATE theses 
                        SET probability_vector = %s::jsonb
                        WHERE id = %s
                    """, (prob_vector_json, thesis_id))

                updated_count += 1

                if updated_count % 50 == 0:
                    print(f"Обработано {updated_count}/{len(theses_to_classify)} тезисов")

            conn.commit()
            db_update_time = time.time() - db_update_start
            self.timing_stats['db_update_time'] = db_update_time

            total_time = time.time() - start_time
            self.timing_stats['total_time'] = total_time

            print(f"Обновлено {updated_count} тезисов")
            print(f"Время обновления БД: {db_update_time:.2f} сек")
            print(f"Общее время выполнения: {total_time:.2f} сек")

            # Вывод статистики
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_classified,
                    COUNT(CASE WHEN chosen_category_id IS NOT NULL THEN 1 END) as auto_assigned
                FROM theses
                WHERE probability_vector IS NOT NULL
            """)

            total, auto_assigned = cursor.fetchone()
            print(f"\nСтатистика:")
            print(f"  Всего классифицировано: {total}")
            print(f"  Автоматически назначено: {auto_assigned} (порог: {threshold})")

            cursor.close()

        except Exception as e:
            print(f"Ошибка при обновлении предсказаний: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()


    # Перераспределение тезисов для выравнивания количества в категориях
    def rebalance_theses_distribution(self,
                                      target_balance_ratio: float = 0.6,
                                      min_confidence_threshold: float = 0.1,
                                      max_iterations: int = 10):
        start_time = time.time()
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            # Получение текущего распределения по категориям
            cursor.execute("""
                SELECT c.id, c.raw_text, COUNT(t.id) as thesis_count
                FROM categories c
                LEFT JOIN theses t ON t.chosen_category_id = c.id
                WHERE c.cleaned_text IS NOT NULL
                GROUP BY c.id, c.raw_text
                ORDER BY thesis_count DESC
            """)

            category_stats = cursor.fetchall()

            if len(category_stats) < 2:
                print("Недостаточно категорий для балансировки")
                return

            # Вычисление статистики
            counts = [stat[2] for stat in category_stats]
            min_count = min(counts)
            max_count = max(counts)
            avg_count = sum(counts) / len(counts)

            print(f"\n{'=' * 60}")
            print("ТЕКУЩЕЕ РАСПРЕДЕЛЕНИЕ ТЕЗИСОВ")
            print(f"{'=' * 60}")
            for cat_id, cat_text, count in category_stats:
                cat_name = cat_text[:40] if cat_text else f"Category_{cat_id}"
                bar = "█" * int(count / max(counts) * 50) if max(counts) > 0 else ""
                print(f"Категория {cat_id:3} ({cat_name:40}): {count:4} тезисов {bar}")

            print(f"\nМин: {min_count}, Макс: {max_count}, Среднее: {avg_count:.1f}")
            print(f"Соотношение мин/макс: {min_count / max_count:.3f}")

            # Проверка, нужно ли балансировать
            if min_count / max_count >= target_balance_ratio:
                print(
                    f"\n Распределение уже сбалансировано (соотношение {min_count / max_count:.3f} >= {target_balance_ratio})")
                return

            print(f"\n Обнаружен дисбаланс! Перераспределение...")

            # Нахождение переполненных и недополненных категорий
            overflow_threshold = avg_count * 1.2
            underflow_threshold = avg_count * 0.8

            overflow_categories = [stat for stat in category_stats if stat[2] > overflow_threshold]
            underflow_categories = [stat for stat in category_stats if stat[2] < underflow_threshold]

            print(f"\nКатегории с избытком (> {overflow_threshold:.1f}): {len(overflow_categories)}")
            print(f"Категории с недостатком (< {underflow_threshold:.1f}): {len(underflow_categories)}")

            # Итеративное перераспределение
            total_moved = 0
            for iteration in range(max_iterations):
                print(f"\n{'=' * 60}")
                print(f"ИТЕРАЦИЯ {iteration + 1}")
                print(f"{'=' * 60}")

                moved_count = 0

                # Обновление статистики для текущей итерации
                cursor.execute("""
                    SELECT c.id, c.raw_text, COUNT(t.id) as thesis_count
                    FROM categories c
                    LEFT JOIN theses t ON t.chosen_category_id = c.id
                    GROUP BY c.id, c.raw_text
                """)

                current_stats = cursor.fetchall()
                current_counts = [stat[2] for stat in current_stats]
                current_avg = sum(current_counts) / len(current_stats)

                overflow = [stat for stat in current_stats if stat[2] > current_avg * 1.2]
                underflow = [stat for stat in current_stats if stat[2] < current_avg * 0.8]

                if not overflow or not underflow:
                    print("Достигнуто равновесие - нет категорий для перераспределения")
                    break

                # Для каждой переполненной категории
                for overflow_cat in overflow:
                    overflow_cat_id = overflow_cat[0]
                    overflow_count = overflow_cat[2]

                    # Сколько нужно переместить из этой категории
                    target_count = int(current_avg)
                    to_move = min(overflow_count - target_count, 20)  # максимум 20 за итерацию

                    if to_move <= 0:
                        continue

                    print(
                        f"\nОбработка категории {overflow_cat_id} (тезисов: {overflow_count}, нужно переместить: {to_move})")

                    # Получение тезисов с наименьшей уверенностью из переполненной категории
                    cursor.execute("""
                        SELECT t.id, t.cleaned_text, t.probability_vector
                        FROM theses t
                        WHERE t.chosen_category_id = %s 
                          AND t.probability_vector IS NOT NULL
                        ORDER BY (t.probability_vector->>%s)::float ASC
                        LIMIT %s
                    """, (overflow_cat_id, str(overflow_cat_id), to_move * 2))

                    theses_to_relocate = cursor.fetchall()

                    # Для каждого тезиса ищется лучшая альтернативная категория
                    for thesis_id, thesis_text, prob_vector in theses_to_relocate:
                        if prob_vector is None:
                            continue

                        # Нахождение категории с недостатком
                        best_alternative = None
                        best_probability = min_confidence_threshold

                        for underflow_cat in underflow:
                            underflow_cat_id = underflow_cat[0]

                            # Получение вероятности для этой категории
                            try:
                                prob = float(prob_vector.get(str(underflow_cat_id), 0))
                            except:
                                prob = 0

                            # Если вероятность выше порога и категория недополнена
                            if prob > best_probability:
                                best_probability = prob
                                best_alternative = underflow_cat_id

                        # Если найдена подходящая альтернатива, перемещаем тезис
                        if best_alternative is not None:
                            cursor.execute("""
                                UPDATE theses 
                                SET chosen_category_id = %s
                                WHERE id = %s
                            """, (best_alternative, thesis_id))

                            moved_count += 1
                            print(
                                f" Тезис {thesis_id}: {overflow_cat_id} → {best_alternative} (уверенность: {best_probability:.3f})")

                            # Обновление статистики underflow категории
                            for i, uc in enumerate(underflow):
                                if uc[0] == best_alternative:
                                    underflow[i] = (uc[0], uc[1], uc[2] + 1)
                                    break

                            # Если категория перестала быть недополненной, она убирается из списка
                            underflow = [uc for uc in underflow if uc[2] < current_avg * 0.8]

                            if moved_count >= to_move:
                                break

                if moved_count > 0:
                    conn.commit()

                total_moved += moved_count
                print(f"\nИтерация {iteration + 1}: перемещено {moved_count} тезисов")

                # Проверка нового распределения
                cursor.execute("""
                    SELECT MIN(count) as min_count, MAX(count) as max_count
                    FROM (
                        SELECT COUNT(t.id) as count
                        FROM categories c
                        LEFT JOIN theses t ON t.chosen_category_id = c.id
                        GROUP BY c.id
                    ) as stats
                """)

                new_min, new_max = cursor.fetchone()
                new_ratio = new_min / new_max if new_max > 0 else 0

                print(f"Новое соотношение мин/макс: {new_ratio:.3f}")

                if new_ratio >= target_balance_ratio:
                    print(f"\n Достигнуто целевое соотношение {target_balance_ratio}")
                    break

                if moved_count == 0:
                    print("Нет тезисов для перемещения на этой итерации")
                    break

            rebalance_time = time.time() - start_time
            self.timing_stats['rebalance_time'] = rebalance_time

            print(f"\n{'=' * 60}")
            print("ПЕРЕРАСПРЕДЕЛЕНИЕ ЗАВЕРШЕНО")
            print(f"{'=' * 60}")
            print(f"Всего перемещено тезисов: {total_moved}")
            print(f" Время перераспределения: {rebalance_time:.2f} сек")

            # Выводим финальное распределение
            cursor.execute("""
                SELECT c.id, c.raw_text, COUNT(t.id) as thesis_count
                FROM categories c
                LEFT JOIN theses t ON t.chosen_category_id = c.id
                GROUP BY c.id, c.raw_text
                ORDER BY thesis_count DESC
            """)

            final_stats = cursor.fetchall()
            final_counts = [stat[2] for stat in final_stats]
            final_max = max(final_counts) if final_counts else 0

            print(f"\n{'=' * 60}")
            print("ФИНАЛЬНОЕ РАСПРЕДЕЛЕНИЕ ТЕЗИСОВ")
            print(f"{'=' * 60}")

            for cat_id, cat_text, count in final_stats:
                cat_name = cat_text[:40] if cat_text else f"Category_{cat_id}"
                bar = "█" * int(count / final_max * 50) if final_max > 0 else ""
                print(f"Категория {cat_id:3} ({cat_name:40}): {count:4} тезисов {bar}")

            # Вывод статистики после балансировки
            final_min = min(final_counts) if final_counts else 0
            final_max = max(final_counts) if final_counts else 0
            final_avg = sum(final_counts) / len(final_counts) if final_counts else 0

            print(f"\nМин: {final_min}, Макс: {final_max}, Среднее: {final_avg:.1f}")
            print(f"Соотношение мин/макс: {final_min / final_max:.3f}")
            print(f"Улучшение: {(min_count / max_count - final_min / final_max) * 100:.1f}%")

            cursor.close()

        except Exception as e:
            print(f"Ошибка при балансировке: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()


    # Получение статистики по классификации из БД
    def get_classification_stats(self) -> Dict:
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT t.id) as total_theses,
                    COUNT(DISTINCT CASE WHEN t.chosen_category_id IS NOT NULL THEN t.id END) as classified,
                    COUNT(DISTINCT c.id) as categories_with_theses,
                    AVG(CASE WHEN t.probability_vector IS NOT NULL 
                        THEN (t.probability_vector->>t.chosen_category_id::text)::float 
                        ELSE NULL END) as avg_confidence
                FROM theses t
                LEFT JOIN categories c ON t.chosen_category_id = c.id
                WHERE t.cleaned_text IS NOT NULL
            """)

            result = cursor.fetchone()
            stats = {
                'total_theses': result[0] or 0,
                'classified_theses': result[1] or 0,
                'categories_used': result[2] or 0,
                'avg_confidence': result[3] or 0,
                'classification_rate': (result[1] / result[0] * 100) if result[0] else 0
            }

            cursor.close()
            return stats

        except Exception as e:
            print(f"Ошибка при получении статистики: {e}")
            return {}
        finally:
            if conn:
                conn.close()


    # Вывод сводки по времени выполнения """
    def print_timing_summary(self):
        print("\n" + "=" * 60)
        print("СВОДКА ПО ВРЕМЕНИ ВЫПОЛНЕНИЯ")
        print("=" * 60)
        print(f" Загрузка модели:        {self.timing_stats['load_time']:.2f} сек")
        print(f" Предсказание:           {self.timing_stats['prediction_time']:.2f} сек")
        print(f" Обновление БД:          {self.timing_stats['db_update_time']:.2f} сек")
        print(f" Перераспределение:      {self.timing_stats['rebalance_time']:.2f} сек")
        print(f" Общее время:            {self.timing_stats['total_time']:.2f} сек")


# Тестирование
"""
def main():
    total_start_time = time.time()
    # Путь к модели
    MODEL_PATH = "thesis_classifier_catboost.pkl"

    # Проверка существования модели
    import os
    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: Модель не найдена по пути {MODEL_PATH}")
        print("Сначала обучите модель с помощью catboost_train.py")
        return

    # Загрузка классификатора
    print("\nЗагрузка модели...")
    classifier = ThesisClassifier(MODEL_PATH)

    # Показ статистики до классификации
    print("\nСтатистика до классификации:")
    stats_before = classifier.get_classification_stats()
    for key, value in stats_before.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Классификация неразмеченных тезисов
    print("\n" + "=" * 60)
    response = input("Классифицировать неразмеченные тезисы? (y/n): ")

    if response.lower() == 'y':
        classifier.update_thesis_predictions(threshold=0.0)

    # Перебалансировка распределения
    print("\n" + "=" * 60)
    response = input("Выполнить перебалансировку категорий? (y/n): ")

    if response.lower() == 'y':
        target_ratio = 0.6
        min_confidence = 0.0

        classifier.rebalance_theses_distribution(
            target_balance_ratio=target_ratio,
            min_confidence_threshold=min_confidence,
            max_iterations=10)

    # Показываем итоговую статистику
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)

    stats_after = classifier.get_classification_stats()
    for key, value in stats_after.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Вывод сводки по времени
    classifier.print_timing_summary()

    total_time = time.time() - total_start_time
    print(f"\n Общее время выполнения программы: {total_time:.2f} сек ({total_time / 60:.2f} мин)")


if __name__ == "__main__":
    main()
"""