import psycopg2
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional

# Конфигурация подключения к БД
DB_CONFIG = {
    'host': '',
    'port': '', # здесь просто цифры без ''
    'database': '',
    'user': '',
    'password': ''
}

initial_distribution: Optional[list[tuple[int, str, int, float]]]


# Класс для визуализации распределения тезисов по категориям
class ThesisVisualizer:
    def __init__(self, db_config):
        self.conn = None
        self.db_config = db_config
        self.setup_connection()


    # Устанавливаем соединение с базой данных
    def setup_connection(self):
        try:
            self.conn = psycopg2.connect(**self.db_config)
            print("Соединение с БД установлено")
        except Exception as e:
            print(f"Ошибка подключения к БД: {e}")
            raise


    # Получаем распределение тезисов по категориям
    def get_category_distribution(self) -> List[Tuple[int, str, int, float]]:
        try:
            cursor = self.conn.cursor()

            # Получение распределения с дополнительной статистикой
            cursor.execute("""
                SELECT 
                    c.id,
                    c.raw_text,
                    COUNT(t.id) as thesis_count,
                    COALESCE(AVG((t.probability_vector->>(c.id::text))::float), 0) as avg_probability
                FROM categories c
                LEFT JOIN theses t ON t.chosen_category_id = c.id
                WHERE c.cleaned_text IS NOT NULL AND c.cleaned_text != ''
                GROUP BY c.id, c.raw_text
                ORDER BY c.id ASC
            """)

            results = cursor.fetchall()
            cursor.close()

            return results

        except Exception as e:
            print(f"Ошибка при получении распределения: {e}")
            return []


    # Получает изначальное распределение из поля initial_category_distribution
    def get_initial_distribution(self) -> Optional[List[Tuple[int, str, int, float]]]:
        try:
            cursor = self.conn.cursor()

            # Проверяем существование поля и получаем данные
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='theses' AND column_name='initial_category_distribution'
            """)

            if not cursor.fetchone():
                print("Поле initial_category_distribution не найдено")
                return None

            # Получаем изначальное распределение
            cursor.execute("""
                SELECT 
                    c.id,
                    c.raw_text,
                    COUNT(t.id) as thesis_count,
                    COALESCE(AVG((t.probability_vector->>(c.id::text))::float), 0) as avg_probability
                FROM categories c
                LEFT JOIN theses t ON t.initial_category_distribution = c.id
                WHERE c.cleaned_text IS NOT NULL AND c.cleaned_text != ''
                GROUP BY c.id, c.raw_text
                ORDER BY c.id ASC
            """)

            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            print(f"Ошибка при получении изначального распределения: {e}")
            return None


    # Построение графиков распределения тезисов по категориям до и после перераспределения
    def plot_comparison_charts(self,
                               top_n: int = None,
                               save_path: str = 'category_distribution_comparison.png',
                               show_values: bool = True,
                               figsize: Tuple[int, int] = (18, 8)):
        # Получаем оба распределения
        initial_distribution = self.get_initial_distribution()
        current_distribution = self.get_category_distribution()

        if not current_distribution:
            print("Нет данных для визуализации")
            return

        # Ограничение количества категорий
        if top_n and top_n < len(current_distribution):
            initial_distribution = initial_distribution[:top_n]
            current_distribution = current_distribution[:top_n]

        # Подготовка данных
        categories = []
        initial_counts = []
        current_counts = []

        for (cat_id, cat_text, init_count, _), (_, _, curr_count, _) in zip(initial_distribution, current_distribution):
            short_text = cat_text[:15] + '...' if len(cat_text) > 15 else cat_text
            categories.append(f"{short_text}\n(ID: {cat_id})")
            initial_counts.append(init_count)
            current_counts.append(curr_count)

        # Создание графика с двумя подграфиками
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # График ДО перераспределения
        bars1 = ax1.bar(range(len(categories)), initial_counts, color='coral', alpha=0.8)
        ax1.set_xlabel('Категории', fontsize=12)
        ax1.set_ylabel('Количество тезисов', fontsize=12)
        ax1.set_title("Результат классификации", fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)

        if show_values:
            for i, (bar, count) in enumerate(zip(bars1, initial_counts)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                         f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax1.grid(axis='y', alpha=0.3)
        ax1.set_axisbelow(True)

        # График ПОСЛЕ перераспределения
        bars2 = ax2.bar(range(len(categories)), current_counts, color='steelblue', alpha=0.8)
        ax2.set_xlabel('Категории', fontsize=12)
        ax2.set_ylabel('Количество тезисов', fontsize=12)
        ax2.set_title("Результат перераспределения", fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)

        if show_values:
            for i, (bar, count) in enumerate(zip(bars2, current_counts)):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                         f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax2.grid(axis='y', alpha=0.3)
        ax2.set_axisbelow(True)

        # Добавление общего заголовка с информацией об изменениях
        total_initial = sum(initial_counts)
        total_current = sum(current_counts)
        changed_theses = sum(abs(c - i) for i, c in zip(initial_counts, current_counts)) // 2

        plt.suptitle(
            f'Сравнение распределений тезисов\nВсего тезисов: {total_current} | Перераспределено: {changed_theses}',
            fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        # Сохранение графика
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Сравнительный график сохранен как {save_path}")
        plt.show()

        return fig


    # Построение столбчатого графика распределения тезисов по категориям
    def plot_bar_chart(self,
                       top_n: int = None,
                       save_path: str = 'category_distribution.png',
                       show_values: bool = True,
                       figsize: Tuple[int, int] = (12, 8)):
        # Получение данных
        distribution = self.get_category_distribution()

        if not distribution:
            print("Нет данных для визуализации")
            return

        # Ограничение количества категорий
        if top_n and top_n < len(distribution):
            distribution = distribution[:top_n]

        # Подготовка данных
        categories = []
        counts = []

        for cat_id, cat_text, count, _ in distribution:
            short_text = cat_text[:20] + '...' if len(cat_text) > 20 else cat_text
            #categories.append(f"{short_text}\n(ID: {cat_id})")
            categories.append(f"{cat_id}")
            counts.append(count)

        # Создание графика
        fig, ax = plt.subplots(figsize=figsize)

        # График количества тезисов
        bars = ax.bar(range(len(categories)), counts, color='steelblue', alpha=0.8)
        ax.set_xlabel('Категории', fontsize=20)
        ax.set_ylabel('Количество тезисов', fontsize=20)
        #ax.set_title("Распределение тезисов по категориям", fontsize=20, fontweight='bold')
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=20)
        ax.tick_params(axis='y', labelsize=20)

        if show_values:
            for i, (bar, count) in enumerate(zip(bars, counts)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                        f'{count}', ha='center', va='bottom', fontsize=15, fontweight='bold')

        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()

        # Сохранение графика
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранен как {save_path}")
        plt.show()

        return fig


    # Вывод статистической информации о распределении
    def print_statistics(self):
        distribution = self.get_category_distribution()

        if not distribution:
            print("Нет данных для анализа")
            return

        counts = [count for _, _, count, _ in distribution]
        probs = [prob for _, _, _, prob in distribution if prob > 0]

        print("\n" + "=" * 70)
        print("СТАТИСТИКА РАСПРЕДЕЛЕНИЯ ТЕЗИСОВ ПО КАТЕГОРИЯМ")
        print("=" * 70)
        print(f"Всего категорий: {len(distribution)}")
        print(f"Категорий с тезисами: {len([c for c in counts if c > 0])}")
        print(f"Всего тезисов: {sum(counts)}")
        print(f"\nСтатистика по количеству тезисов в категории:")
        print(f"  Среднее: {np.mean(counts):.2f}")
        print(f"  Медиана: {np.median(counts):.2f}")
        print(f"  Стандартное отклонение: {np.std(counts):.2f}")
        print(f"  Минимум: {min(counts)}")
        print(f"  Максимум: {max(counts)}")

        if probs:
            print(f"\nСтатистика по средней вероятности:")
            print(f"  Средняя вероятность: {np.mean(probs):.4f}")
            print(f"  Медианная вероятность: {np.median(probs):.4f}")
            print(f"  Стандартное отклонение: {np.std(probs):.4f}")
            print(f"  Минимальная вероятность: {min(probs):.4f}")
            print(f"  Максимальная вероятность: {max(probs):.4f}")

        print("\n10 категорий с наибольшим количеством тезисов:")
        print("-" * 70)
        for i, (cat_id, cat_text, count, avg_prob) in enumerate(distribution[:10], 1):
            print(f"{i:2d}. Категория {cat_id:3d}: {count:4d} тезисов | "
                  f"Ср.вероятность: {avg_prob:.4f} | {cat_text[:50]}")


    # Закрываем соединение с БД
    def close(self):
        if self.conn:
            self.conn.close()
            print("Соединение с БД закрыто")


# Тестирование
"""
def main():
    visualizer = None

    try:
        # Создаем экземпляр визуализатора
        visualizer = ThesisVisualizer()

        # Выводим статистику
        visualizer.print_statistics()

        # Проверяем наличие данных о начальном распределении
        global initial_distribution
        initial_distribution = visualizer.get_initial_distribution()

        if initial_distribution:
            print("\nОбнаружены данные о начальном распределении. Строим сравнительный график.")
            visualizer.plot_comparison_charts(
                top_n=None,
                save_path='category_distribution_comparison.png',
                show_values=True,
                figsize=(18, 8)
            )
        else:
            print("\nДанные о начальном распределении не найдены. Строим только текущий график.")
            visualizer.plot_bar_chart(
                top_n=None,
                save_path='category_distribution_current.png',
                show_values=True,
                figsize=(12, 8)
            )

    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if visualizer:
            visualizer.close()


if __name__ == "__main__":
    main()
"""