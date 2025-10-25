from google import genai
from google.genai import types
from typing import List, Tuple, Dict, Optional

from dotenv import find_dotenv, load_dotenv
import os

from RAG.retrieval_augmented import RetrievalAugmented


class RAGeneration(RetrievalAugmented):
    """
       Класс RAGeneration реализует Retrieval-Augmented Generation поверх RetrievalAugmented.

       Он комбинирует:
         - поиск похожих вопросов из FAQ (через наследуемый класс RetrievalAugmented);
         - генерацию итогового ответа с помощью модели Gemini от Google.

       Основная идея: сначала найти релевантные вопросы и ответы из FAQ,
       затем передать их модели LLM, чтобы та выбрала наиболее подходящий ответ
       или сформулировала новый на основе найденных данных.

       Пример использования:
           rag = RAGeneration(basic_prompt="Ты — помощник центра обучения.", dictionary=faq_data)
           answer = rag.get_response("Какие у вас есть скидки?")
    """

    def __init__(self, basic_prompt: str, dictionary: Dict[str, str], embedding_model: str = 'paraphrase-multilingual-MiniLM-L12-v2', min_score : float = 0.46) -> None:
        """
        Инициализация экземпляра класса RAGeneration.

        Args:
            basic_prompt: Базовый системный промпт (контекст), определяющий поведение модели.
            dictionary: База FAQ в формате {вопрос: ответ}.
            embedding_model: Название модели эмбеддингов для поиска по смыслу.
            min_score: Минимальный порог схожести для эмбеддингового поиска.
        """

        # Инициализация родительского класса RetrievalAugmented
        super().__init__(dictionary, embedding_model, min_score)

        # Инициализация клиента для Gemini API с использованием ключа из .env
        self.__ai_client = genai.Client(api_key=self.__get_api_key())

        # Сохранение базового промпта (контекста)
        self.__basic_prompt = basic_prompt

    # Получить ключ API из env
    @staticmethod
    def __get_api_key() -> str:
        """
        Загрузка ключа API из файла .env.

        Возвращает:
            str: Значение переменной окружения GEMINI_API_KEY.

        Raises:
            RuntimeError: Если ключ не найден в .env файле.
        """
        load_dotenv(find_dotenv())
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Не найден GEMINI_API_KEY в .env")
        return key

    def get_response(self, text : str, model : str="gemini-2.5-flash", kw_res_max_len=2, emb_res_max_len=3) -> str:
        """
        Основной метод RAG — получение ответа от модели на основе FAQ и запроса пользователя.

        Алгоритм:
            1. Находит похожие вопросы из FAQ (через RetrievalAugmented.retrieval()).
            2. Формирует промпт для LLM, включающий контекст, вопрос пользователя и найденные данные.
            3. Передаёт промпт модели Gemini, получает сгенерированный ответ.
            4. Возвращает текст ответа.

        Args:
            text: Вопрос пользователя.
            model: Название модели Gemini (по умолчанию gemini-2.5-flash).
            kw_res_max_len: Максимальное количество результатов от keyword-анализа.
            emb_res_max_len: Максимальное количество результатов от embedding-поиска.

        Returns:
            str: Сгенерированный ответ от модели.

        Raises:
            RuntimeError: Если база FAQ не установлена.
        """

        # Проверяем, что база FAQ загружена
        if not self._data_dict:
            raise RuntimeError("Не установлена база FAQ")

        # Получаем похожие вопросы через RetrievalAugmented
        similar_questions = super().retrieval(text, kw_res_max_len, emb_res_max_len)

        # Формируем части промпта
        parts = [
            self.__basic_prompt,
            f"Вопрос пользователя: {text}",
            (
                "Из базы FAQ найдены похожие вопросы. "
                "Проанализируй их и выбери те, что действительно отвечают на вопрос пользователя. "
                "Если подходящих нет — попроси пользователя переформулировать запрос."
            )
        ]

        # Добавляем найденные вопросы и ответы в промпт
        if similar_questions:
            for q in similar_questions:
                a = self._data_dict[q]
                parts.append(f"\nВопрос: {q}\nОтвет: {a}")
        else:
            # Если ничего не найдено — информируем модель
            parts.append("Результаты не найдены в базе часто-задаваемых вопросов.")

        # Собираем итоговый промпт
        prompt = "\n".join(parts)

        # Отправляем промпт в модель Gemini и получаем ответ
        response = self.__ai_client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )

        # Возвращаем только текст ответа
        return response.text
    

