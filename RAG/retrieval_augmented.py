from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pymorphy3
from typing import List, Tuple, Dict, Optional
import re

class RetrievalAugmented:
    """
    Класс для Retrieval-Augmented Generation (RAG) с FAQ.

    Комбинирует два уровня поиска:
    1. Keyword-анализ с лемматизацией через pymorphy3
    2. Векторный семантический поиск через SentenceTransformer + FAISS

    Подходит для построения FAQ-ботов или поддержки LLM.
    """

    def __init__(self, dictionary: Dict[str, str], embedding_model: str = 'paraphrase-multilingual-MiniLM-L12-v2', min_score : float = 0.46) -> None:
        """
        Инициализация класса.

        Args:
            dictionary: Словарь {вопрос: ответ}.
            embedding_model: Модель для генерации эмбеддингов.
        """
        self._data_dict: Dict[str, str] = {}
        self._data_keys: List[str] = []
        self._search_key_to_original: Dict[str, str] = {}
        self.__embedding_model: SentenceTransformer = SentenceTransformer(embedding_model)
        self.__L1_search: Optional[faiss.IndexFlatIP] = None
        self.__morph: pymorphy3.MorphAnalyzer = pymorphy3.MorphAnalyzer()
        self.__ACCURACY: float = min_score  # Порог релевантности для embedding

        self.set_dict(dictionary)

    def __lemmatize_words(self, words: List[str]) -> List[str]:
        """
        Лемматизация списка слов.

        Args:
            words: Список слов.

        Returns:
            Список нормализованных слов.
        """
        return [self.__morph.parse(word)[0].normal_form for word in words]

    @staticmethod
    def __normalize_text(text: str) -> str:
        """Нормализация текста: приведение к нижнему регистру, удаление пунктуации."""
        return re.sub(r"[^\w\s]", "", text.lower()).strip()

    def keyword_analysis(self, question: str, top_k: int = 5) -> List[str]:
        """
        Поиск ключевых вопросов из FAQ на основе пересечения лемматизированных слов.

        Args:
            question: Входной вопрос.
            top_k: Сколько топ вопросов вернуть.

        Returns:
            Список ключевых вопросов из FAQ.
        """
        if not question:
            raise ValueError("Пустой вопрос")

        question = self.__normalize_text(question)
        question_words = set(self.__lemmatize_words(question.split()))
        scores: List[Tuple[int, str]] = []

        for normalized_q in self._data_keys:
            faq_words = set(self.__lemmatize_words(normalized_q.split()))
            match_count = len(question_words & faq_words)
            if match_count > 0:
                scores.append((match_count, self._search_key_to_original[normalized_q]))

        # Сортировка по количеству совпадений
        scores.sort(reverse=True, key=lambda x: x[0])

        # Возврат top_k вопросов
        return [q for _, q in scores[:top_k]]

    def find_similar_text_in_data(self, text: str, count: int = 2) -> List[Tuple[str, float]]:
        """
        Векторный поиск по всей базе FAQ.

        Args:
            text: Входной вопрос.
            count: Сколько топ результатов вернуть.

        Returns:
            Список кортежей (вопрос, score).
        """
        if not text:
            raise ValueError("Пустой вопрос")

        text = self.__normalize_text(text)

        vector_text = self.__embedding_model.encode([text], convert_to_numpy=True)
        vector_text = np.ascontiguousarray(vector_text.astype('float32'))
        faiss.normalize_L2(vector_text)

        distances, indices = self.__L1_search.search(vector_text, count)
        results = [(self._search_key_to_original[self._data_keys[idx]], distances[0][i]) for i, idx in enumerate(indices[0])]


        return results

    def __update_embedding_faq(self) -> None:
        """Обновление векторного индекса FAISS для всей базы FAQ."""
        embedding = self.__embedding_model.encode(self._data_keys, convert_to_numpy=True)
        embedding = np.ascontiguousarray(embedding.astype('float32'))
        faiss.normalize_L2(embedding)

        dimension = embedding.shape[1]
        self.__L1_search = faiss.IndexFlatIP(dimension)
        self.__L1_search.add(embedding)

    def get_dict(self):
        """
        Получение словаря FAQ
        """
        return self._data_dict

    def set_dict(self, dictionary: Dict[str, str]) -> None:
        """
        Обновление словаря FAQ с нормализацией текста.

        Нормализует только ключи для поиска. Оригинальные вопросы и ответы
        сохраняются для передачи в prompt модели.

        Args:
            dictionary: Новый словарь {вопрос: ответ}.
        """
        if not dictionary:
            raise ValueError("Пустой словарь")

        original_dict = dict(dictionary)
        search_key_to_original = {}

        for q in dictionary:
            # Нормализуем только ключи для поиска, ответы сохраняем как есть.
            clean_q = self.__normalize_text(q)
            search_key_to_original[clean_q] = q

        self._data_dict = original_dict
        self._data_keys = list(search_key_to_original.keys())
        self._search_key_to_original = search_key_to_original

        self.__update_embedding_faq()

    def add_aq(self, question: str, answer: str):
        """
        Добавить к словарю FAQ новую пару вопрос-ответ

        Args:
            question: Вопрос
            answer: Ответ
        """
        if not isinstance(question, str) or not isinstance(answer, str):
            raise TypeError("Ключ и значение должны быть строками.")
        if not question or not answer:
            raise ValueError("Пустой вопрос или ответ.")

        normalized_question = self.__normalize_text(question)

        if normalized_question in self._search_key_to_original:
            raise ValueError("Такой вопрос уже есть.")

        self._data_dict[question] = answer
        self._data_keys.append(normalized_question)
        self._search_key_to_original[normalized_question] = question
        self.__update_embedding_faq()

    def retrieval(self, question: str, kw_res_len: int = 5, emb_res_len: int = 2) -> List[str]:
        """
        Основной метод retrieval для RAG.

        Комбинирует keyword-анализ и embedding поиск.
        Сначала добавляет результаты keyword, затем embedding по всей базе.

        Args:
            question: Входной вопрос.
            kw_res_len: Сколько топ результатов взять с keyword анализа.
            emb_res_len: Сколько топ результатов взять с embedding поиска.

        Returns:
            Список релевантных вопросов из FAQ.
        """
        if not question:
            raise ValueError("Пустой вопрос")
        if kw_res_len <= 0 or emb_res_len <= 0:
            raise ValueError("Кол-во элементов после поиска должно быть больше 0")

        retrieval_result: List[str] = []

        # 1. Keyword-анализ
        keyword_analyzed = self.keyword_analysis(question, kw_res_len)
        if keyword_analyzed:
            retrieval_result.extend(keyword_analyzed)

        # 2. Векторный поиск по всей базе
        similar_texts = self.find_similar_text_in_data(question, emb_res_len)
        similar_texts = [
            text for text, accuracy in similar_texts
            if (accuracy > self.__ACCURACY) and (text not in retrieval_result)
        ]
        retrieval_result.extend(similar_texts)
        return retrieval_result
