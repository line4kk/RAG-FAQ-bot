from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pymorphy3
from typing import List, Tuple, Dict, Optional

from dotenv import find_dotenv, load_dotenv
import os

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
        self.__data_dict: Dict[str, str] = dictionary
        self.__data_keys: List[str] = []
        self.__embedding_model: SentenceTransformer = SentenceTransformer(embedding_model)
        self.__L1_search: Optional[faiss.IndexFlatIP] = None
        self.__morph: pymorphy3.MorphAnalyzer = pymorphy3.MorphAnalyzer()
        self.__ACCURACY: float = min_score  # Порог релевантности для embedding

        self.set_dict(dictionary)
        self.__update_embedding_faq()

    def __lemmatize_words(self, words: List[str]) -> List[str]:
        """
        Лемматизация списка слов.

        Args:
            words: Список слов.

        Returns:
            Список нормализованных слов.
        """
        return [self.__morph.parse(word)[0].normal_form for word in words]

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

        question_words = set(self.__lemmatize_words(question.lower().split()))
        scores: List[Tuple[int, str]] = []

        for q in self.__data_dict.keys():
            faq_words = set(self.__lemmatize_words(q.lower().split()))
            match_count = len(question_words & faq_words)
            if match_count > 0:
                scores.append((match_count, q))
                print(f"Совпадений: {match_count}, Вопрос: {q}")

        # Сортировка по количеству совпадений
        scores.sort(reverse=True, key=lambda x: x[0])

        # Возврат top_k вопросов
        return [q for _, q in scores[:top_k]]

    def find_similar_text_in_list(self, text: str, data_list: List[str], count: int = 2) -> List[Tuple[str, float]]:
        """
        Векторный поиск по небольшому списку вопросов.

        Args:
            text: Входной вопрос.
            data_list: Список вопросов для поиска.
            count: Сколько топ результатов вернуть.

        Returns:
            Список кортежей (вопрос, score).
        """
        if not text:
            raise ValueError("Пустой вопрос")
        if not data_list:
            raise ValueError("Список для поиска пуст")

        # Создание embedding для списка
        embedding = self.__embedding_model.encode(data_list, convert_to_numpy=True)
        embedding = np.ascontiguousarray(embedding.astype('float32'))
        faiss.normalize_L2(embedding)

        dimension = embedding.shape[1]
        l1_search = faiss.IndexFlatIP(dimension)
        l1_search.add(embedding)

        # Векторизация входного текста
        vector_text = self.__embedding_model.encode([text], convert_to_numpy=True)
        vector_text = np.ascontiguousarray(vector_text.astype('float32'))
        faiss.normalize_L2(vector_text)

        distances, indices = l1_search.search(vector_text, count)
        results = [(data_list[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

        for i, idx in enumerate(indices[0]):
            print(f"{i+1}. {data_list[idx]} — score: {distances[0][i]:.3f}")

        return results

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

        vector_text = self.__embedding_model.encode([text], convert_to_numpy=True)
        vector_text = np.ascontiguousarray(vector_text.astype('float32'))
        faiss.normalize_L2(vector_text)

        distances, indices = self.__L1_search.search(vector_text, count)
        results = [(self.__data_keys[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

        for i, idx in enumerate(indices[0]):
            print(f"{i+1}. {self.__data_keys[idx]} — score: {distances[0][i]:.3f}")

        return results

    def __update_embedding_faq(self) -> None:
        """
        Обновление векторного индекса FAISS для всей базы FAQ.
        """
        embedding = self.__embedding_model.encode(self.__data_keys, convert_to_numpy=True)
        embedding = np.ascontiguousarray(embedding.astype('float32'))
        faiss.normalize_L2(embedding)

        dimension = embedding.shape[1]
        self.__L1_search = faiss.IndexFlatIP(dimension)
        self.__L1_search.add(embedding)

    def set_dict(self, dictionary: Dict[str, str]) -> None:
        """
        Обновление словаря FAQ.

        Args:
            dictionary: Новый словарь {вопрос: ответ}.
        """
        if not dictionary:
            raise ValueError("Пустой словарь")
        self.__data_dict = dictionary
        self.__data_keys = list(dictionary.keys())

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

        return retrieval_result if retrieval_result else []



class RAG:
    def __init__(self, basic_prompt : str, faq : dict):
        self.__ai_client = genai.Client(api_key=self.__get_api_key())
        self.__basic_prompt = basic_prompt  # Базовый промпт для ИИ, чтобы он понимал кто он. Должен заканчиваться на просьбу ответить на следующий вопрос и последующим двоеточием
        self.__faq : dict = {}  # Кеш для FAQ
        self.__questions: list = []  # Список вопросов
        self.__L1_search = None  # L1 поисковик

        self.set_faq(faq)

    # Получить ключ API из env
    def __get_api_key(self) -> str:
        load_dotenv(find_dotenv())
        return os.getenv("GEMINI_API_KEY")

    # Поиск похожих вопросов


    # Добавить новую пару вопрос-ответ в кеш
    def add_aq(self, question : str, answer : str):
        if not isinstance(question, str) or not isinstance(answer, str):
            raise TypeError("Ключ и значение должны быть строками.")
        if not question or not answer:
            raise ValueError("Пустой вопрос или ответ")
        if question in self.__faq:
            raise ValueError("Такой вопрос уже есть.")
        self.__faq[question] = answer
        self.__questions.append(question)
        self.__update_vector_faq()

    # Кэшировать FAQ
    def set_faq(self, faq : dict) -> None:
        if not faq:
            raise ValueError("Пустой словарь")
        self.__faq = faq
        self.__questions = list(faq.keys())
        self.__update_vector_faq()
    
    def get_response(self, text : str, model : str="gemini-2.5-flash", accuracy=2) -> str:
        if not self.__faq:
            raise RuntimeError("Не установлена база FAQ")
        similary_questions = self.__find_similar_questions(text, count=accuracy)
        # prompt = self.__basic_prompt + " " + text + f"Используй ИСКЛЮЧИТЕЛЬНО следующие данные о вопросах и ответах на них:"
        # for question in similary_questions:
        #     prompt += f"\nВопрос: {question}; Ответ: {self.__faq[question]}"
        # response = self.__ai_client.models.generate_content(
        #     model=model,
        #     contents=prompt,
        #     config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0)))

        # return response.text
    

