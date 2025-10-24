from google import genai
from google.genai import types
from typing import List, Tuple, Dict, Optional

from dotenv import find_dotenv, load_dotenv
import os

from RAG.retrieval_augmented import RetrievalAugmented


class RAGeneration:
    def __init__(self, basic_prompt : str, faq : dict, retrieval : RetrievalAugmented) -> None:
        if retrieval.get_dict() != faq:
            raise ValueError("Словарь, переданный в конструкторе, отличается от словаря в RetrievalAugmented.")

        self.__ai_client = genai.Client(api_key=self.__get_api_key())
        self.__basic_prompt = basic_prompt  # Базовый промпт для ИИ, чтобы он понимал кто он. Должен заканчиваться на просьбу ответить на следующий вопрос и последующим двоеточием
        self.__faq : dict = {}  # Кеш для FAQ
        self.__questions: list = []  # Список вопросов
        self.__retrieval_augmented = retrieval  # Объект класса RetrievalAugmented

        self.set_faq(faq)

    # Получить ключ API из env
    def __get_api_key(self) -> str:
        load_dotenv(find_dotenv())
        return os.getenv("GEMINI_API_KEY")

    # Добавить новую пару вопрос-ответ в кеш
    def add_aq(self, question : str, answer : str):
        if not isinstance(question, str) or not isinstance(answer, str):
            raise TypeError("Ключ и значение должны быть строками.")
        if not question or not answer:
            raise ValueError("Пустой вопрос или ответ.")
        if question in self.__faq:
            raise ValueError("Такой вопрос уже есть.")
        self.__faq[question] = answer
        self.__questions.append(question)

    # Кэшировать FAQ
    def set_faq(self, faq : dict) -> None:
        if not faq:
            raise ValueError("Пустой словарь")
        self.__faq = faq
        self.__questions = list(faq.keys())

    def get_response(self, text : str, model : str="gemini-2.5-flash", accuracy=2) -> str:
        if not self.__faq:
            raise RuntimeError("Не установлена база FAQ")
        prompt = self.__basic_prompt + " " + text + f"Используй ИСКЛЮЧИТЕЛЬНО следующие данные о вопросах и ответах на них:"

        response = self.__ai_client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0)))

        return response.text
    

