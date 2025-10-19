from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
import faiss

from dotenv import find_dotenv, load_dotenv
import os


class RAG:
    def __init__(self, basic_prompt : str, faq : dict):
        self.__ai_client = genai.Client(api_key=self.__get_api_key())
        self.__basic_prompt = basic_prompt  # Базовый промпт для ИИ, чтобы он понимал кто он. Должен заканчиваться на просьбу ответить на следующий вопрос и последующим двоеточием
        self.__embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.__faq : dict = {}  # Кеш для FAQ
        self.__questions : list = []  # Список вопросов
        self.__L1_search = None  # L1 поисковик

        self.set_faq(faq)

    # Получить ключ API из env
    def __get_api_key(self) -> str:
        load_dotenv(find_dotenv())
        return os.getenv("GEMINI_API_KEY")

    # Поиск похожих вопросов
    def __find_similar_questions(self, question : str, count : int = 2):
        if not question:
            raise ValueError("Пустой вопрос")
        vector_question = self.__embedding_model.encode([question], convert_to_numpy=True)
        distances, indices = self.__L1_search.search(vector_question, count)
        results = [self.__questions[i] for i in indices[0]]
        for i, idx in enumerate(indices[0]):
            print(f"{i+1}. {self.__questions[idx]} — score: {distances[0][i]:.3f}")
        return results

    def __update_vector_faq(self):
        vector_questions = self.__embedding_model.encode(self.__questions, convert_to_numpy=True)
        dimension = vector_questions.shape[1]
        self.__L1_search = faiss.IndexFlatL2(dimension)
        self.__L1_search.add(vector_questions)

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
    
    def get_responce(self, text : str, model : str="gemini-2.5-flash", accuracy=2) -> str:
        if not self.__faq:
            raise RuntimeError("Не установлена база FAQ")
        similary_questions = self.__find_similar_questions(text)
        prompt = self.__basic_prompt + " " + text + f"Используй ИСКЛЮЧИТЕЛЬНО следующие данные о вопросах и ответах на них:"
        for question in similary_questions:
            prompt += f"\nВопрос: {question}; Ответ: {self.__faq[question]}"
        response = self.__ai_client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0)))

        return response.text
    

