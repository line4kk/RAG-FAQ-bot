from google import genai
from google.genai import types
from typing import List, Tuple, Dict, Optional

from dotenv import find_dotenv, load_dotenv
import os

from RAG.retrieval_augmented import RetrievalAugmented


class RAGeneration(RetrievalAugmented):
    def __init__(self, basic_prompt: str, dictionary: Dict[str, str], embedding_model: str = 'paraphrase-multilingual-MiniLM-L12-v2', min_score : float = 0.46) -> None:
        super().__init__(dictionary, embedding_model, min_score)
        self.__ai_client = genai.Client(api_key=self.__get_api_key())
        self.__basic_prompt = basic_prompt  # Базовый промпт для ИИ, чтобы он понимал кто он (контекст)

    # Получить ключ API из env
    @staticmethod
    def __get_api_key(self) -> str:
        load_dotenv(find_dotenv())
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Не найден GEMINI_API_KEY в .env")
        return key

    def get_response(self, text : str, model : str="gemini-2.5-flash", kw_res_max_len=2, emb_res_max_len=3) -> str:
        if not self._data_dict:
            raise RuntimeError("Не установлена база FAQ")
        similar_questions = super().retrieval(text, kw_res_max_len, emb_res_max_len)
        parts = [
            self.__basic_prompt,
            f"Вопрос пользователя: {text}",
            "Из базы FAQ найдены похожие вопросы. "
            "Проанализируй их и выбери те, что действительно отвечают на вопрос пользователя. "
            "Если подходящих нет — попроси пользователя переформулировать запрос."
        ]
        if similar_questions:
            for q in similar_questions:
                a = self._data_dict[q]
                parts.append(f"\nВопрос: {q}\nОтвет: {a}")
        else:
            parts.append("Результаты не найдены в базе часто-задаваемых вопросов.")

        prompt = "\n".join(parts)

        response = self.__ai_client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0)))

        return response.text
    

