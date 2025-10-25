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
    def __get_api_key(self) -> str:
        load_dotenv(find_dotenv())
        return os.getenv("GEMINI_API_KEY")


    def get_response(self, text : str, model : str="gemini-2.5-flash", kw_res_max_len=2, emb_res_max_len=3) -> str:
        if not self._data_dict:
            raise RuntimeError("Не установлена база FAQ")
        similar_questions = super().retrieval(text, kw_res_max_len, emb_res_max_len)
        prompt = self.__basic_prompt + '\n'
        prompt += "Вопрос пользователя: " + text + "\n"
        prompt += "Из базы данных FAQ были найдены похожие на вопрос пользователя результаты. Сейчас я их тебе перечислю. Проанализируй их и выбери только тот (те), что действительно подходит(ят) под вопрос пользователя и отвечают на него. Затем, сформулируй ответ используя только его (их). Иначе - попроси пользователя переформулировать вопрос.\n"
        if similar_questions:
            for question in similar_questions:
                prompt += '\n'
                prompt += "Вопрос: " + question + "\n"
                prompt += "Ответ: " + self._data_dict[question] + "\n"
        else:
            prompt += "Результаты в FAQ не найдены."

        response = self.__ai_client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0)))

        return response.text
    

