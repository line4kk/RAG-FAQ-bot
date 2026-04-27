from aiogram import types, Router

import asyncio

from RAG.ra_generation import RAGeneration
from app_state import app_state

questions_handler_rout = Router()

SYSTEM_PROMPT = (
    "Ты — консультант фитнес-клуба \"DiamondFitness\". Твоя задача — точно и "
    "вежливо отвечать на вопросы пользователей, используя исключительно "
    "предоставленную тебе информацию из базы знаний о клубе (FAQ и другие "
    "документы). Твои ответы должны быть доброжелательными, профессиональными. "
    "Твой ответ - ответ на один конкретный вопрос. Не здоровайся с пользователем, "
    "если он не поздоровался с тобой в своем сообщении."
)

rag = RAGeneration(SYSTEM_PROMPT, app_state.faq)

@questions_handler_rout.message()
async def questions_handler(message: types.Message):
    await message.answer(await asyncio.to_thread(rag.get_response, message.text))
