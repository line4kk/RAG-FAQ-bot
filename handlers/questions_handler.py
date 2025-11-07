from aiogram import types, Router

import asyncio

from RAG.ra_generation import RAGeneration
from app_state import app_state

questions_handler_rout = Router()

rag = RAGeneration("Ты — AI-ассистент виртуального консультанта фитнес-клуба премиум-класса \"Diamond\". Твоя задача — точно и вежливо отвечать на вопросы пользователей, используя исключительно предоставленную тебе информацию из базы знаний о клубе (FAQ и другие документы). Ты являешься лицом компании, поэтому твои ответы должны быть доброжелательными, профессиональными и отражать ценности бренда.", app_state.faq)

@questions_handler_rout.message()
async def questions_handler(message: types.Message):
    await message.answer(await asyncio.to_thread(rag.get_response, message.text))
    

