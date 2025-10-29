from aiogram import types, Router

from RAG.ra_generation import RAGeneration

questions_handler_rout = Router()

@questions_handler_rout.message()
async def questions_handler(message: types.Message):
    pass
    

