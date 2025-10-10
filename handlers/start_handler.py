from aiogram import types, Router
from aiogram.filters import CommandStart

start_rout = Router()

@start_rout.message(CommandStart())
async def start(message: types.Message):
    await message.answer(
    "💎 Добро пожаловать в DiamondFitness!\n\n"
    "Я помогу вам с расписанием, абонементами и услугами клуба.\n"
    "Просто напишите свой вопрос 👇"
    )