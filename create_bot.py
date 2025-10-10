import os
from aiogram import Bot, Dispatcher
from dotenv import find_dotenv, load_dotenv

from handlers.start_handler import start_rout

dp = Dispatcher()
dp.include_router(start_rout)

load_dotenv(find_dotenv())
bot = Bot(token=os.getenv("TOKEN"))
