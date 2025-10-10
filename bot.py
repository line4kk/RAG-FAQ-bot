from create_bot import bot, dp
from aiogram import types  # 
import asyncio, logging, sys


@dp.message()
async def hello_world(message: types.Message):
    await message.answer(f"Hello, world!\n\nYou just said: {message.text}")



async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
    