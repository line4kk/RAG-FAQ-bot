from aiogram import types, Router, F, Bot
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.filters import Command

from filters.admin_filter import IsAdmin

from app_state import app_state

admin_panel_rout = Router()
admin_panel_rout.message.filter(IsAdmin())


class AdminStates(StatesGroup):
    admin_panel_state = State()
    upload_faq_state = State()


@admin_panel_rout.message(Command("admin"))
async def start_admin_panel(message: types.Message, state: FSMContext):
    await message.answer(
        "Вы перешли в панель администратора. Доступные команды:\n\n"
        "/upload - загрузить новую базу данных FAQ\n"
        "/exit - выйти из панели администратора"
    )
    await state.set_state(AdminStates.admin_panel_state)

@admin_panel_rout.message(Command("upload"), AdminStates.admin_panel_state)
async def start_upload(message: types.Message, state: FSMContext):
    await message.answer(
        "Отправьте файл с новой базой часто-задаваемых вопросов и ответов с именем файла \"faq.tsv\". "
        "Данные в файле должны располагаться через знак табуляции: Вопрос \\t Ответ\n\n"
        "Чтобы вернуться обратно в панель администратора используйте /admin"
    )
    await state.set_state(AdminStates.upload_faq_state)

@admin_panel_rout.message(AdminStates.upload_faq_state, F.document)
async def upload_faq(message: types.Message, state: FSMContext, bot: Bot):
    if message.document.file_name != "faq.tsv":
        await message.answer("Не допустимое разрешение файла. Поддерживается только .tsv")
        return
    
    path = "data/faq.tsv"
    file_info = await bot.get_file(message.document.file_id)

    try:
        await bot.download_file(file_info.file_path, path)
        await message.answer("Успешно.")
        app_state.reload_faq()
        await start_admin_panel(message, state)

    except Exception:
        await message.answer("Произошла ошибка. Попробуйте еще раз.\n\n/admin - вернуться в панель администратора\n/exit - выйти из панели администратора")
    



@admin_panel_rout.message(Command("exit"))
async def cancel_state(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if not current_state:
        return
    await state.clear()
    await message.answer("Вы вышли из панели администратора")

