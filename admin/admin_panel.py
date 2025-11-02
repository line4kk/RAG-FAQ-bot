from aiogram import types, Router
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext
from aiogram.filters import Command

from filters.admin_filter import IsAdmin

admin_panel_rout = Router()
admin_panel_rout.message.filter(IsAdmin())


class AdminStates(StatesGroup):
    admin_panel_state = State()


@admin_panel_rout.message(Command("admin"))
async def start_admin_panel(message: types.Message, state: FSMContext):
    await message.answer(
        "Вы перешли в панель администратора. Доступные команды:\n\n"
        "/download - загрузить новую базу данных FAQ\n"
        "/exit - выйти из панели администратора"
    )
    await state.set_state(AdminStates.admin_panel_state)


@admin_panel_rout.message(Command("exit"))
async def cancel_state(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if not current_state:
        return
    await state.clear()
    await message.answer("Вы вышли из панели администратора")

