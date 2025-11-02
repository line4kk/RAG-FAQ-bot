from aiogram.filters import Filter
from aiogram.types import Message

from app_state import app_state

class IsAdmin(Filter):
    def __init__(self) -> None:
        pass

    async def __call__(self, message: Message) -> bool:
        return str(message.from_user.id) in app_state.admins