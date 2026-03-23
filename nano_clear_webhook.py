import asyncio
from telegram import Bot

async def clear():
    bot = Bot(token="8663010744:AAEXs29ahTYmKHC43xa3VdkykbtfwE5MLVo")  # <-- apna token yahan
    await bot.delete_webhook(drop_pending_updates=True)
    print("Webhook cleared!")

asyncio.run(clear())