import time
from datetime import datetime

from peewee_aio import AIOModel, Manager
from peewee_aio.fields import AutoField, DateTimeField, IntegerField, TextField

manager = Manager(
    "aiosqlite:///logs.sqlite",
    pragmas=[
        ("journal_mode", "wal"),
        ("cache_size", -64000),
        ("foreign_keys", 1),
        ("ignore_check_constraints", 0),
        ("synchronous", 0),
    ],
)


@manager.register
class Logs(AIOModel):
    id = AutoField(primary_key=True)
    response = TextField()
    tokens = IntegerField()
    timestamp = DateTimeField(default=datetime.now)


async def handler():
    async with manager:
        async with manager.connection():
            await Logs.create_table()


async def insert_log(response: str, tokens: int):
    async with manager:
        async with manager.connection():
            await Logs.create(
                response=response,
                tokens=tokens,
                timestamp=int(time.time()),
            )


async def get_latest_log():
    async with manager:
        async with manager.connection():
            return await Logs.select(Logs).order_by(Logs.timestamp.desc()).limit(1)


async def get_logs():
    async with manager:
        async with manager.connection():
            return await Logs.select(Logs).order_by(Logs.timestamp.desc())
