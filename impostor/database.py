from typing import *

import yaml
import os

import sqlite3

from datetime import datetime, timezone

config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))

ddl_chat = """CREATE TABLE IF NOT EXISTS chat (
    id integer NOT NULL PRIMARY KEY,
    in_guild boolean NOT NULL,
    type varchar(20) NOT NULL,
    guild_id integer,
    name varchar(1024) NOT NULL,
    last_forget datetime,
    prefix varchar(8) DEFAULT ",",
    eagerness real DEFAULT 2.3,
    interactivity real DEFAULT 0.4,
    override_chat_settings boolean DEFAULT false,
    since_last_reply integer DEFAULT 0,
    FOREIGN KEY (guild_id) REFERENCES guild (id)
);"""

ddl_guild = """CREATE TABLE IF NOT EXISTS guild (
    id integer NOT NULL PRIMARY KEY,
    name varchar(1024) NOT NULL,
    prefix varchar(8) DEFAULT ",",
    eagerness real DEFAULT 1,
    interactivity real DEFAULT 0
);"""

add_guild_sql = """INSERT INTO guild (id, name) VALUES (?, ?);"""

add_chat_sql = """INSERT INTO chat (id, in_guild, type, guild_id, name) VALUES (?, ?, ?, ?, ?);"""

guild_exists_sql = """SELECT COUNT(id) FROM guild WHERE id=?"""

chat_exists_sql = """SELECT COUNT(id) FROM chat WHERE id=?"""

get_guild_by_id_sql = """SELECT id, name, prefix, eagerness, interactivity FROM guild WHERE id = ?;"""

get_chat_by_id_sql = """SELECT
id, in_guild, type, guild_id, name, last_forget, prefix, eagerness, interactivity, override_chat_settings, since_last_reply
FROM chat
WHERE id = ?;"""

reset_last_forget_chat_sql = """UPDATE chat SET last_forget=? WHERE id=?"""

set_chat_prefix_sql = """UPDATE chat SET prefix=? WHERE id=?"""

set_guild_prefix_sql = """UPDATE guild SET prefix=? WHERE id=?"""

set_chat_eagerness_interactivity_sql = """UPDATE chat SET eagerness=?, interactivity=? WHERE id=?"""

set_guild_eagerness_interactivity_sql = """UPDATE guild SET eagerness=?, interactivity=? WHERE id=?"""

get_since_last_reply_sql = """SELECT since_last_reply FROM chat WHERE id=?"""

set_since_last_reply_sql = """UPDATE chat SET since_last_reply=? WHERE id=?"""


class GuildObject:
    def __init__(self, guild_id: int, name: str, prefix: str, eagerness: float, interactivity: float):
        self.guild_id = guild_id
        self.name = name
        self.prefix = prefix
        self.eagerness = eagerness
        self.interactivity = interactivity


class ChatObject:
    def __init__(self, chat_id: int, in_guild: bool, type: str, guild_id: Optional[int], name: str,
                 last_forget: Optional[datetime], prefix: str, eagerness: float, interactivity: float,
                 override_chat_settings: bool, since_last_reply: int):
        self.chat_id = chat_id
        self.in_guild = in_guild
        self.type = type
        self.guild_id = guild_id
        self.name = name
        self.last_forget = last_forget
        self.prefix = prefix
        self.eagerness = eagerness
        self.interactivity = interactivity
        self.override_chat_settings = override_chat_settings
        self.since_last_reply = since_last_reply


class DatabaseAccessor:
    def __init__(self):
        print("Initialising database...")
        self.conn = sqlite3.connect(config["bot"]["db_path"])
        self.c = self.conn.cursor()
        self.c.execute(ddl_guild)
        self.c.execute(ddl_chat)
        self.conn.commit()
        print("Done initialising")

    def add_guild(self, guild_id: int, name: str):
        self.c.execute(add_guild_sql, (guild_id, name))
        self.conn.commit()

    def add_chat(self, chat_id: int, in_guild: bool, type: str, guild_id: Optional[int], name: str):
        self.c.execute(add_chat_sql, (chat_id, in_guild, type, guild_id, name))
        self.conn.commit()

    def guild_exists(self, guild_id: int) -> bool:
        rows = list(self.c.execute(guild_exists_sql, (guild_id, )))
        return rows[0][0] > 0

    def chat_exists(self, chat_id: int) -> bool:
        rows = list(self.c.execute(chat_exists_sql, (chat_id, )))
        return rows[0][0] > 0

    def get_guild_by_id(self, guild_id: int) -> Optional[GuildObject]:
        guilds = list(self.c.execute(get_guild_by_id_sql, (guild_id,)))
        if len(guilds) == 0:
            return None
        guild = guilds[0]
        return GuildObject(*guild)

    def get_chat_by_id(self, chat_id: int) -> Optional[ChatObject]:
        chats = list(self.c.execute(get_chat_by_id_sql, (chat_id,)))
        if len(chats) == 0:
            return None
        chat = chats[0]
        return ChatObject(*chat[0:5],
                          None if chat[5] is None else datetime.strptime(chat[5], "%Y-%m-%d %H:%M:%S"), *chat[6:])

    def reset_last_forget_chat(self, chat_id: int):
        self.c.execute(reset_last_forget_chat_sql, (datetime.strftime(datetime.now(timezone.utc), "%Y-%m-%d %H:%M:%S"),
                                                    chat_id))
        self.set_since_last_reply(chat_id, 0)
        self.conn.commit()

    def set_chat_prefix(self, chat_id: int, prefix: str):
        assert(len(prefix) <= 8)
        self.c.execute(set_chat_prefix_sql, (prefix, chat_id))
        self.conn.commit()

    def set_guild_prefix(self, guild_id: int, prefix: str):
        assert (len(prefix) <= 8)
        self.c.execute(set_guild_prefix_sql, (prefix, guild_id))
        self.conn.commit()

    def set_chat_eagerness_interactivity(self, chat_id: int, eagerness: float, interactivity: float):
        self.c.execute(set_chat_eagerness_interactivity_sql, (eagerness, interactivity, chat_id))
        self.conn.commit()

    def set_guild_eagerness_interactivity(self, guild_id: int, eagerness: float, interactivity: float):
        self.c.execute(set_guild_eagerness_interactivity_sql, (eagerness, interactivity, guild_id))
        self.conn.commit()

    def get_since_last_reply(self, chat_id: int) -> Optional[int]:
        rows = list(self.c.execute(get_since_last_reply_sql, (chat_id, )))
        if len(rows) == 0:
            return None
        return rows[0][0]

    def set_since_last_reply(self, chat_id: int, since_last_reply: int):
        self.c.execute(set_since_last_reply_sql, (since_last_reply, chat_id))
        self.conn.commit()

    def increment_last_reply(self, chat_id: int) -> bool:
        value = self.get_since_last_reply(chat_id)
        if value is None:
            return False
        self.set_since_last_reply(chat_id, value + 1)
        return True

    def close_connection(self):
        self.conn.close()


if __name__ == "__main__":
    da = DatabaseAccessor()
