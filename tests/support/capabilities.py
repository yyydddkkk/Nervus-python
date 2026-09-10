"""SQLite is a test Adapter; no database knowledge belongs in nervus."""

import json
from pathlib import Path
import sqlite3
from uuid import uuid4


class SQLiteQuery:
    def __init__(self, audit_path, marker, fail_initialize=False, fail_close=False):
        # Resource-free construction is important for initialization failure cleanup.
        self.audit_path = audit_path
        self.marker = marker
        self.fail_initialize = fail_initialize
        self.fail_close = fail_close
        self.connection = None
        self.instance = uuid4().hex
        self.close_count = 0

    def record(self, phase, **details):
        with Path(self.audit_path).open("a") as stream:
            stream.write(json.dumps({"phase": phase, "marker": self.marker,
                                     "instance": self.instance, **details}) + "\n")

    async def initialize(self):
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.record("connected")
        if self.fail_initialize:
            raise RuntimeError("injected initialization failure")
        self.connection.execute("CREATE TABLE state(marker TEXT, count INTEGER)")
        self.connection.execute("INSERT INTO state VALUES (?, 0)", (self.marker,))
        self.record("ready", probe=self.connection.execute("SELECT 1").fetchone()[0])

    async def __call__(self, value):
        self.connection.execute("UPDATE state SET count = count + 1")
        marker, count = self.connection.execute("SELECT marker, count FROM state").fetchone()
        self.record("query", count=count)
        return {"marker": marker, "count": count, "value": value, "instance": self.instance}

    async def close(self):
        self.close_count += 1
        if self.close_count != 1:
            raise AssertionError("close called more than once")
        if self.connection is not None:
            self.connection.close()
            try:
                self.connection.execute("SELECT 1")
            except sqlite3.ProgrammingError:
                self.record("closed", verified=True, close_count=self.close_count)
            else:
                raise AssertionError("Connection remained usable after close")
        if self.fail_close:
            raise RuntimeError("injected release failure")


class Echo:
    async def initialize(self):
        pass

    async def __call__(self, value):
        return value

    async def close(self):
        pass
