"""Bounded Python text output, routed by the writer's execution origin."""

import io


class OutputBuffer:
    def __init__(self, limit=4096):
        self.limit = limit
        self.chunks = []
        self.characters = 0
        self.dropped = 0

    def write(self, execution, channel, text):
        if not text:
            return
        previous = self.chunks[-1] if self.chunks else None
        merge = previous is not None and (previous["execution"], previous["channel"]) == (execution, channel)
        capacity = max(0, self.limit - self.characters)
        if not merge and len(self.chunks) >= 128:
            capacity = 0
        retained = text[:capacity]
        if retained:
            if merge:
                previous["text"] += retained
            else:
                self.chunks.append({"execution": execution, "channel": channel, "text": retained})
            self.characters += len(retained)
        self.dropped += len(text) - len(retained)

    def snapshot(self):
        return {"chunks": [dict(chunk) for chunk in self.chunks],
                "limit": self.limit, "truncated": self.dropped > 0,
                "dropped_characters": self.dropped}


class CapturedStream(io.TextIOBase):
    def __init__(self, channel, original):
        self.channel = channel
        self.original = original

    @property
    def encoding(self):
        return "utf-8"

    def writable(self):
        return True

    def write(self, text):
        from .scope import origin
        if not isinstance(text, str):
            raise TypeError("Text stream writes require str")
        source = origin.get()
        if source is None:
            return self.original.write(text)  # Host/provider lifecycle diagnostics.
        scope, execution = source
        scope.output.write(execution, self.channel, text)
        return len(text)  # Truncation must not turn print() into a code error.

    def flush(self):
        return None
