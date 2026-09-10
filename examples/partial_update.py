"""uv run --locked python examples/partial_update.py — no model API."""
from dataclasses import replace
from functools import partial

from nervus import Capability, Session
from basic_session import TaggedEcho


def main():
    cache = Capability('cache', partial(TaggedEcho, 'cache'))
    reader = Capability('read', partial(TaggedEcho, 'old'), implementation_version='old')
    updated = replace(reader, factory=partial(TaggedEcho, 'new'), implementation_version='new')
    with Session() as session:
        session.publish([cache, reader])
        session.begin_turn()
        session.execute('saved = 42\nalias = tools.read\nfirst = await tools.cache(saved)')
        session.update(replace={reader.identity: updated})
        during = session.execute('value = await alias(saved)', exports=['value'])['value']
        assert during['marker'] == 'old'
        print('Active Turn:', during)
        session.end_turn()
        session.begin_turn()
        result = session.execute('cached = await tools.cache(saved)\nvalue = await alias(saved)',
                                 exports=['cached', 'value'])
        assert result['cached']['calls'] == 2  # Untouched instance kept its state.
        assert result['value']['marker'] == 'new'
        print('Next Turn:', result)
        session.end_turn()
        session.update(remove=[reader.identity])
        session.update(add=[updated])  # Explicit restoration of the same identity/interface.
    print('Session closed.')


if __name__ == '__main__':
    main()
