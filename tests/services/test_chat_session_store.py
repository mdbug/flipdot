from app.services.chat_session_store import ChatSessionStore, derive_title


def test_derive_title_collapses_and_truncates():
    assert derive_title("  hello   world ") == "hello world"
    assert derive_title("") == "New conversation"
    long = "x" * 80
    title = derive_title(long)
    assert len(title) <= 60
    assert title.endswith("…")


def test_create_save_load_roundtrip(tmp_path):
    store = ChatSessionStore(tmp_path)
    record = store.create(title="Write HELLO on the display", model="claude-opus-4-8")
    assert record["title"] == "Write HELLO on the display"
    assert record["messages"] == []

    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello!"}]},
    ]
    store.save(record["id"], messages=messages)

    loaded = store.load(record["id"])
    assert loaded["messages"] == messages
    assert loaded["model"] == "claude-opus-4-8"


def test_list_summaries_sorted_and_lightweight(tmp_path):
    store = ChatSessionStore(tmp_path)
    first = store.create(title="first", model=None)
    second = store.create(title="second", model=None)
    store.save(
        second["id"],
        messages=[
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
            {"role": "user", "content": "c"},
        ],
    )

    summaries = store.list_summaries()
    assert [s["id"] for s in summaries][0] == second["id"]  # most recently updated first
    assert all("messages" not in s for s in summaries)
    second_summary = next(s for s in summaries if s["id"] == second["id"])
    assert second_summary["message_count"] == 2  # only the two user turns
    assert first["id"] in {s["id"] for s in summaries}


def test_rename_and_delete(tmp_path):
    store = ChatSessionStore(tmp_path)
    record = store.create(title="old", model=None)
    summary = store.rename(record["id"], "new name")
    assert summary["title"] == "new name"
    assert store.load(record["id"])["title"] == "new name"

    assert store.delete(record["id"]) is True
    assert store.delete(record["id"]) is False
    assert store.load(record["id"]) is None


def test_sanitize_id_rejects_traversal(tmp_path):
    store = ChatSessionStore(tmp_path)
    import pytest

    for bad in ["../etc/passwd", "not-hex", "", "abc"]:
        with pytest.raises(ValueError):
            store.sanitize_id(bad)
