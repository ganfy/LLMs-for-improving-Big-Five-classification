import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hope_docs import load_hope_pairs, retrieve_therapist_examples  # noqa: E402


def _csv(d: Path, name: str, rows, cols=("Type", "Utterance")):
    pd.DataFrame(rows, columns=list(cols)).to_csv(d / name, index=False)


def test_load_pairs_merges_turns_and_ignores_other_files(tmp_path):
    _csv(tmp_path, "s1.csv", [("P", "I feel tired."), ("P", "Work is heavy."), ("T", "What makes work heavy?"), ("T", "Tell me more."),
                              ("P", "My boss."), ("T", "How is that for you?")])
    _csv(tmp_path, "s2.csv", [("type", "x")], cols=("Foo", "Bar"))   # sin columnas Type/Utterance: se ignora
    pairs = load_hope_pairs(tmp_path)
    assert [p["client"] for p in pairs] == ["I feel tired. Work is heavy.", "My boss."]
    assert pairs[0]["therapist"] == "What makes work heavy? Tell me more."
    assert pairs[0]["text"] == "Client: I feel tired. Work is heavy.\nPsychotherapist: What makes work heavy? Tell me more."


def test_pairs_contain_the_marker_the_old_filter_looked_for_in_its_own_case(tmp_path):
    """El filtro antiguo exigía «psychotherapist:» en minúsculas; el formato real de HOPE no lo contiene (por eso nunca pasaba nada)."""
    _csv(tmp_path, "s1.csv", [("P", "hello there"), ("T", "hi, how are you?")])
    raw = (tmp_path / "s1.csv").read_text()
    assert "psychotherapist:" not in raw.lower()
    assert "Psychotherapist:" in load_hope_pairs(tmp_path)[0]["text"]


def test_header_case_and_long_turns(tmp_path):
    _csv(tmp_path, "s1.csv", [("p", "a " * 300), ("t", "b " * 300)], cols=("type", "utterance"))
    p = load_hope_pairs(tmp_path, max_words=50)[0]
    assert len(p["client"].split()) == 50 and len(p["therapist"].split()) == 50


class FakeStore:
    def __init__(self, results):
        self.results, self.calls = results, []

    def similarity_search(self, q, k=4):
        self.calls.append((q, k))
        return self.results[:k]


def test_retrieve_returns_pair_from_metadata_dedupes_and_limits():
    r = lambda client, pair: SimpleNamespace(page_content=client, metadata={"pair": pair})  # noqa: E731
    store = FakeStore([r("a", "Client: a\nPsychotherapist: A"), r("a", "Client: a\nPsychotherapist: A"), r("b", "Client: b\nPsychotherapist: B"),
                       r("c", "Client: c\nPsychotherapist: C"), r("d", "Client: d\nPsychotherapist: D")])
    out = retrieve_therapist_examples(store, "q", k=3)
    assert out == ["Client: a\nPsychotherapist: A", "Client: b\nPsychotherapist: B", "Client: c\nPsychotherapist: C"]
    assert store.calls == [("q", 10)]


def test_retrieve_falls_back_to_page_content_and_fails_loudly_when_empty():
    assert retrieve_therapist_examples(FakeStore([SimpleNamespace(page_content="texto", metadata={})]), "q") == ["texto"]
    with pytest.raises(RuntimeError):
        retrieve_therapist_examples(FakeStore([]), "q")


@pytest.mark.skipif(not Path("/mnt/user-data/uploads/TESIS/HOPE_WSDM_2022/All").is_dir(), reason="HOPE local no disponible")
def test_real_hope_corpus_gives_thousands_of_pairs():
    assert len(load_hope_pairs("/mnt/user-data/uploads/TESIS/HOPE_WSDM_2022/All")) > 1000
