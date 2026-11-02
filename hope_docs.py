"""Carga de HOPE como pares (cliente -> terapeuta) y recuperación de ejemplos para memoryChatPinecone.py / memoryChatGraph.py.

Corrige un fallo de las versiones anteriores: el filtro `"psychotherapist:" in result.page_content` descartaba TODO lo recuperado, porque las
transcripciones HOPE (columnas Type = T/P y Utterance) no contienen esa cadena; el contexto HOPE nunca llegaba al prompt. Aquí el ejemplo
recuperado es, por construcción, un par «Client: … / Psychotherapist: …», y si no hay ejemplos se falla con un error en lugar de seguir en silencio.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"no se pudo leer {path}")


def _trunc(text: str, max_words: int, tail: bool = False) -> str:
    w = text.split()
    if len(w) <= max_words:
        return text
    return " ".join(w[-max_words:] if tail else w[:max_words])


def load_hope_pairs(docs_dir, max_words: int = 120) -> list[dict]:
    """Pares (turno(s) del cliente -> primera respuesta del terapeuta) de las sesiones HOPE. Une turnos consecutivos del mismo hablante.
    Cada par: {file, client, therapist, text}; `text` = «Client: …\\nPsychotherapist: …» (lo que se muestra al LLM como ejemplo)."""
    pairs = []
    for f in sorted(Path(docs_dir).glob("*.csv")):
        try:
            d = _read_csv(f)
        except Exception:
            continue
        cols = {str(c).strip().lower(): c for c in d.columns}
        if "type" not in cols or "utterance" not in cols:
            continue
        blocks: list[list] = []
        for typ, utt in zip(d[cols["type"]], d[cols["utterance"]]):
            if not isinstance(utt, str) or not utt.strip():
                continue
            sp = str(typ).strip().upper()
            if sp not in ("T", "P"):
                continue
            if blocks and blocks[-1][0] == sp:
                blocks[-1][1].append(utt.strip())
            else:
                blocks.append([sp, [utt.strip()]])
        for (s1, u1), (s2, u2) in zip(blocks, blocks[1:]):
            if s1 == "P" and s2 == "T":
                client, therapist = _trunc(" ".join(u1), max_words, tail=True), _trunc(" ".join(u2), max_words)
                pairs.append({"file": f.name, "client": client, "therapist": therapist,
                              "text": f"Client: {client}\nPsychotherapist: {therapist}"})
    return pairs


def retrieve_therapist_examples(vectorstore, query: str, k: int = 3, fetch_k: int = 10) -> list[str]:
    """Los `k` ejemplos «Client/Psychotherapist» más cercanos a la consulta. Se indexa el texto del cliente y el par completo va en el
    metadato `pair`. Lanza RuntimeError si no hay ninguno (índice vacío o mal construido), para no degradar a «sin RAG» sin avisar."""
    results = vectorstore.similarity_search(query, k=fetch_k)
    out: list[str] = []
    for r in results:
        text = (getattr(r, "metadata", None) or {}).get("pair") or r.page_content
        if text and text not in out:
            out.append(text)
    if not out:
        raise RuntimeError("El RAG no devolvió ejemplos: ¿el índice de documentos está vacío o se construyó con otro formato?")
    return out[:k]
