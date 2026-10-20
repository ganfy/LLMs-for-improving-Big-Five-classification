#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_rag.py
===============
Evaluación de la parte RAG del proyecto (recuperación, reranking, memoria y
calidad del diálogo). Es un archivo independiente: no modifica ni importa
ninguno de los scripts existentes del repositorio y no reentrena modelos.

Subcomandos
-----------
dialogue         Calidad del diálogo generado con RAG, a partir de los logs
                 conversation.txt y conversationGraph.txt. Compara
                 RAG sin memoria / memoria vectorial / memoria de grafo.
                 Offline: no necesita GPU, LLM ni internet.
labels           Fidelidad de las etiquetas de RAGv.csv y RAGg.csv frente al
                 código Big Five de la persona simulada (personalidades.txt)
                 y balance de clases de todos los datasets. Offline.
corpus-check     Diagnóstico del corpus HOPE (turnos, pares cliente→terapeuta) y
                 comprobación del filtro "psychotherapist:" de memoryChat*.py.
retrieval-dump   Re-ejecuta la recuperación (bge-small + cross-encoder, igual
                 que RAGv4.py) sobre las consultas reales de los logs y guarda
                 los candidatos en JSONL. Requiere el corpus HOPE y
                 llama-index + fastembed + sentence-transformers.
retrieval-judge  (Opcional) Etiqueta la relevancia de cada candidato con un
                 LLM local vía Ollama (por defecto llama3).
retrieval-score  Métricas de recuperación y reranking a partir del JSONL.
all              Ejecuta dialogue + labels y escribe rag_eval_report.md.

Ejemplos
--------
    python evaluate_rag.py all
    python evaluate_rag.py corpus-check --docs HOPE_WSDM_2022/All
    python evaluate_rag.py retrieval-dump --docs HOPE_WSDM_2022/All --max-queries 60
    python evaluate_rag.py retrieval-judge --model llama3
    python evaluate_rag.py retrieval-score

Para comparar una mejora con la configuración de la tesis, usa --out distinto:
    ... retrieval-dump --docs ... --out rag_eval_results/base
    ... retrieval-dump --docs ... --out rag_eval_results/pares --corpus pairs
    ... retrieval-dump --docs ... --out rag_eval_results/e5 --corpus pairs --embedder st \\
        --embed-model intfloat/multilingual-e5-small --query-prefix "query: " --doc-prefix "passage: "

Dependencias: numpy, pandas, scikit-learn, scipy (las cuatro ya las usa el
proyecto). Solo retrieval-dump necesita además llama-index (0.10.x) y
fastembed; retrieval-judge solo necesita un servidor Ollama en marcha.

Los resultados se guardan en ./rag_eval_results/ (CSV + informes .md).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "rag_eval_results"
TRAITS = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
COND_LABEL = {
    "no_memory": "RAG sin memoria",
    "vector": "RAG + memoria vectorial",
    "graph": "RAG + memoria de grafo",
}
COND_ORDER = ["no_memory", "vector", "graph"]
N_BOOT = 2000


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------
def fold(s: str) -> str:
    """Minúsculas y sin tildes."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


_ES = set("el la los las de del que y en un una por con para mi es se pero como muy más al su sus estoy siento".split())
_EN = set("the and of to in is that it i my you for with but not feel am are was".split())


def detect_lang(text: str) -> str:
    toks = re.findall(r"[a-záéíóúñü]+", text.lower())
    es, en = sum(t in _ES for t in toks), sum(t in _EN for t in toks)
    return "es" if es > en else "en" if en > es else "unk"


def boot_mean_ci(x, n=N_BOOT, seed=0):
    """Media y IC 95 % bootstrap (percentiles)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def md_table(df: pd.DataFrame, ndigits: int = 3) -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "—" if np.isnan(v) else f"{v:.{ndigits}f}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def read_csv_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def read_flat_csv(path: Path) -> str:
    """Une todas las celdas de un CSV en un solo texto (formato de conversation*.csv)."""
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        return " ".join(" ".join(r) for r in csv.reader(f)).strip()


# ---------------------------------------------------------------------------
# Personas simuladas (verdad de terreno) y emparejamiento texto -> persona
# ---------------------------------------------------------------------------
def load_persona_codes(path: Path) -> dict[int, list[int]]:
    """personalidades.txt -> {id: [E, N, A, C, O]} con valores 0/1."""
    codes = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\s*(\d+)\.\s+\*\*([01]{5})\*\*", line)
        if m:
            codes[int(m.group(1))] = [int(c) for c in m.group(2)]
    return codes


def load_persona_sources(root: Path, kind: str) -> dict[int, str]:
    """Respuestas del cliente por persona. kind='vector' usa conversation<N>.csv
    (o el archivo llamado '<N>'), kind='graph' usa conversationGraph<N>.csv."""
    src = {}
    for n in range(1, 51):
        cands = [root / f"conversationGraph{n}.csv"] if kind == "graph" else [root / f"conversation{n}.csv", root / str(n)]
        for p in cands:
            if p.is_file():
                src[n] = read_flat_csv(p)
                break
    return src


def match_personas(texts: list[str], sources: dict[int, str], min_sim: float = 0.4):
    """Vecino más cercano TF-IDF entre cada texto y las respuestas de cada persona.
    Devuelve [(persona|None, similitud)]."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not sources or not texts:
        return [(None, 0.0)] * len(texts)
    keys = list(sources)
    vec = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode").fit([*sources.values(), *texts])
    sim = cosine_similarity(vec.transform(texts), vec.transform([sources[k] for k in keys]))
    best = sim.argmax(axis=1)
    return [(keys[b] if sim[i, b] >= min_sim else None, float(sim[i, b])) for i, b in enumerate(best)]


# ---------------------------------------------------------------------------
# Parseo de los logs de conversación
# ---------------------------------------------------------------------------
_SPEAKER = re.compile(r"^(Client|Psychotherapist):[ \t]*(.*)$")
_LABELS = re.compile(r'^\s*"?[yn]"?(\s*,\s*"?[yn]"?){4}\s*$')
_GREETINGS = ("hola, cómo te sientes hoy", "hi, how are you feeling today")


def parse_block(block: str):
    """Un bloque = una conversación. Devuelve (turnos, etiquetas|None).
    Los párrafos extra de un turno del psicoterapeuta se anexan a ese turno."""
    turns, labels, para_break = [], None, False
    for raw in block.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            para_break = True
            continue
        if _LABELS.match(line):
            labels = re.findall(r"[yn]", line.lower())
            continue
        m = _SPEAKER.match(line)
        if m:
            role = "client" if m.group(1) == "Client" else "therapist"
            turns.append({"role": role, "text": m.group(2).strip(), "paras": 1})
        elif not turns and fold(line.strip()).startswith(tuple(fold(g) for g in _GREETINGS)):
            turns.append({"role": "therapist", "text": line.strip(), "paras": 1})
        elif turns:
            if para_break:
                turns[-1]["paras"] += 1
            turns[-1]["text"] += ("\n" if para_break else " ") + line.strip()
        para_break = False
    if turns and turns[0]["role"] == "therapist":
        turns[0]["greeting"] = True  # saludo fijo: no cuenta como pregunta generada
    return turns, labels


def parse_log(path: Path, cond_rule: str):
    """cond_rule: 'mixed' (conversation.txt: sin memoria si el bloque empieza con
    'Psychotherapist:', vectorial si empieza con el saludo suelto) o 'graph'."""
    if not path.is_file():
        print(f"  [aviso] no existe {path.name}; se omite", file=sys.stderr)
        return []
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    convs = []
    for block in (b for b in re.split(r"\n\s*\n\s*\n", text) if b.strip()):
        turns, labels = parse_block(block)
        if not turns:
            continue
        first = next(l for l in block.split("\n") if l.strip())
        cond = "graph" if cond_rule == "graph" else ("no_memory" if first.strip().startswith("Psychotherapist:") else "vector")
        client_text = " ".join(t["text"] for t in turns if t["role"] == "client")
        convs.append({"cond": cond, "turns": turns, "labels": labels, "client_text": client_text, "lang": detect_lang(client_text)})
    return convs


def load_conversations(root: Path):
    convs = parse_log(root / "conversation.txt", "mixed") + parse_log(root / "conversationGraph.txt", "graph")
    # persona de cada conversación (solo posible en memoria vectorial y de grafo)
    for kind in ("vector", "graph"):
        idx = [i for i, c in enumerate(convs) if c["cond"] == kind]
        res = match_personas([convs[i]["client_text"] for i in idx], load_persona_sources(root, kind))
        for i, (p, s) in zip(idx, res):
            convs[i]["persona"], convs[i]["persona_sim"] = p, s
    for c in convs:
        c.setdefault("persona", None)
        c.setdefault("persona_sim", float("nan"))
    return convs


# ---------------------------------------------------------------------------
# Métricas de diálogo
# ---------------------------------------------------------------------------
_WH = r"(?:what|how|why|when|where|which|who|whom|whose|que|como|por que|para que|cuando|donde|cual|cuales|quien|quienes|cuanto|cuanta|cuantos|cuantas)"
_PREP = r"(?:a|de|del|en|con|por|para|desde|hasta|sobre|about|in|on|with|for|to|from|at|of)"
_INVITE = (
    r"(?:(?:can|could|would)\s+you\s+(?:please\s+)?(?:tell|share|describe|elaborate|explain|talk|walk|say)"
    r"|(?:puedes|podrias|quisieras|te gustaria)\s+(?:por favor\s+)?(?:contarme|compartir|describir|explicar|hablar|decirme"
    r"|profundizar|explorar|detallar|ampliar|comentarme|comentar))"
)
_OPEN = re.compile(rf"^(?:(?:y|and|but|pero|so|entonces),?\s+)?(?:{_PREP}\s+)?{_WH}\b|^{_INVITE}\b")


def split_sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"(?<=[.!?…])\s+|\n+", text) if p.strip()]


def extract_questions(text: str) -> list[str]:
    out = []
    for s in split_sentences(text):
        if "?" in s:
            q = s.split("¿")[-1] if "¿" in s else s
            out.append(re.sub(r"^[\W_]+", "", q))
    return out


def is_open_question(q: str) -> bool:
    """Heurística: abierta si empieza con palabra interrogativa (qué, cómo, why...)
    o con una invitación ('¿Puedes contarme...?', 'Can you tell me...')."""
    return bool(_OPEN.search(fold(re.sub(r"^[\W_]+", "", q))))


def make_vectorizer(corpus: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    # max_df=0.5 elimina automáticamente palabras vacías en ES y EN; si el corpus es tan
    # pequeño o repetitivo que no queda vocabulario, se reintenta sin podar.
    for max_df in ((0.5, 1.0) if len(corpus) >= 20 else (1.0,)):
        try:
            return TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", max_df=max_df).fit(corpus)
        except ValueError:
            continue
    raise SystemExit("No hay texto suficiente para construir el vocabulario TF-IDF.")


def conv_metrics(conv: dict, vec) -> dict:
    turns = conv["turns"]
    gen = []  # turnos generados por el LLM (se excluye el saludo fijo)
    last_client = None
    for i, t in enumerate(turns):
        if t["role"] == "client":
            last_client = t["text"]
        elif not t.get("greeting"):
            nxt = next((u["text"] for u in turns[i + 1:] if u["role"] == "client"), None)
            gen.append({"text": t["text"], "paras": t["paras"], "prev": last_client, "next": nxt})
    m = {"n_gen_turns": len(gen)}
    if not gen:
        return m
    qs = [extract_questions(g["text"]) for g in gen]
    has_q = np.array([len(q) > 0 for q in qs])
    has_open = np.array([any(is_open_question(x) for x in q) for q in qs])
    m.update(
        words_per_turn=np.mean([len(words(g["text"])) for g in gen]),
        paragraphs_per_turn=np.mean([g["paras"] for g in gen]),
        q_rate=has_q.mean(),
        no_q_rate=1 - has_q.mean(),
        open_q_rate=has_open.mean(),
        closed_only_rate=(has_q & ~has_open).mean(),
        questions_per_turn=np.mean([len(q) for q in qs]),
    )
    replies = [len(words(g["next"])) for g in gen if g["next"]]
    m["client_reply_words"] = float(np.mean(replies)) if replies else float("nan")

    V = vec.transform([g["text"] for g in gen])
    echo = [float(V[i].multiply(vec.transform([g["prev"]])).sum()) for i, g in enumerate(gen) if g["prev"]]
    m["client_echo"] = float(np.mean(echo)) if echo else float("nan")

    toks = [words(g["text"]) for g in gen]
    uni = [w for t in toks for w in t]
    bi = [b for t in toks for b in zip(t, t[1:])]
    m["distinct1"] = len(set(uni)) / len(uni) if uni else float("nan")
    m["distinct2"] = len(set(bi)) / len(bi) if bi else float("nan")

    if len(gen) >= 2:
        S = (V @ V.T).toarray()
        iu = np.triu_indices(len(gen), 1)
        sims = S[iu]
        m["rep_sim_mean"], m["rep_sim_max"] = sims.mean(), sims.max()
        seen, shares = set(), []
        for t in toks:
            tri = list(zip(t, t[1:], t[2:]))
            if tri and seen:
                shares.append(np.mean([g in seen for g in tri]))
            seen.update(tri)
        m["repeated_trigram_rate"] = float(np.mean(shares)) if shares else float("nan")
        openers = [tuple(t[:3]) for t in toks]
        m["repeated_opening_rate"] = float(np.mean([o in openers[:i] for i, o in enumerate(openers) if i > 0]))
    return m


METRIC_INFO = [  # (columna, descripción, ¿mejor?)
    ("n_gen_turns", "Turnos generados por el LLM (sin saludo)", ""),
    ("words_per_turn", "Palabras por turno del terapeuta", ""),
    ("paragraphs_per_turn", "Párrafos por turno del terapeuta", "↓"),
    ("q_rate", "Turnos con al menos una pregunta", "↑"),
    ("no_q_rate", "Turnos sin ninguna pregunta (solo afirmaciones)", "↓"),
    ("open_q_rate", "Turnos con pregunta abierta (heurística)", "↑"),
    ("closed_only_rate", "Turnos con solo preguntas cerradas", "↓"),
    ("questions_per_turn", "Preguntas por turno", ""),
    ("client_reply_words", "Palabras de la respuesta del cliente al turno", "↑"),
    ("client_echo", "Similitud léxica turno–mensaje previo del cliente (TF-IDF)", ""),
    ("rep_sim_mean", "Similitud media entre turnos de la misma conversación", "↓"),
    ("rep_sim_max", "Similitud máxima entre dos turnos de la misma conversación", "↓"),
    ("repeated_trigram_rate", "Trigramas de un turno ya usados en turnos anteriores", "↓"),
    ("repeated_opening_rate", "Turnos que repiten las 3 primeras palabras de uno anterior", "↓"),
    ("distinct1", "Diversidad léxica distinct-1", "↑"),
    ("distinct2", "Diversidad léxica distinct-2", "↑"),
]
METRICS = [m for m, _, _ in METRIC_INFO]


def run_dialogue(root: Path, out: Path) -> str:
    convs = load_conversations(root)
    if not convs:
        raise SystemExit("No se encontraron conversaciones (conversation.txt / conversationGraph.txt).")
    corpus = [t["text"] for c in convs for t in c["turns"]]
    vec = make_vectorizer(corpus)
    rows = []
    for k, c in enumerate(convs):
        rows.append({"conv_id": k, "cond": c["cond"], "persona": c["persona"], "lang": c["lang"], **conv_metrics(c, vec)})
    per = pd.DataFrame(rows)
    per.to_csv(out / "dialogue_per_conversation.csv", index=False)

    agg = []
    for cond in COND_ORDER:
        sub = per[per.cond == cond]
        for m in METRICS:
            if m in sub and len(sub):
                mean, lo, hi = boot_mean_ci(sub[m].values)
                agg.append({"cond": cond, "metric": m, "n_conv": int(sub[m].notna().sum()), "mean": mean, "ci_lo": lo, "ci_hi": hi})
    agg = pd.DataFrame(agg)
    agg.to_csv(out / "dialogue_metrics_by_condition.csv", index=False)

    # Comparación pareada memoria vectorial vs grafo (misma persona, mismos prompts)
    from scipy.stats import wilcoxon

    first = per[per.persona.notna()].drop_duplicates(["cond", "persona"])
    pv = first[first.cond == "vector"].set_index("persona")
    pg = first[first.cond == "graph"].set_index("persona")
    shared = pv.index.intersection(pg.index)
    paired = []
    for m in METRICS:
        if m == "n_gen_turns" or m not in per:
            continue
        a, b = pg.loc[shared, m].astype(float), pv.loc[shared, m].astype(float)
        ok = a.notna() & b.notna()
        d = (a - b)[ok]
        p = float("nan")
        if len(d) >= 5 and np.any(d != 0):
            p = float(wilcoxon(a[ok], b[ok]).pvalue)
        paired.append({"metric": m, "n_pairs": int(ok.sum()), "mean_vector": b[ok].mean(), "mean_graph": a[ok].mean(), "diff_graph_minus_vector": d.mean() if len(d) else float("nan"), "wilcoxon_p": p})
    paired = pd.DataFrame(paired)
    paired.to_csv(out / "dialogue_paired_vector_vs_graph.csv", index=False)

    # ---- informe ----
    comp = per.groupby("cond").agg(n_conv=("conv_id", "size"), pct_es=("lang", lambda s: (s == "es").mean()), n_personas=("persona", "nunique")).reindex(COND_ORDER).reset_index()
    comp["cond"] = comp["cond"].map(COND_LABEL)
    wide = []
    for m, desc, better in METRIC_INFO:
        row = {"Métrica": f"{desc} {better}".strip()}
        for cond in COND_ORDER:
            r = agg[(agg.cond == cond) & (agg.metric == m)]
            row[COND_LABEL[cond]] = "—" if r.empty else f"{r['mean'].iloc[0]:.3f} [{r['ci_lo'].iloc[0]:.3f}, {r['ci_hi'].iloc[0]:.3f}]"
        wide.append(row)
    pt = paired.copy()
    pt["metric"] = pt["metric"].map({m: d for m, d, _ in METRIC_INFO})
    pt.columns = ["Métrica", "Pares", "Vectorial", "Grafo", "Grafo − Vectorial", "p (Wilcoxon)"]
    md = [
        "## 1. Calidad del diálogo generado con RAG",
        "",
        "Fuente: `conversation.txt` y `conversationGraph.txt`. Se excluye el saludo fijo inicial; "
        "cada métrica se calcula por conversación y se resume con la media e IC 95 % bootstrap.",
        "",
        "**Composición de los datos analizados**",
        "",
        md_table(comp.rename(columns={"cond": "Condición", "n_conv": "Conversaciones", "pct_es": "% en español", "n_personas": "Personas distintas"})),
        "",
        "**Resultados por condición** (media [IC 95 %]; ↑ mejor si es mayor, ↓ mejor si es menor)",
        "",
        md_table(pd.DataFrame(wide)),
        "",
        "**Comparación pareada memoria vectorial vs. memoria de grafo** (misma persona simulada)",
        "",
        md_table(pt),
        "",
        "> Lectura cuidadosa: la condición *sin memoria* proviene de otros participantes y casi siempre en inglés, "
        "por lo que su diferencia con las otras dos mezcla el efecto de la memoria con el del idioma y los participantes. "
        "La comparación controlada es la pareada vectorial vs. grafo. Los valores p no están corregidos por comparaciones múltiples. "
        "La clasificación abierta/cerrada es una heurística por palabras interrogativas: conviene revisar a mano una muestra.",
    ]
    return "\n".join(md)


# ---------------------------------------------------------------------------
# Fidelidad de etiquetas vs. persona simulada y balance de clases
# ---------------------------------------------------------------------------
def norm_label(v) -> float:
    s = str(v).strip().strip('"').strip().lower()
    return 1.0 if s == "y" else 0.0 if s == "n" else float("nan")


def _f1(g, y):
    tp, fp, fn = int(((g == 1) & (y == 1)).sum()), int(((g == 0) & (y == 1)).sum()), int(((g == 1) & (y == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def macro_f1(G, Y):
    return float(np.mean([_f1(G[:, j], Y[:, j]) for j in range(G.shape[1])]))


def macro_acc(G, Y):
    return float((G == Y).mean())


def boot_stat(G, Y, fn, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    vals = [fn(G[i], Y[i]) for i in (rng.integers(0, len(G), len(G)) for _ in range(n))]
    return fn(G, Y), *np.percentile(vals, [2.5, 97.5])


def per_trait_table(G, Y) -> pd.DataFrame:
    from sklearn.metrics import cohen_kappa_score

    rows = []
    for j, t in enumerate(TRAITS):
        g, y = G[:, j], Y[:, j]
        tp, fp, fn = int(((g == 1) & (y == 1)).sum()), int(((g == 0) & (y == 1)).sum()), int(((g == 1) & (y == 0)).sum())
        kappa = cohen_kappa_score(g, y) if len(set(g)) > 1 or len(set(y)) > 1 else float("nan")
        rows.append({
            "Rasgo": t[1:], "Accuracy": (g == y).mean(), "Precision": tp / (tp + fp) if tp + fp else 0.0,
            "Recall": tp / (tp + fn) if tp + fn else 0.0, "F1": _f1(g, y), "Kappa": kappa,
            "% positivo (persona)": g.mean(), "% positivo (etiqueta)": y.mean(),
            "Acc. clase mayoritaria": max(g.mean(), 1 - g.mean()),
        })
    return pd.DataFrame(rows)


def run_labels(root: Path, out: Path) -> str:
    codes = load_persona_codes(root / "personalidades.txt")
    if not codes:
        raise SystemExit("No se pudo leer personalidades.txt")
    datasets = {"RAG + memoria vectorial (RAGv.csv)": ("RAGv.csv", "vector"), "RAG + memoria de grafo (RAGg.csv)": ("RAGg.csv", "graph")}
    md = ["## 2. Fidelidad de las etiquetas frente a la persona simulada", "",
          "Cada muestra de RAGv/RAGg proviene de una persona simulada (`prompt` + `personalidades.txt`) con un código Big Five conocido "
          "(orden E, N, A, C, O). Se compara la etiqueta asignada por GPT-4 al texto recolectado con ese código. "
          "Mide cuánto de la información de personalidad sobrevive al pipeline *diálogo RAG → texto → etiquetado*.", ""]
    aligned, summary, per_rows = {}, [], []
    for name, (fname, kind) in datasets.items():
        path = root / fname
        if not path.is_file():
            md.append(f"_No se encontró {fname}._\n")
            continue
        df = read_csv_any(path)
        res = match_personas(df["TEXT"].astype(str).tolist(), load_persona_sources(root, kind))
        ids = [p for p, _ in res]
        keep = [i for i, p in enumerate(ids) if p in codes]
        dup = len(keep) - len({ids[i] for i in keep})
        if dup:
            print(f"  [aviso] {fname}: {dup} filas se emparejaron con una persona ya usada", file=sys.stderr)
        G = np.array([codes[ids[i]] for i in keep], dtype=float)
        Y = np.array([[norm_label(df[t].iloc[i]) for t in TRAITS] for i in keep])
        ok = ~np.isnan(Y).any(axis=1)
        G, Y, keep = G[ok], Y[ok], [k for k, o in zip(keep, ok) if o]
        aligned[name] = (G, Y, [ids[i] for i in keep])
        tt = per_trait_table(G, Y)
        tt.insert(0, "Dataset", name)
        per_rows.append(tt)
        f1m, f1lo, f1hi = boot_stat(G, Y, macro_f1)
        acc, alo, ahi = boot_stat(G, Y, macro_acc)
        exact = float((G == Y).all(axis=1).mean())
        sims = [res[i][1] for i in keep]
        summary.append({"Dataset": name, "Muestras emparejadas": len(G), "Sim. mínima del emparejamiento": min(sims),
                        "Macro-F1 [IC95]": f"{f1m:.3f} [{f1lo:.3f}, {f1hi:.3f}]",
                        "Accuracy media [IC95]": f"{acc:.3f} [{alo:.3f}, {ahi:.3f}]", "Hamming loss": 1 - acc,
                        "Acierto de los 5 rasgos (azar 0.031)": exact})
    if per_rows:
        pd.concat(per_rows).to_csv(out / "labels_per_trait.csv", index=False)
        pd.DataFrame(summary).to_csv(out / "labels_summary.csv", index=False)
        md += [md_table(pd.DataFrame(summary)), ""]
        for tt in per_rows:
            md += [f"**{tt['Dataset'].iloc[0]}**", "", md_table(tt.drop(columns="Dataset")), ""]
        names = list(aligned)
        if len(names) == 2:  # diferencia pareada sobre personas comunes
            (Gv, Yv, iv), (Gg, Yg, ig) = aligned[names[0]], aligned[names[1]]
            shared = sorted(set(iv) & set(ig))
            if len(shared) >= 10:
                pos_v, pos_g = [iv.index(s) for s in shared], [ig.index(s) for s in shared]
                G0, Yv0, Yg0 = Gv[pos_v], Yv[pos_v], Yg[pos_g]
                stack = np.concatenate([G0, Yv0, Yg0], axis=1)
                f = lambda G_, S: macro_f1(S[:, :5], S[:, 5:10]) - macro_f1(S[:, :5], S[:, 10:])  # noqa: E731
                d, lo, hi = boot_stat(stack, stack, f)
                verdict = ("El intervalo no contiene 0: la diferencia es consistente con este tamaño de muestra." if lo > 0 or hi < 0
                           else "El intervalo contiene 0: la diferencia no es concluyente con este tamaño de muestra.")
                md += [f"**Diferencia de macro-F1 (vectorial − grafo)** sobre {len(shared)} personas comunes: {d:+.3f} [IC95 {lo:+.3f}, {hi:+.3f}]. {verdict}", ""]

    # balance de clases
    bal_files = {"Essays original (essays/essays.csv)": root / "essays" / "essays.csv",
                 "Essays aumentado (essays_augmented.csv)": root / "essays_augmented.csv",
                 "Essays aumentado (essays/essays_augmented.csv)": root / "essays" / "essays_augmented.csv",
                 "RAG sin memoria (RAGs.csv)": root / "RAGs.csv", "RAG + memoria vectorial (RAGv.csv)": root / "RAGv.csv",
                 "RAG + memoria de grafo (RAGg.csv)": root / "RAGg.csv"}
    brows = []
    for name, p in bal_files.items():
        if not p.is_file():
            continue
        d = read_csv_any(p)
        row = {"Dataset": name, "N": len(d), "Textos únicos": int(d["TEXT"].astype(str).nunique())}
        vals = []
        for t in TRAITS:
            col = d[t]
            pos = (col.astype(float) > 0.5).mean() if pd.api.types.is_numeric_dtype(col) else col.map(norm_label).mean()
            row[t[1:]] = pos
            vals.append(min(pos, 1 - pos))
        row["Clase minoritaria media"] = float(np.mean(vals))
        brows.append(row)
    if brows:
        bal = pd.DataFrame(brows)
        bal.to_csv(out / "labels_class_balance.csv", index=False)
        md += ["**Balance de clases y calidad de los datos** (proporción de etiqueta positiva por rasgo; 0.5 = balanceado). "
               "«Textos únicos» < N indica filas duplicadas.", "", md_table(bal), ""]
    md += ["> Limitaciones: (i) n≈50 → los intervalos son amplios; (ii) las personas son LLM, no personas reales, y la etiqueta la pone otro LLM (GPT-4): "
           "un acuerdo bajo puede deberse al pipeline de recolección, a que la persona simulada no refleja bien su código o al etiquetador. "
           "Para aislar el efecto del RAG haría falta un control: etiquetar con GPT-4 un texto de la misma persona *sin* diálogo RAG."]
    return "\n".join(md)


# ---------------------------------------------------------------------------
# Recuperación y reranking
# ---------------------------------------------------------------------------
def build_queries(root: Path, max_queries: int | None, seed: int, wrap: bool) -> list[dict]:
    """Una consulta por turno generado: el último mensaje del cliente que precede al turno."""
    qs = []
    for k, c in enumerate(load_conversations(root)):
        last = None
        for t in c["turns"]:
            if t["role"] == "client":
                last = t["text"]
            elif not t.get("greeting") and last:
                text = f"Client: {last}\nPsychotherapist question: " if wrap else last
                qs.append({"qid": f"{k}-{len(qs)}", "cond": c["cond"], "persona": c["persona"], "lang": detect_lang(last), "query": text, "generated": t["text"]})
    if max_queries and len(qs) > max_queries:
        rng = np.random.default_rng(seed)
        qs = [qs[i] for i in sorted(rng.choice(len(qs), max_queries, replace=False))]
    return qs


def run_dump_pipeline(queries, search_fn, out_path: Path):
    """search_fn(query) -> (dense, rerank): dos listas de {id, score, text, source};
    `dense` es el top-k por similitud y `rerank` esos mismos k candidatos reordenados."""
    with open(out_path, "w", encoding="utf-8") as f:
        for q in queries:
            dense, rer = search_fn(q["query"])
            f.write(json.dumps({**q, "dense": dense, "rerank": rer}, ensure_ascii=False) + "\n")


def _trunc_words(text: str, n: int, tail: bool = False) -> str:
    w = text.split()
    return " ".join(w[-n:] if tail else w[:n])


def hope_pairs(docs_dir: Path, recursive: bool = False, max_words: int = 120) -> list[dict]:
    """Pares (turno del cliente → respuesta del terapeuta) de las transcripciones HOPE (columnas Type y Utterance).
    Une turnos consecutivos del mismo hablante. Es la unidad natural de este RAG: dado lo que dijo el cliente,
    ¿qué suele responder un terapeuta?"""
    docs_dir = Path(docs_dir)
    files = sorted(docs_dir.rglob("*.csv") if recursive else docs_dir.glob("*.csv"))
    pairs = []
    for f in files:
        try:
            d = read_csv_any(f)
        except Exception:
            continue
        if not {"Type", "Utterance"} <= set(d.columns):
            continue
        blocks: list[list] = []
        for typ, utt in zip(d["Type"], d["Utterance"]):
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
                pairs.append({"file": f.name, "client": _trunc_words(" ".join(u1), max_words, tail=True),
                              "therapist": _trunc_words(" ".join(u2), max_words)})
    return pairs


def run_corpus_check(docs_dir: Path, out: Path) -> str:
    """Describe el corpus tal como lo ven los scripts de la tesis y comprueba el filtro 'psychotherapist:' de memoryChat*.py."""
    docs_dir = Path(docs_dir)
    files = sorted(p for p in docs_dir.rglob("*") if p.suffix.lower() in (".csv", ".txt", ".md"))
    n_filter = 0
    n_turns = n_t = 0
    acts: dict[str, int] = {}
    for f in files:
        raw = f.read_text(encoding="utf-8", errors="replace")
        if "psychotherapist:" in raw:  # la comprobación exacta (sensible a mayúsculas) de memoryChat*.py
            n_filter += 1
        if f.suffix.lower() != ".csv":
            continue
        try:
            d = read_csv_any(f)
        except Exception:
            continue
        if "Type" not in d.columns:
            continue
        ty = d["Type"].astype(str).str.strip().str.upper()
        n_turns += int(ty.isin(["T", "P"]).sum())
        n_t += int((ty == "T").sum())
        col = next((c for c in ("Dialog_Act", "Dialogue_Act") if c in d.columns), None)
        if col:
            for a in d.loc[ty == "T", col].dropna().astype(str):
                for tok in a.split(","):
                    acts[tok.strip()] = acts.get(tok.strip(), 0) + 1
    pairs = hope_pairs(docs_dir)
    md = ["# Diagnóstico del corpus HOPE", "",
          f"- Archivos leídos: **{len(files)}** en `{docs_dir}`",
          f"- Turnos totales (T+P): **{n_turns}**; del terapeuta (T): **{n_t}** ({n_t / max(n_turns, 1):.1%})",
          f"- Pares cliente→terapeuta: **{len(pairs)}**",
          f"- Archivos que contienen el texto `psychotherapist:` (filtro de `memoryChatPinecone.py` y `memoryChatGraph.py`; sensible a mayúsculas, igual que ellos): **{n_filter} de {len(files)}**", ""]
    if acts:
        top = sorted(acts.items(), key=lambda kv: -kv[1])[:10]
        md += ["Actos de diálogo del terapeuta más frecuentes: " + ", ".join(f"`{a}` {c}" for a, c in top), ""]
    if files and n_filter == 0:
        md += ["> ⚠️ Ningún documento contiene `psychotherapist:`. Con este corpus, `retrieve_relevant_documents()` de `memoryChat*.py` "
               "devuelve una lista vacía en todos los turnos: el contexto HOPE nunca se añade al prompt y la única «memoria» activa sería el historial del chat. "
               "Si tu ejecución original usó otro formato de HOPE, ignora este aviso; si no, hay que corregir el filtro y volver a generar los diálogos."]
    text = "\n".join(md)
    (out / "corpus_check.md").write_text(text, encoding="utf-8")
    return text


def make_embedding(embed_model: str, embedder: str = "fastembed", query_prefix: str = "", doc_prefix: str = ""):
    """`fastembed` = lo que usa la tesis (RAGv4/RAGAnvil); `st` = sentence-transformers, que permite probar
    otros modelos (p. ej. multilingües como BAAI/bge-m3 o intfloat/multilingual-e5-small con sus prefijos query:/passage:)."""
    if embedder == "fastembed":
        from llama_index.embeddings.fastembed import FastEmbedEmbedding
        return FastEmbedEmbedding(model_name=embed_model)
    from llama_index.core.embeddings import BaseEmbedding
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(embed_model)

    class STEmbedding(BaseEmbedding):
        def _get_query_embedding(self, query):
            return st.encode(query_prefix + query, normalize_embeddings=True).tolist()

        def _get_text_embedding(self, text):
            return st.encode(doc_prefix + text, normalize_embeddings=True).tolist()

        def _get_text_embeddings(self, texts):
            return st.encode([doc_prefix + t for t in texts], normalize_embeddings=True, batch_size=32).tolist()

        async def _aget_query_embedding(self, query):
            return self._get_query_embedding(query)

    return STEmbedding(model_name=embed_model)


def build_llamaindex_pipeline(docs_dir: Path, embed_model: str, rerank_model: str, top_k: int, chunk_size: int, recursive: bool,
                              embedder: str = "fastembed", query_prefix: str = "", doc_prefix: str = "", corpus: str = "raw"):
    """Misma configuración que RAGv4.py: bge-small, chunk 512, top-10 y cross-encoder."""
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.postprocessor import SentenceTransformerRerank

    if corpus == "pairs":
        from llama_index.core import Document
        docs = [Document(text=f"Client: {p['client']}\nPsychotherapist: {p['therapist']}", metadata={"file_name": p["file"]},
                         excluded_embed_metadata_keys=["file_name"], excluded_llm_metadata_keys=["file_name"])
                for p in hope_pairs(docs_dir, recursive)]
        if not docs:
            raise SystemExit(f"No se encontraron pares cliente→terapeuta en {docs_dir} (se esperan CSV con columnas Type y Utterance).")
    else:
        docs = SimpleDirectoryReader(str(docs_dir), recursive=recursive).load_data()
    Settings.embed_model = make_embedding(embed_model, embedder, query_prefix, doc_prefix)
    Settings.chunk_size = chunk_size
    index = VectorStoreIndex.from_documents(docs)  # búsqueda exacta por coseno, equivalente al Qdrant en memoria de la tesis
    retriever = index.as_retriever(similarity_top_k=top_k)
    reranker = SentenceTransformerRerank(model=rerank_model, top_n=top_k)  # se conserva el orden completo; el pipeline real usa los 3 primeros

    def pack(n):
        return {"id": n.node.node_id, "score": float(n.score or 0.0), "text": n.node.get_content(), "source": str(n.node.metadata.get("file_name", ""))}

    def search(q):
        nodes = retriever.retrieve(q)
        dense = [pack(n) for n in nodes]  # antes del reranker: este modifica n.score en sitio
        rer = [pack(n) for n in reranker.postprocess_nodes(list(nodes), query_str=q)]
        return dense, rer

    return search, len(docs)


def ollama_generate(host: str, model: str, prompt: str, timeout: int = 180) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json", "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(f"{host.rstrip('/')}/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"]


JUDGE_PROMPT = """You are evaluating the retrieval step of a RAG system that helps an AI psychotherapist write its next question.

CLIENT MESSAGE:
{query}

RETRIEVED EXCERPT (from counselling session transcripts):
{text}

Is the excerpt relevant? It is relevant if it shows how a therapist responds in a comparable situation, topic or emotional state, so it could guide the next therapist question for this client message. The client message may be in Spanish and the excerpt in English: judge meaning, not language.
Answer only with JSON: {{"relevant": 1}} or {{"relevant": 0}}."""


def parse_judgement(resp: str):
    try:
        v = json.loads(resp).get("relevant")
        if v in (0, 1, "0", "1"):
            return int(v)
    except (ValueError, AttributeError):
        pass
    m = re.search(r'"?relevant"?\s*[:=]\s*"?([01])\b', resp)
    return int(m.group(1)) if m else None


def run_judge(dump: Path, out_path: Path, host: str, model: str, judge_fn=None):
    if judge_fn is None:
        def judge_fn(prompt):
            try:
                return ollama_generate(host, model, prompt)
            except OSError as e:  # URLError, timeouts, conexión rechazada
                raise SystemExit(f"No se pudo consultar Ollama en {host} con el modelo '{model}': {e}\n"
                                 f"¿Está corriendo `ollama serve` y descargaste el modelo (`ollama pull {model}`)?")
    done = set()
    if out_path.is_file():
        done = {json.loads(l)["qid"] for l in open(out_path, encoding="utf-8") if l.strip()}
    recs = [json.loads(l) for l in open(dump, encoding="utf-8") if l.strip()]
    todo = [r for r in recs if r["qid"] not in done]
    print(f"  consultas por juzgar: {len(todo)} (ya hechas: {len(done)})")
    with open(out_path, "a", encoding="utf-8") as f:
        for n, r in enumerate(todo, 1):
            for c in r["dense"]:
                prompt = JUDGE_PROMPT.format(query=r["query"].strip(), text=c["text"][:1500])
                c["rel"] = parse_judgement(judge_fn(prompt))
            r["judge"] = model
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"  {n}/{len(todo)}")


def _dcg(rels):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


_HOPE_T = re.compile(r"(?:^|[\n,])\s*T\s*,")  # fila de HOPE con Type == T (CSV crudo o texto de PandasCSVReader)


def _therapist(text: str) -> bool:
    """¿El fragmento contiene algún turno del terapeuta? Acepta el formato «Psychotherapist: ...» (el que filtran
    memoryChat*.py) y las filas crudas de HOPE (columna Type == T)."""
    t = text.lower()
    return "psychotherapist:" in t or "therapist:" in t or bool(_HOPE_T.search(text))


def retrieval_query_metrics(rec: dict, vec) -> dict:
    d, r = rec["dense"], rec["rerank"]
    d3, r3 = d[:3], r[:3]
    m = {
        "top1_changed": float(d[0]["id"] != r[0]["id"]),
        "overlap3": len({c["id"] for c in d3} & {c["id"] for c in r3}) / 3,
        "top1_dense_score": d[0]["score"],
        "therapist_rate3_dense": np.mean([_therapist(c["text"]) for c in d3]),
        "therapist_rate3_rerank": np.mean([_therapist(c["text"]) for c in r3]),
    }
    for tag, lst in (("dense", d3), ("rerank", r3)):
        V = vec.transform([c["text"] for c in lst])
        S = (V @ V.T).toarray()
        m[f"redundancy3_{tag}"] = S[np.triu_indices(len(lst), 1)].mean() if len(lst) > 1 else float("nan")
    rel = {c["id"]: c.get("rel") for c in d}
    if all(v is not None for v in rel.values()) and rel:
        pool = list(rel.values())
        m["pool_hit"] = float(any(pool))
        m["p10_dense"] = float(np.mean(pool))
        for tag, lst in (("dense", d3), ("rerank", r3)):
            rl = [rel[c["id"]] for c in lst]
            first = next((i for i, v in enumerate(rl) if v), None)
            m[f"p3_{tag}"] = float(np.mean(rl))
            m[f"hit3_{tag}"] = float(any(rl))
            m[f"mrr3_{tag}"] = 1 / (first + 1) if first is not None else 0.0
            ideal = _dcg(sorted(pool, reverse=True)[:3])
            m[f"ndcg3_{tag}"] = _dcg(rl) / ideal if ideal > 0 else float("nan")
    return m


def run_retrieval_score(path: Path, out: Path) -> str:
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not recs:
        raise SystemExit(f"{path} está vacío")
    vec = make_vectorizer([c["text"] for r in recs for c in r["dense"]])
    per = pd.DataFrame([{"qid": r["qid"], "lang": r.get("lang", "unk"), "cond": r.get("cond", ""), **retrieval_query_metrics(r, vec)} for r in recs])
    per.to_csv(out / "retrieval_per_query.csv", index=False)
    judged = "p3_dense" in per and per["p3_dense"].notna().any()

    rows = []
    for g in ["all", *sorted(per.lang.unique())]:
        sub = per if g == "all" else per[per.lang == g]
        row = {"Idioma de la consulta": g, "Consultas": len(sub)}

        def ms(col, s=sub):
            mean, lo, hi = boot_mean_ci(s[col].values)
            return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"

        def delta(a, b, s=sub):
            x = (s[b] - s[a]).dropna().values
            mean, lo, hi = boot_mean_ci(x)
            return f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]" if len(x) else "—"

        row["Similitud top-1 densa"] = ms("top1_dense_score")
        row["El reranker cambia el top-1"] = ms("top1_changed")
        row["Solape top-3 denso/rerank"] = ms("overlap3")
        row["% fragmentos con turno de terapeuta (denso→rerank)"] = f"{sub['therapist_rate3_dense'].mean():.3f} → {sub['therapist_rate3_rerank'].mean():.3f}"
        row["Redundancia top-3 (denso→rerank)"] = f"{sub['redundancy3_dense'].mean():.3f} → {sub['redundancy3_rerank'].mean():.3f}"
        if judged:
            row["Consultas con juicio completo"] = int(sub["p3_dense"].notna().sum())
            row["Hit@10 (el pool contiene algo relevante)"] = ms("pool_hit")
            row["Precision@10 denso (≈ azar dentro del pool)"] = ms("p10_dense")
            for name, c in (("Precision@3", "p3"), ("Hit@3", "hit3"), ("MRR@3", "mrr3"), ("nDCG@3", "ndcg3")):
                row[f"{name} denso"] = ms(f"{c}_dense")
                row[f"{name} rerank"] = ms(f"{c}_rerank")
                row[f"Δ {name} (rerank − denso)"] = delta(f"{c}_dense", f"{c}_rerank")
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / "retrieval_metrics.csv", index=False)

    md = ["## 3. Recuperación y reranking", "",
          f"Fuente: `{path.name}` ({len(recs)} consultas). Pipeline: bge-small-en-v1.5 → top-10 → cross-encoder → top-3 (como `RAGv4.py`). "
          "Cada celda es media [IC 95 % bootstrap sobre consultas].", ""]
    for _, r in res.iterrows():
        md += [f"**Consultas: {r.iloc[0]}** (n={r.iloc[1]})", "", md_table(pd.DataFrame({"Métrica": r.index[2:], "Valor": r.values[2:]})), ""]
    if not judged:
        md += ["_Sin etiquetas de relevancia: ejecuta `retrieval-judge` y vuelve a puntuar para obtener Precision@k, Hit@k, MRR y nDCG._", ""]
    md += ["> Notas: nDCG se calcula respecto al pool de 10 candidatos (el reranker solo puede reordenar lo que recuperó el modelo denso). "
           "El juez LLM comparte familia con el generador, lo que puede sesgar la relevancia; conviene validar a mano una muestra (≈50 pares)."]
    return "\n".join(md)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def write_md(out: Path, name: str, text: str):
    (out / name).write_text(text + "\n", encoding="utf-8")
    print(f"  → {out / name}")


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=ROOT, help="carpeta con los logs y CSV (por defecto, la del script)")
    common.add_argument("--out", type=Path, default=OUT_DIR, help="carpeta de resultados")
    ap = argparse.ArgumentParser(description="Evaluación de la parte RAG del proyecto.", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("dialogue", "labels", "all"):
        sub.add_parser(name, parents=[common])
    d = sub.add_parser("retrieval-dump", parents=[common])
    d.add_argument("--docs", type=Path, required=True, help="carpeta del corpus HOPE (la misma que usan RAGv4.py/RAGAnvil.py)")
    d.add_argument("--max-queries", type=int, default=150)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--wrap", action="store_true", help='usar el formato "Client: ...\\nPsychotherapist question:" de RAGAnvil.py')
    d.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    d.add_argument("--embedder", choices=["fastembed", "st"], default="fastembed",
                   help="fastembed = como la tesis; st = sentence-transformers (para probar modelos multilingües)")
    d.add_argument("--query-prefix", default="", help='p. ej. "query: " para modelos e5')
    d.add_argument("--doc-prefix", default="", help='p. ej. "passage: " para modelos e5')
    d.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-2-v2")
    d.add_argument("--top-k", type=int, default=10)
    d.add_argument("--chunk-size", type=int, default=512)
    d.add_argument("--recursive", action="store_true")
    d.add_argument("--corpus", choices=["raw", "pairs"], default="raw",
                   help="raw = como la tesis (CSV completos troceados); pairs = un documento por par cliente→respuesta del terapeuta")
    cc = sub.add_parser("corpus-check", parents=[common], help="diagnóstico del corpus HOPE y del filtro 'psychotherapist:'")
    cc.add_argument("--docs", type=Path, required=True)
    j = sub.add_parser("retrieval-judge", parents=[common])
    j.add_argument("--model", default="llama3")
    j.add_argument("--host", default="http://localhost:11434")
    s = sub.add_parser("retrieval-score", parents=[common])
    s.add_argument("--file", type=Path, default=None, help="JSONL (por defecto el juzgado si existe, si no el volcado)")
    args = ap.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    dump, judged = out / "retrieval_dump.jsonl", out / "retrieval_judged.jsonl"

    if args.cmd in ("dialogue", "all"):
        print("[dialogue]")
        txt_d = run_dialogue(args.root, out)
        write_md(out, "report_dialogue.md", txt_d)
    if args.cmd in ("labels", "all"):
        print("[labels]")
        txt_l = run_labels(args.root, out)
        write_md(out, "report_labels.md", txt_l)
    if args.cmd == "all":
        parts = ["# Evaluación de la parte RAG", "", txt_d, "", txt_l]
        if (out / "report_retrieval.md").is_file():
            parts += ["", (out / "report_retrieval.md").read_text(encoding="utf-8")]
        write_md(out, "rag_eval_report.md", "\n".join(parts))
    if args.cmd == "corpus-check":
        print(run_corpus_check(args.docs, out))
    if args.cmd == "retrieval-dump":
        qs = build_queries(args.root, args.max_queries, args.seed, args.wrap)
        print(f"[retrieval-dump] {len(qs)} consultas; indexando {args.docs} ...")
        t0 = time.time()
        try:
            search, n_docs = build_llamaindex_pipeline(args.docs, args.embed_model, args.rerank_model, args.top_k, args.chunk_size, args.recursive,
                                                      args.embedder, args.query_prefix, args.doc_prefix, args.corpus)
        except ImportError as e:
            raise SystemExit(f"Falta una dependencia ({e.name}). Instala: pip install llama-index-core sentence-transformers  (+ llama-index-embeddings-fastembed si usas --embedder fastembed)")
        print(f"  {n_docs} documentos indexados en {time.time() - t0:.0f}s")
        (out / "retrieval_config.json").write_text(json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, indent=2, ensure_ascii=False), encoding="utf-8")
        run_dump_pipeline(qs, search, dump)
        print(f"  → {dump}")
    if args.cmd == "retrieval-judge":
        if not dump.is_file():
            raise SystemExit("Primero ejecuta retrieval-dump.")
        print(f"[retrieval-judge] modelo {args.model} en {args.host}")
        run_judge(dump, judged, args.host, args.model)
        print(f"  → {judged}")
    if args.cmd == "retrieval-score":
        path = args.file or (judged if judged.is_file() else dump)
        if not path.is_file():
            raise SystemExit("No hay JSONL: ejecuta retrieval-dump primero.")
        print(f"[retrieval-score] {path.name}")
        write_md(out, "report_retrieval.md", run_retrieval_score(path, out))


if __name__ == "__main__":
    main()
