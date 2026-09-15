"""
track_compare.py -- So sanh hai ket qua tracking (MOTChallenge 7 cot) KE CA lien ket ID
xuyen suot sequence, bang ANH XA ID MOT-MOT CO DINH cho toan sequence.

Vi sao can: bo dem ID cua ultralytics (`BaseTrack._count`) la toan cuc trong tien trinh,
nen hai lan chay cung mot tracker (vd xen ke voi tracker khac, hoac sau mot lan chay
truoc do) co the cap ID khac nhau du cau truc track y het. So box theo tung frame (bo
qua ID) hoac anh xa lai ID rieng tung frame KHONG phat hien duoc truong hop track bi
cat/noi khac nhau. Phep so o day:

    1. Ghep dong theo (frame, box) voi dung sai `tol` (px) - moi dong cua A ghep voi dung
       mot dong cua B trong cung frame (ghep tham theo khoang cach L_inf nho nhat).
    2. Tu cac cap dong ghep duoc, thu duoc tap cap (id_a, id_b).
    3. PASS khi tap cap la SONG ANH tren toan sequence: moi id_a ung voi dung mot id_b va
       nguoc lai (cung mot track cua A luon la cung mot track cua B, va khong co track
       nao cua B gop hai track cua A).

Ket qua tra ve du de ghi bao cao: so dong moi ben, so dong ghep duoc, so cap ID, so
id_a co > 1 id_b (track bi tach/noi khac nhau), so id_b co > 1 id_a, va bang anh xa.

    from track_compare import id_bijection
    r = id_bijection(df_a, df_b)          # df: cot frame,id,bb_left,bb_top,bb_width,bb_height[,conf]
    assert r.is_bijection, r.summary()
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BOX = ["bb_left", "bb_top", "bb_width", "bb_height"]


@dataclass
class BijectionResult:
    n_rows_a: int
    n_rows_b: int
    n_matched_rows: int
    n_unmatched_a: int
    n_unmatched_b: int
    n_pairs: int
    n_ids_a: int
    n_ids_b: int
    a_multi: dict = field(default_factory=dict)     # id_a -> {id_b: so dong} khi id_a ung voi > 1 id_b
    b_multi: dict = field(default_factory=dict)     # id_b -> {id_a: so dong}
    mapping: dict = field(default_factory=dict)     # id_a -> id_b (chi khi song anh)
    is_identity: bool = False

    @property
    def is_bijection(self) -> bool:
        return (self.n_unmatched_a == 0 and self.n_unmatched_b == 0
                and not self.a_multi and not self.b_multi and self.n_pairs == self.n_ids_a == self.n_ids_b)

    def summary(self) -> str:
        return (f"rows A={self.n_rows_a} B={self.n_rows_b} matched={self.n_matched_rows} "
                f"unmatched A={self.n_unmatched_a} B={self.n_unmatched_b}; ids A={self.n_ids_a} B={self.n_ids_b} "
                f"pairs={self.n_pairs}; A->nhieu B: {len(self.a_multi)}, B->nhieu A: {len(self.b_multi)}; "
                f"bijection={self.is_bijection} identity={self.is_identity}")


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "id" not in d.columns and "track_id" in d.columns:
        d = d.rename(columns={"track_id": "id"})
    if not set(BOX) <= set(d.columns):
        if {"x", "y", "w", "h"} <= set(d.columns):
            d = d.rename(columns={"x": "bb_left", "y": "bb_top", "w": "bb_width", "h": "bb_height"})
        else:
            raise ValueError(f"thieu cot box: co {list(d.columns)}")
    return d[["frame", "id"] + BOX].reset_index(drop=True)


def match_rows(df_a: pd.DataFrame, df_b: pd.DataFrame, tol: float = 1e-3) -> pd.DataFrame:
    """Ghep dong A <-> B trong cung frame theo box (L_inf <= tol). Tra ve DataFrame (frame, id_a, id_b, dist)
    cho cac cap ghep duoc; dong khong ghep duoc co id_b/id_a = -1."""
    a, b = _norm(df_a), _norm(df_b)
    out = []
    bf = {f: g for f, g in b.groupby("frame", sort=False)}
    used_b: set[int] = set()
    for f, ga in a.groupby("frame", sort=False):
        gb = bf.get(f)
        if gb is None or not len(gb):
            for r in ga.itertuples(index=False):
                out.append((int(f), int(r.id), -1, np.inf))
            continue
        A = ga[BOX].to_numpy(float); B = gb[BOX].to_numpy(float)
        D = np.abs(A[:, None, :] - B[None, :, :]).max(axis=2)          # (na, nb) L_inf
        taken = np.zeros(len(B), bool)
        # ghep tham theo khoang cach tang dan (bit-identical -> D = 0 chinh xac)
        order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
        assigned_a = np.full(len(A), -1)
        for i, j in order:
            if D[i, j] > tol:
                break
            if assigned_a[i] >= 0 or taken[j]:
                continue
            assigned_a[i] = j; taken[j] = True
        ids_b = gb["id"].to_numpy()
        for i, r in enumerate(ga.itertuples(index=False)):
            j = assigned_a[i]
            if j >= 0:
                out.append((int(f), int(r.id), int(ids_b[j]), float(D[i, j])))
                used_b.add(int(gb.index[j]))
            else:
                out.append((int(f), int(r.id), -1, np.inf))
    for idx, r in b.iterrows():
        if int(idx) not in used_b:
            out.append((int(r.frame), -1, int(r.id), np.inf))
    return pd.DataFrame(out, columns=["frame", "id_a", "id_b", "dist"])


def id_bijection(df_a: pd.DataFrame, df_b: pd.DataFrame, tol: float = 1e-3) -> BijectionResult:
    """Kiem tra hai ket qua tracking co cung cau truc track (anh xa ID 1-1 co dinh toan sequence)."""
    a, b = _norm(df_a), _norm(df_b)
    m = match_rows(a, b, tol)
    ok = m[(m.id_a >= 0) & (m.id_b >= 0)]
    pairs = ok.groupby(["id_a", "id_b"]).size().reset_index(name="n")
    a_multi = {int(k): {int(r.id_b): int(r.n) for r in g.itertuples()} for k, g in pairs.groupby("id_a") if len(g) > 1}
    b_multi = {int(k): {int(r.id_a): int(r.n) for r in g.itertuples()} for k, g in pairs.groupby("id_b") if len(g) > 1}
    res = BijectionResult(
        n_rows_a=len(a), n_rows_b=len(b), n_matched_rows=len(ok),
        n_unmatched_a=int(((m.id_a >= 0) & (m.id_b < 0)).sum()), n_unmatched_b=int(((m.id_b >= 0) & (m.id_a < 0)).sum()),
        n_pairs=len(pairs), n_ids_a=int(a.id.nunique()), n_ids_b=int(b.id.nunique()), a_multi=a_multi, b_multi=b_multi)
    if res.is_bijection:
        res.mapping = {int(r.id_a): int(r.id_b) for r in pairs.itertuples()}
        res.is_identity = all(k == v for k, v in res.mapping.items())
    return res


def read_mot(path) -> pd.DataFrame:
    """Doc file MOTChallenge 7 cot cua runner (frame,id,x,y,w,h,conf). File rong -> DataFrame rong."""
    import os
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=["frame", "id"] + BOX + ["conf"])
    return pd.read_csv(path, header=None, names=["frame", "id"] + BOX + ["conf"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="So sanh 2 file tracking MOT: song anh ID toan sequence")
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--tol", type=float, default=1e-3)
    ns = ap.parse_args()
    r = id_bijection(read_mot(ns.a), read_mot(ns.b), ns.tol)
    print(r.summary())
    raise SystemExit(0 if r.is_bijection else 1)
