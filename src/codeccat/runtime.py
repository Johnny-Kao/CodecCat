from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .features import byte_signals, feature_vector, replacement_rate, route
from .model import LinearModel, RuntimeBundle

ROUTES = ("U", "N", "RL", "RH")
TRIAD = ("utf-8", "cp1251", "gb18030")
SIG_PAIR = ("utf-8", "utf-8-sig")
GB_PAIR = ("utf-8", "gb18030")
RATE_THRESHOLD = 0.02
TOP_CANDIDATES = 3


def family(label: str | None) -> str:
    if label is None:
        return "other"
    if label.startswith("utf-"):
        return "utf"
    if label.startswith("cp12"):
        return "singlebyte-win"
    if label.startswith("iso-8859") or label.startswith("iso8859"):
        return "singlebyte-iso"
    if label in ("shift-jis", "euc-jp", "euc-kr", "big5", "gb18030", "gbk", "cp949"):
        return "cjk"
    if label.startswith("koi8"):
        return "koi8"
    if label == "ascii":
        return "ascii"
    return "other"


@dataclass(frozen=True)
class InferenceContext:
    data: bytes
    route: str
    rank: tuple[str, ...]
    sorted_scores: np.ndarray
    raw_scores: np.ndarray
    classes: tuple[str, ...]
    has_utf8_bom: bool
    strict_utf8: bool
    ascii_only: bool
    high_byte_ratio: float
    nul_ratio: float
    sample_len_4k: int


def _raw(model: LinearModel, vector: np.ndarray) -> np.ndarray:
    x = np.asarray(vector, dtype=np.float64)
    z = np.dot(model.weight, x) + model.bias
    z = np.asarray(z, dtype=np.float64)
    if len(model.classes) == 2 and z.shape == (1,):
        score = float(z[0])
        return np.asarray([-score, score], dtype=np.float64)
    return z


def build_context(data: bytes, model: LinearModel) -> InferenceContext:
    vector = feature_vector(data)
    raw = _raw(model, vector)
    order = raw.argsort()[::-1]
    bom, strict, ascii_only, high, nul, length = byte_signals(data)
    return InferenceContext(
        data=data,
        route=route(data),
        rank=tuple(np.asarray(model.classes, dtype=object)[order]),
        sorted_scores=raw[order],
        raw_scores=raw,
        classes=model.classes,
        has_utf8_bom=bom,
        strict_utf8=strict,
        ascii_only=ascii_only,
        high_byte_ratio=high,
        nul_ratio=nul,
        sample_len_4k=length,
    )


def _score(ctx: InferenceContext, label: str) -> float:
    try:
        return float(ctx.raw_scores[ctx.classes.index(label)])
    except ValueError:
        return -20.0


def _rule_rerank(ctx: InferenceContext) -> list[str]:
    ranked = list(ctx.rank)
    if not ranked:
        return ranked

    if ctx.has_utf8_bom and "utf-8-sig" in ranked:
        ranked.remove("utf-8-sig")
        ranked.insert(0, "utf-8-sig")
        return ranked

    if ctx.strict_utf8 and "utf-8" in ranked:
        if ranked[0] != "utf-8" and family(ranked[0]) in (
            "utf", "cjk", "singlebyte-win", "singlebyte-iso", "ascii"
        ):
            ranked.remove("utf-8")
            ranked.insert(0, "utf-8")
        return ranked

    if ctx.ascii_only and "ascii" in ranked:
        ranked.remove("ascii")
        ranked.insert(0, "ascii")
        return ranked

    top3 = ranked[:3]
    if ranked[0] in ("iso-8859-9", "cp1257") and "iso-8859-1" in top3:
        ranked.remove("iso-8859-1")
        ranked.insert(0, "iso-8859-1")
    return ranked


class Runtime:
    def __init__(self, bundle: RuntimeBundle):
        self.bundle = bundle
        self.class_idx = {c: i for i, c in enumerate(bundle.classes)}
        self.fam_idx = {c: i for i, c in enumerate(bundle.families)}
        self.route_idx = {r: i for i, r in enumerate(ROUTES)}

        self._cal_static: dict[tuple[str, str], float] = {}
        self._cal_coef = 0.0
        self._cal_pos = (0.0, 0.0, 0.0)
        if bundle.calibrator is not None:
            vector = bundle.calibrator.weight[0]
            for route_name in ROUTES:
                route_term = vector[7 + self.route_idx[route_name]]
                for candidate in bundle.classes:
                    value = float(route_term)
                    ci = self.class_idx.get(candidate)
                    if ci is not None:
                        value += vector[11 + ci]
                    fi = self.fam_idx.get(family(candidate))
                    if fi is not None:
                        value += vector[11 + len(bundle.classes) + fi]
                    self._cal_static[(route_name, candidate)] = float(value)
            self._cal_coef = float(vector[0] + vector[1] + vector[2])
            self._cal_pos = (0.0, float(vector[3]), float(2.0 * vector[3]))

    def _choose_cal(self, ctx: InferenceContext) -> list[str]:
        model = self.bundle.calibrator
        if model is None or len(ctx.rank) < 3:
            return list(ctx.rank)

        vector = model.weight[0]
        common = float(model.bias[0])
        top = float(ctx.sorted_scores[0])
        second = float(ctx.sorted_scores[1])
        common -= vector[1] * top
        common -= vector[2] * second
        common += vector[4] * float(ctx.has_utf8_bom)
        common += vector[5] * float(ctx.strict_utf8)
        common += vector[6] * float(ctx.ascii_only)

        cand = ctx.rank[:TOP_CANDIDATES]
        z = (
            common + self._cal_coef * float(ctx.sorted_scores[0]) + self._cal_pos[0] + self._cal_static[(ctx.route, cand[0])],
            common + self._cal_coef * float(ctx.sorted_scores[1]) + self._cal_pos[1] + self._cal_static[(ctx.route, cand[1])],
            common + self._cal_coef * float(ctx.sorted_scores[2]) + self._cal_pos[2] + self._cal_static[(ctx.route, cand[2])],
        )
        winner = 0
        best = z[0]
        if z[1] > best:
            winner, best = 1, z[1]
        if z[2] > best:
            winner = 2
        if winner == 0:
            return list(ctx.rank)
        out = list(ctx.rank)
        chosen = cand[winner]
        out.remove(chosen)
        out.insert(0, chosen)
        return out

    def _hybrid(self, ctx: InferenceContext) -> list[str]:
        rule = _rule_rerank(ctx)
        if rule and ctx.rank and rule[0] != ctx.rank[0]:
            return rule
        return self._choose_cal(ctx)

    def _triad_logits(self, ctx: InferenceContext) -> tuple[tuple[str, ...], tuple[float, float, float]] | None:
        model = self.bundle.triad
        if model is None:
            return None
        values = [_score(ctx, c) for c in TRIAD]
        second = max(min(values[0], values[1]), min(max(values[0], values[1]), values[2]))
        xs = (
            values[0], values[1], values[2],
            values[0] - values[1],
            values[0] - values[2],
            values[1] - values[2],
            max(values) - second,
            float(ctx.strict_utf8),
            float(ctx.has_utf8_bom),
            float(ctx.ascii_only),
            ctx.high_byte_ratio,
            ctx.nul_ratio,
            float(ctx.sample_len_4k),
        )
        logits = [float(v) for v in model.bias]
        for j, x in enumerate(xs):
            for i in range(3):
                logits[i] += float(model.weight[i, j]) * x
        route_col = 13 + self.route_idx[ctx.route]
        for i in range(3):
            logits[i] += float(model.weight[i, route_col])
        return model.classes, (logits[0], logits[1], logits[2])

    def _gate(self, ctx: InferenceContext, baseline: list[str]) -> list[str]:
        if self.bundle.triad is None or not baseline or baseline[0] not in TRIAD:
            return baseline
        top3 = ctx.rank[:3]
        if sum(c in TRIAD for c in top3) < 2:
            return baseline
        packed = self._triad_logits(ctx)
        assert packed is not None
        classes, logits = packed
        a, b, c = logits
        if a >= b and a >= c:
            idx, maximum, other1, other2 = 0, a, b, c
        elif b >= c:
            idx, maximum, other1, other2 = 1, b, a, c
        else:
            idx, maximum, other1, other2 = 2, c, a, b
        if math.exp(other1 - maximum) + math.exp(other2 - maximum) > 1.0:
            return baseline
        pred = classes[idx]
        if pred not in top3:
            return baseline
        out = list(baseline)
        if pred in out:
            out.remove(pred)
            out.insert(0, pred)
        return out

    def _pair(self, model: LinearModel | None, pair: tuple[str, str], ctx: InferenceContext, baseline: list[str]) -> list[str]:
        if model is None or not baseline or baseline[0] not in pair:
            return baseline
        top3 = ctx.rank[:3]
        if pair[0] not in top3 or pair[1] not in top3:
            return baseline

        a = _score(ctx, pair[0])
        b = _score(ctx, pair[1])
        xs = (
            a, b, a - b, abs(a - b),
            float(ctx.strict_utf8), float(ctx.has_utf8_bom), float(ctx.ascii_only),
            ctx.high_byte_ratio, ctx.nul_ratio, float(ctx.sample_len_4k),
        )
        value = float(model.bias[0])
        for j, x in enumerate(xs):
            value += float(model.weight[0, j]) * x
        value += float(model.weight[0, 10 + self.route_idx[ctx.route]])
        pred = model.classes[1] if value > 0.0 else model.classes[0]
        if pred not in top3:
            return baseline
        out = list(baseline)
        if pred in out:
            out.remove(pred)
            out.insert(0, pred)
        return out

    def rank(self, data: bytes) -> tuple[str, ...]:
        route_name = route(data)
        ctx = build_context(data, self.bundle.route_models[route_name])
        hybrid = self._hybrid(ctx)
        gated = self._gate(ctx, hybrid)
        sig = self._pair(self.bundle.utf8_sig, SIG_PAIR, ctx, gated)
        out = self._pair(self.bundle.utf8_gb, GB_PAIR, ctx, sig)
        if (
            out and sig and out[0] != sig[0]
            and sig[0] == "gb18030" and out[0] == "utf-8"
            and replacement_rate(data) > RATE_THRESHOLD
        ):
            out = sig
        return tuple(out)
