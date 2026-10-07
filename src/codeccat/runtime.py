from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .features import (
    baseline_from_array,
    full_b_extra_from_array,
    hmt768_array,
    replacement_rate,
)
from .model import LinearModel, RuntimeBundle, RuntimeStack

ROUTES = ("U", "N", "RL", "RH")
TRIAD = ("utf-8", "cp1251", "gb18030")
SIG_PAIR = ("utf-8", "utf-8-sig")
GB_PAIR = ("utf-8", "gb18030")
RATE_THRESHOLD = 0.02
TOP_CANDIDATES = 3
R12_TARGETS = ("utf-8", "big5", "euc-kr", "cp1251", "iso-8859-1")
R12_OTHER = "__other__"


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
class ByteSignals:
    route: str
    has_utf8_bom: bool
    strict_utf8: bool
    ascii_only: bool
    high_byte_ratio: float
    nul_ratio: float
    sample_len_4k: int


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


def _signals(data: bytes) -> ByteSignals:
    sample = data[:4096]
    array = np.frombuffer(sample, dtype=np.uint8)
    try:
        sample.decode("utf-8", "strict")
        strict = True
    except UnicodeDecodeError:
        strict = False

    bom = data.startswith(b"\xef\xbb\xbf")
    if len(array):
        high_count = int(np.count_nonzero(array >= 128))
        nul_count = int(np.count_nonzero(array == 0))
        high = high_count / len(array)
        nul = nul_count / len(array)
        ascii_only = high_count == 0
    else:
        high = 0.0
        nul = 0.0
        ascii_only = True
        nul_count = 0

    if strict:
        route_name = "U"
    elif nul_count:
        route_name = "N"
    elif high <= 0.02:
        route_name = "RL"
    else:
        route_name = "RH"

    return ByteSignals(
        route=route_name,
        has_utf8_bom=bom,
        strict_utf8=strict,
        ascii_only=ascii_only,
        high_byte_ratio=high,
        nul_ratio=nul,
        sample_len_4k=len(array),
    )


def _raw(model: LinearModel, vector: np.ndarray) -> np.ndarray:
    x = np.asarray(vector, dtype=np.float64)
    z = np.dot(model.weight, x) + model.bias
    z = np.asarray(z, dtype=np.float64)
    if len(model.classes) == 2 and z.shape == (1,):
        score = float(z[0])
        return np.asarray([-score, score], dtype=np.float64)
    return z


def _context(data: bytes, signals: ByteSignals, model: LinearModel, vector: np.ndarray) -> InferenceContext:
    raw = _raw(model, vector)
    order = raw.argsort()[::-1]
    return InferenceContext(
        data=data,
        route=signals.route,
        rank=tuple(np.asarray(model.classes, dtype=object)[order]),
        sorted_scores=raw[order],
        raw_scores=raw,
        classes=model.classes,
        has_utf8_bom=signals.has_utf8_bom,
        strict_utf8=signals.strict_utf8,
        ascii_only=signals.ascii_only,
        high_byte_ratio=signals.high_byte_ratio,
        nul_ratio=signals.nul_ratio,
        sample_len_4k=signals.sample_len_4k,
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


class _StackRuntime:
    def __init__(self, stack: RuntimeStack, classes: tuple[str, ...], families: tuple[str, ...]):
        self.stack = stack
        self.classes = classes
        self.families = families
        self.class_idx = {c: i for i, c in enumerate(classes)}
        self.fam_idx = {c: i for i, c in enumerate(families)}
        self.route_idx = {r: i for i, r in enumerate(ROUTES)}

        self._cal_static: dict[tuple[str, str], float] = {}
        self._cal_coef = 0.0
        self._cal_pos = (0.0, 0.0, 0.0)
        if stack.calibrator is not None:
            vector = stack.calibrator.weight[0]
            for route_name in ROUTES:
                route_term = vector[7 + self.route_idx[route_name]]
                for candidate in classes:
                    value = float(route_term)
                    ci = self.class_idx.get(candidate)
                    if ci is not None:
                        value += vector[11 + ci]
                    fi = self.fam_idx.get(family(candidate))
                    if fi is not None:
                        value += vector[11 + len(classes) + fi]
                    self._cal_static[(route_name, candidate)] = float(value)
            self._cal_coef = float(vector[0] + vector[1] + vector[2])
            self._cal_pos = (0.0, float(vector[3]), float(2.0 * vector[3]))

    def _choose_cal(self, ctx: InferenceContext) -> list[str]:
        model = self.stack.calibrator
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
        winner = int(np.argmax(z))
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

    def _triad_logits(self, ctx: InferenceContext):
        model = self.stack.triad
        if model is None:
            return None
        values = [_score(ctx, c) for c in TRIAD]
        second = sorted(values)[1]
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
        return model.classes, tuple(logits)

    def _gate(self, ctx: InferenceContext, baseline: list[str]) -> list[str]:
        if self.stack.triad is None or not baseline or baseline[0] not in TRIAD:
            return baseline
        top3 = ctx.rank[:3]
        if sum(c in TRIAD for c in top3) < 2:
            return baseline
        packed = self._triad_logits(ctx)
        assert packed is not None
        classes, logits = packed
        idx = int(np.argmax(logits))
        maximum = float(logits[idx])
        others = [float(v) for i, v in enumerate(logits) if i != idx]
        if sum(math.exp(v - maximum) for v in others) > 1.0:
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

    def rank_context(self, ctx: InferenceContext) -> list[str]:
        hybrid = self._hybrid(ctx)
        gated = self._gate(ctx, hybrid)
        sig = self._pair(self.stack.utf8_sig, SIG_PAIR, ctx, gated)
        out = self._pair(self.stack.utf8_gb, GB_PAIR, ctx, sig)
        if (
            out and sig and out[0] != sig[0]
            and sig[0] == "gb18030" and out[0] == "utf-8"
            and replacement_rate(ctx.data) > RATE_THRESHOLD
        ):
            out = sig
        return out


def _decode_metrics(data: bytes, enc: str) -> tuple[float, float]:
    sample = data[:4096]
    try:
        text = sample.decode(enc, "strict")
        return 1.0, 0.0
    except UnicodeDecodeError:
        text = sample.decode(enc, "replace")
        return 0.0, text.count("\ufffd") / max(1, len(text))


def _specialist_vector(ctx: InferenceContext) -> np.ndarray:
    vals = [_score(ctx, c) for c in R12_TARGETS]
    sorted_vals = sorted(vals, reverse=True)
    target_margin = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else 0.0

    positions = []
    rank = list(ctx.rank)
    for target in R12_TARGETS:
        try:
            positions.append(float(rank.index(target) + 1))
        except ValueError:
            positions.append(99.0)

    u_valid, u_repl = _decode_metrics(ctx.data, "utf-8")
    b_valid, b_repl = _decode_metrics(ctx.data, "big5")
    k_valid, k_repl = _decode_metrics(ctx.data, "euc-kr")

    return np.asarray(
        vals
        + positions
        + [
            target_margin,
            u_valid, u_repl,
            b_valid, b_repl,
            k_valid, k_repl,
            ctx.high_byte_ratio,
            float(ctx.sample_len_4k),
        ],
        dtype=np.float32,
    )


def _softmax(raw: np.ndarray) -> np.ndarray:
    z = np.asarray(raw, dtype=np.float64)
    z = z - float(np.max(z))
    e = np.exp(z)
    return e / np.sum(e)


class Runtime:
    def __init__(self, bundle: RuntimeBundle):
        self.bundle = bundle
        self.baseline = _StackRuntime(bundle.baseline, bundle.classes, bundle.families)
        self.full_b = (
            _StackRuntime(bundle.full_b, bundle.classes, bundle.families)
            if bundle.full_b is not None
            else None
        )

    def _apply_rl_specialist(self, ctx: InferenceContext, baseline: list[str]) -> list[str]:
        model = self.bundle.rl_specialist
        if model is None or ctx.route != "RL" or not baseline:
            return baseline
        raw = _raw(model, _specialist_vector(ctx))
        probs = _softmax(raw)
        idx = int(np.argmax(probs))
        pred = model.classes[idx]
        if pred == R12_OTHER or float(probs[idx]) < self.bundle.r12_threshold:
            return baseline
        if pred not in ctx.rank:
            return baseline
        out = list(baseline)
        if pred in out:
            out.remove(pred)
        out.insert(0, pred)
        return out

    def rank(self, data: bytes) -> tuple[str, ...]:
        signals = _signals(data)
        array = hmt768_array(data)
        baseline_vector = baseline_from_array(array)

        bmodel = self.bundle.baseline.route_models[signals.route]
        bctx = _context(data, signals, bmodel, baseline_vector)
        baseline = self.baseline.rank_context(bctx)

        threshold = self.bundle.gate_thresholds.get(signals.route)
        margin = (
            float(bctx.sorted_scores[0] - bctx.sorted_scores[1])
            if len(bctx.sorted_scores) >= 2
            else float("inf")
        )

        if (
            self.full_b is not None
            and threshold is not None
            and signals.route in ("U", "RH")
            and margin <= threshold
        ):
            extra = full_b_extra_from_array(array, signals.route)
            full_vector = np.concatenate([baseline_vector, extra])
            fmodel = self.bundle.full_b.route_models[signals.route]  # type: ignore[union-attr]
            fctx = _context(data, signals, fmodel, full_vector)
            return tuple(self.full_b.rank_context(fctx))

        return tuple(self._apply_rl_specialist(bctx, baseline))
