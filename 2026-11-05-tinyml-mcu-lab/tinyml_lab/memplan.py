"""Tensor-arena planning: given a graph, assign byte offsets in one SRAM arena.

This is the memory-planner problem inside TFLite-Micro / TVM-micro: tensors whose lifetimes
do not overlap may share addresses. Finding the minimum arena is NP-hard (dynamic storage
allocation), so we compare several heuristics against the max-live-bytes lower bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Op:
    name: str
    inputs: list
    output: str
    macs: int = 0


@dataclass
class Graph:
    name: str
    tensor_bytes: dict       # tensor name -> bytes
    ops: list                # ordered list[Op]
    graph_inputs: list = field(default_factory=list)

    def total_macs(self) -> int:
        return sum(o.macs for o in self.ops)


@dataclass
class Buf:
    name: str
    size: int
    first: int
    last: int
    offset: int = -1


def liveness(g: Graph) -> list:
    """Lifetime [first, last] (op indices, inclusive) of each activation tensor.

    Inputs and the output of the same op are live simultaneously (no in-place ops).
    Graph inputs are live from op 0, the final output until the last op.
    """
    first, last = {}, {}
    for t in g.graph_inputs:
        first[t] = 0
        last[t] = 0
    for i, op in enumerate(g.ops):
        first.setdefault(op.output, i)
        last[op.output] = max(last.get(op.output, i), i)
        for t in op.inputs:
            last[t] = max(last.get(t, 0), i)
    final = g.ops[-1].output
    last[final] = len(g.ops) - 1
    return [Buf(t, g.tensor_bytes[t], first[t], last[t]) for t in first]


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _overlap(a: Buf, b: Buf) -> bool:
    return not (a.last < b.first or b.last < a.first)


def lower_bound(bufs, align: int = 4) -> int:
    """Max over time of the sum of live (aligned) tensor sizes: no plan can beat this."""
    n = max(b.last for b in bufs) + 1
    return max(sum(_align(b.size, align) for b in bufs if b.first <= t <= b.last) for t in range(n))


def plan_naive(bufs, align: int = 4):
    """Every tensor gets private memory (what you get with no planner)."""
    off = 0
    out = []
    for b in bufs:
        out.append(Buf(b.name, b.size, b.first, b.last, off))
        off += _align(b.size, align)
    return out, off


def _first_fit(order, align):
    placed = []
    for b in order:
        sz = _align(b.size, align)
        conflicts = sorted((p for p in placed if _overlap(p, b)), key=lambda p: p.offset)
        off = 0
        for p in conflicts:
            if off + sz <= p.offset:
                break
            off = max(off, p.offset + _align(p.size, align))
        placed.append(Buf(b.name, b.size, b.first, b.last, off))
    arena = max(p.offset + _align(p.size, align) for p in placed)
    return placed, arena


def plan_greedy_by_size(bufs, align: int = 4):
    """Largest tensors first, each at the lowest address that avoids live conflicts."""
    return _first_fit(sorted(bufs, key=lambda b: (-b.size, b.first)), align)


def plan_greedy_by_start(bufs, align: int = 4):
    """Allocate in execution order (a simple online allocator)."""
    return _first_fit(sorted(bufs, key=lambda b: (b.first, -b.size)), align)


def plan_random_search(bufs, align: int = 4, trials: int = 300, seed: int = 0):
    """Best of many random orderings fed to first-fit; seeded with the greedy plan."""
    rng = np.random.default_rng(seed)
    best, best_arena = plan_greedy_by_size(bufs, align)
    for _ in range(trials):
        order = [bufs[i] for i in rng.permutation(len(bufs))]
        p, a = _first_fit(order, align)
        if a < best_arena:
            best, best_arena = p, a
    return best, best_arena


PLANNERS = {
    "naive": plan_naive,
    "greedy_by_start": plan_greedy_by_start,
    "greedy_by_size": plan_greedy_by_size,
    "random_search": plan_random_search,
}


def validate(plan, arena: int, align: int = 4) -> bool:
    """No two simultaneously-live tensors may overlap in address space; all must fit the arena."""
    for i, a in enumerate(plan):
        if a.offset < 0 or a.offset + a.size > arena:
            return False
        for b in plan[i + 1:]:
            if _overlap(a, b):
                if not (a.offset + a.size <= b.offset or b.offset + b.size <= a.offset):
                    return False
    return True


# ------------------------------------------------------------------ example graphs
def mlp_graph(sizes, act_bytes: int = 1) -> Graph:
    tb = {"x": sizes[0] * act_bytes}
    ops = []
    prev = "x"
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        out = f"h{i + 1}"
        tb[out] = b * act_bytes
        ops.append(Op(f"fc{i + 1}", [prev], out, macs=a * b))
        prev = out
    return Graph("mlp", tb, ops, ["x"])


def _conv_macs(h, w, cin, cout, k):
    return h * w * cin * cout * k * k


def tiny_resnet_graph(hw: int = 32, ch=(8, 16, 32), act_bytes: int = 1) -> Graph:
    """Residual CNN with skip connections that keep tensors alive across several ops."""
    tb = {"img": hw * hw * 3 * act_bytes}
    ops = []
    h, c = hw, 3
    cur = "img"
    tb["stem"] = h * h * ch[0] * act_bytes
    ops.append(Op("stem", [cur], "stem", _conv_macs(h, h, c, ch[0], 3)))
    cur, c = "stem", ch[0]
    for s, cout in enumerate(ch):
        if s > 0:
            h //= 2
            name = f"down{s}"
            tb[name] = h * h * cout * act_bytes
            ops.append(Op(name, [cur], name, _conv_macs(h, h, c, cout, 3)))
            cur, c = name, cout
        a, b, r = f"b{s}_conv1", f"b{s}_conv2", f"b{s}_add"
        for n in (a, b, r):
            tb[n] = h * h * c * act_bytes
        ops.append(Op(a, [cur], a, _conv_macs(h, h, c, c, 3)))
        ops.append(Op(b, [a], b, _conv_macs(h, h, c, c, 3)))
        ops.append(Op(r, [cur, b], r, h * h * c))
        cur = r
    tb["gap"] = c * act_bytes
    tb["logits"] = 10 * act_bytes
    ops.append(Op("gap", [cur], "gap", h * h * c))
    ops.append(Op("fc", ["gap"], "logits", c * 10))
    return Graph("tiny_resnet", tb, ops, ["img"])


def unet_graph(hw: int = 32, base: int = 8, depth: int = 3, act_bytes: int = 1) -> Graph:
    """Encoder-decoder with long skips: encoder feature maps stay live until the decoder."""
    tb, ops = {}, []
    tb["img"] = hw * hw * 1 * act_bytes
    cur, c, h = "img", 1, hw
    skips = []
    for d in range(depth):
        cout = base * 2 ** d
        e = f"enc{d}"
        tb[e] = h * h * cout * act_bytes
        ops.append(Op(e, [cur], e, _conv_macs(h, h, c, cout, 3)))
        skips.append((e, h, cout))
        p = f"pool{d}"
        h //= 2
        tb[p] = h * h * cout * act_bytes
        ops.append(Op(p, [e], p, h * h * cout))
        cur, c = p, cout
    mid = "mid"
    tb[mid] = h * h * c * 2 * act_bytes
    ops.append(Op(mid, [cur], mid, _conv_macs(h, h, c, c * 2, 3)))
    cur, c = mid, c * 2
    for d in reversed(range(depth)):
        e, eh, ec = skips[d]
        up = f"up{d}"
        tb[up] = eh * eh * ec * act_bytes
        ops.append(Op(up, [cur], up, _conv_macs(eh, eh, c, ec, 1)))
        dec = f"dec{d}"
        tb[dec] = eh * eh * ec * act_bytes
        ops.append(Op(dec, [up, e], dec, _conv_macs(eh, eh, 2 * ec, ec, 3)))
        cur, c, h = dec, ec, eh
    tb["seg"] = hw * hw * act_bytes
    ops.append(Op("seg", [cur], "seg", hw * hw * c))
    return Graph("unet", tb, ops, ["img"])


def random_graph(n_ops: int = 40, seed: int = 0, max_kb: int = 48) -> Graph:
    """Adversarial synthetic DAG: random tensor sizes and random long-range reuse of old tensors.

    Unlike CNN graphs (which are nearly interval-graph friendly), this makes the heuristics
    diverge from the lower bound, exposing the NP-hard side of arena planning.
    """
    rng = np.random.default_rng(seed)
    tb = {"t0": int(rng.integers(1, max_kb)) * 1024}
    ops = []
    names = ["t0"]
    for i in range(1, n_ops + 1):
        out = f"t{i}"
        tb[out] = int(rng.integers(1, max_kb)) * 1024 + int(rng.integers(0, 1024))
        ins = [names[-1]]
        if len(names) > 3 and rng.random() < 0.5:
            ins.append(names[int(rng.integers(0, len(names) - 2))])
        ops.append(Op(f"op{i}", ins, out, macs=tb[out]))
        names.append(out)
    return Graph("random_dag", tb, ops, ["t0"])
