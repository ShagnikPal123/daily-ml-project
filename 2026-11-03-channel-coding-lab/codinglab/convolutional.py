"""Rate-1/2 feed-forward convolutional code with hard/soft-decision Viterbi decoding.

Constraint length K, generators in octal (default K=7: 133,171 -- the NASA/CCSDS code).
State = last K-1 input bits.  The batch Viterbi is vectorised across frames.
"""
import numpy as np


class ConvCode:
    def __init__(self, K=7, gens=(0o133, 0o171)):
        self.K, self.gens = K, gens
        self.n_out = len(gens)
        self.n_states = 1 << (K - 1)
        self.rate_nominal = 1.0 / self.n_out
        self._build_trellis()

    def _parity(self, x):
        x = np.asarray(x)
        p = np.zeros_like(x)
        while np.any(x):
            p ^= x & 1
            x = x >> 1
        return p

    def _build_trellis(self):
        S, K = self.n_states, self.K
        states = np.arange(S)
        self.next_state = np.zeros((S, 2), dtype=np.int64)
        self.out_bits = np.zeros((S, 2, self.n_out), dtype=np.uint8)
        for u in (0, 1):
            reg = (u << (K - 1)) | states         # newest bit is MSB of the K-bit register
            self.next_state[:, u] = reg >> 1
            for j, g in enumerate(self.gens):
                self.out_bits[:, u, j] = self._parity(reg & g)
        # predecessor table: for each next state, the two (prev_state, input) pairs
        self.pred = np.zeros((S, 2, 2), dtype=np.int64)
        cnt = np.zeros(S, dtype=int)
        for s in range(S):
            for u in (0, 1):
                ns = self.next_state[s, u]
                self.pred[ns, cnt[ns]] = (s, u)
                cnt[ns] += 1
        self.pred_s = self.pred[:, :, 0]
        self.pred_u = self.pred[:, :, 1]
        self.pred_out = self.out_bits[self.pred_s, self.pred_u]   # (S,2,n_out)

    def rate(self, n_info):
        return n_info / (self.n_out * (n_info + self.K - 1))

    def encode(self, info):
        """info (B,L) -> coded bits (B, (L+K-1)*n_out), zero-terminated."""
        info = np.asarray(info, dtype=np.uint8)
        B, L = info.shape
        padded = np.concatenate([info, np.zeros((B, self.K - 1), dtype=np.uint8)], axis=1)
        state = np.zeros(B, dtype=np.int64)
        out = np.zeros((B, L + self.K - 1, self.n_out), dtype=np.uint8)
        for t in range(padded.shape[1]):
            u = padded[:, t].astype(np.int64)
            out[:, t] = self.out_bits[state, u]
            state = self.next_state[state, u]
        return out.reshape(B, -1)

    def viterbi(self, metric_input, soft=True):
        """Decode. soft=True: metric_input is LLR (positive -> bit 0); soft=False: hard bits.

        Branch metric (to maximise): sum_j (1-2c_j) * LLR_j  (equivalent to min Euclidean distance).
        For hard decisions the LLR is replaced with +-1.  Returns info bits (B, L).
        """
        x = np.asarray(metric_input, dtype=np.float64)
        if not soft:
            x = 1.0 - 2.0 * x
        B = x.shape[0]
        T = x.shape[1] // self.n_out
        x = x.reshape(B, T, self.n_out)
        sgn = 1.0 - 2.0 * self.pred_out.astype(np.float64)      # (S,2,n_out)
        NEG = -1e18
        pm = np.full((B, self.n_states), NEG)
        pm[:, 0] = 0.0
        surv = np.zeros((T, B, self.n_states), dtype=np.uint8)
        for t in range(T):
            bm = np.einsum("bj,sij->bsi", x[:, t], sgn)          # (B,S,2)
            cand = pm[:, self.pred_s] + bm                       # (B,S,2)
            choose = (cand[:, :, 1] > cand[:, :, 0]).astype(np.uint8)
            surv[t] = choose
            pm = np.where(choose == 1, cand[:, :, 1], cand[:, :, 0])
        # terminated -> trace back from state 0
        state = np.zeros(B, dtype=np.int64)
        dec = np.zeros((B, T), dtype=np.uint8)
        idx = np.arange(B)
        for t in range(T - 1, -1, -1):
            c = surv[t, idx, state]
            dec[:, t] = self.pred_u[state, c]
            state = self.pred_s[state, c]
        return dec[:, : T - (self.K - 1)]
