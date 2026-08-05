"""SimCT loss for cross-tokenizer co-OPSD — arXiv 2605.07711.

Replaces GOLD's `ULDLoss` (token-merging + sorted-probability L1 on the
unmatched remainder) with SimCT's principled supervision space:
`U = (V_T ∩ V_S) ∪ A`, where `A` is the set of minimal aligned units (short
multi-token spans both tokenizers can jointly realize — see `simct_align`).

Call signature matches `ULDLoss.__call__` so it drops into
`CoOPSDTrainer._distill_one_direction` at the same switch point.

Design (documented approximation, kept deliberately simple per repo AGENTS.md):
  * **Shared 1:1 units** (~96% of positions here): student and teacher next-token
    distributions, each restricted to the shared vocab `V_T ∩ V_S` via a
    precomputed index bridge, compared with the generalized JSD already used on
    the same-tokenizer path. This is the exact cross-tokenizer form of GKD.
  * **Minimal aligned units** (the divergences — almost all multi-digit numbers
    for Llama×Qwen): SimCT's recovered signal. For the unit the student actually
    produced, match the two models' **length-normalized** log-probabilities
    `s_M(u)=(1/k)Σ log p_M`, so the teacher's preference over the whole span
    supervises the student even though the two tokenizers cut it differently.
"""

import torch
import torch.nn.functional as F

from simct_align import align


class SimCTLoss:
    def __init__(self, student_tok, teacher_tok, beta=0.0, temperature=1.0,
                 token_clip=None, unit_weight=1.0):
        self.student_tok = student_tok
        self.teacher_tok = teacher_tok
        self.beta = beta
        self.temperature = temperature
        self.token_clip = token_clip
        self.unit_weight = unit_weight
        # ---- vocab bridge over the shared surface strings (V_T ∩ V_S) ----
        s_vocab = student_tok.get_vocab()   # str -> student id
        t_vocab = teacher_tok.get_vocab()   # str -> teacher id
        shared = sorted(set(s_vocab) & set(t_vocab))
        # parallel index tensors: column c of the shared space is `shared[c]`,
        # realized by student id s_idx[c] and teacher id t_idx[c].
        self.s_idx = torch.tensor([s_vocab[w] for w in shared], dtype=torch.long)
        self.t_idx = torch.tensor([t_vocab[w] for w in shared], dtype=torch.long)
        self.n_shared = len(shared)

    def _jsd_shared(self, s_logits_pos, t_logits_pos):
        """Generalized JSD between one student and one teacher next-token dist,
        both restricted+reordered to the shared vocab. Shapes: [N_shared]."""
        s_lp = F.log_softmax(s_logits_pos / self.temperature, dim=-1)
        t_lp = F.log_softmax(t_logits_pos / self.temperature, dim=-1)
        if self.beta == 0:
            jsd = F.kl_div(s_lp, t_lp, reduction="none", log_target=True)
        elif self.beta == 1:
            jsd = F.kl_div(t_lp, s_lp, reduction="none", log_target=True)
        else:
            b = self.beta
            m = torch.logsumexp(torch.stack(
                [s_lp + torch.log(torch.tensor(1 - b)), t_lp + torch.log(torch.tensor(b))]), dim=0)
            jsd = b * F.kl_div(m, t_lp, reduction="none", log_target=True) \
                + (1 - b) * F.kl_div(m, s_lp, reduction="none", log_target=True)
        if self.token_clip is not None:
            jsd = jsd.clamp(max=self.token_clip)
        return jsd.sum()

    def __call__(self, student_logits, teacher_logits, student_labels,
                 teacher_labels, student_input_ids, teacher_input_ids):
        device = student_logits.device
        s_idx = self.s_idx.to(device)
        t_idx = self.t_idx.to(device)
        total = student_logits.new_zeros(())
        n_terms = 0

        B = student_logits.size(0)
        for b in range(B):
            # trajectory (answer) region = positions the collator left unmasked
            s_pos = (student_labels[b] != -100).nonzero(as_tuple=True)[0]
            t_pos = (teacher_labels[b] != -100).nonzero(as_tuple=True)[0]
            if s_pos.numel() == 0 or t_pos.numel() == 0:
                continue
            s0, s1 = int(s_pos[0]), int(s_pos[-1]) + 1
            t0, t1 = int(t_pos[0]), int(t_pos[-1]) + 1
            s_traj = student_input_ids[b, s0:s1].tolist()
            t_traj = teacher_input_ids[b, t0:t1].tolist()
            units = align(s_traj, t_traj, self.student_tok, self.teacher_tok)
            if units is None:
                continue  # fall back: skip this example's SimCT term

            # log-softmax over the FULL vocab at every trajectory position, plus
            # the shared-restricted logits for the JSD term.
            for u in units:
                # student position predicting the unit's first token: (s0+u.s_start-1)
                sp = s0 + u.s_start - 1
                tp = t0 + u.t_start - 1
                if sp < 0 or tp < 0:
                    continue
                if u.is_shared:
                    # cross-vocab JSD on shared support at this aligned position
                    total = total + self._jsd_shared(
                        student_logits[b, sp, s_idx], teacher_logits[b, tp, t_idx])
                    n_terms += 1
                else:
                    # minimal aligned unit: match length-normalized log-probs of
                    # the produced span under each model (teacher no-grad already).
                    s_lp = 0.0
                    for j in range(u.s_len):
                        pos = s0 + u.s_start + j - 1
                        tok = s_traj[u.s_start + j]
                        s_lp = s_lp + F.log_softmax(student_logits[b, pos], dim=-1)[tok]
                    s_lp = s_lp / u.s_len
                    t_lp = 0.0
                    for j in range(u.t_len):
                        pos = t0 + u.t_start + j - 1
                        tok = t_traj[u.t_start + j]
                        t_lp = t_lp + F.log_softmax(teacher_logits[b, pos], dim=-1)[tok]
                    t_lp = t_lp / u.t_len
                    # forward-KL-flavored: pull student's unit log-prob toward the
                    # teacher's (detached target). Squared gap is a stable surrogate.
                    total = total + self.unit_weight * (t_lp.detach() - s_lp) ** 2
                    n_terms += 1

        if n_terms == 0:
            return student_logits.new_zeros(()) + student_logits.sum() * 0.0
        return total / n_terms
