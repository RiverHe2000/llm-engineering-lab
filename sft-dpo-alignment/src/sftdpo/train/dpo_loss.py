"""The DPO objective, derived and implemented over plain tensors.

No models, no tokenizers, no I/O: everything here is a pure function of log-probabilities,
which is exactly what makes the objective testable. The trainer that calls this module is
responsible for producing those numbers; this module is responsible for being right.

The derivation, so that the code below can be read against it. DPO starts from the
Bradley-Terry model of a preference between a chosen response ``y_w`` and a rejected
response ``y_l`` given a prompt ``x``::

    p(y_w > y_l | x) = sigma( r(x, y_w) - r(x, y_l) )

and from the closed form of the KL-regularised RLHF optimum::

    max_pi  E_pi[ r(x, y) ] - beta * KL( pi || pi_ref )
      =>    pi*(y|x) = pi_ref(y|x) * exp( r(x, y) / beta ) / Z(x)

Rearranging the second line for ``r`` and substituting it into the first, the intractable
partition function ``Z(x)`` cancels, because both responses share the same prompt::

    r(x, y) = beta * log( pi(y|x) / pi_ref(y|x) ) + beta * log Z(x)

    p(y_w > y_l | x) = sigma( beta * [ log pi(y_w|x)/pi_ref(y_w|x)
                                       - log pi(y_l|x)/pi_ref(y_l|x) ] )

Maximum likelihood on the observed preferences is then the DPO loss, and no reward model is
ever fitted: the policy *is* the reward model, read through that log-ratio. Everything
downstream is a function of one scalar per pair, the difference of log-ratios::

    h = ( logp_pi(y_w) - logp_pi(y_l) ) - ( logp_ref(y_w) - logp_ref(y_l) )

:func:`dpo_loss` still takes four log-probabilities rather than ``h`` itself, because the
two implicit rewards ``beta * (logp_pi - logp_ref)`` have to be reported separately. Their
individual values are the diagnostic that says whether a run is lifting the chosen response
or merely crushing the rejected one; ``h`` alone cannot tell those apart, and the second is
the failure mode that quietly degrades a model while the loss curve looks healthy.

A memory note that shapes the signature. The reference policy is not a second set of weights
when the policy is a LoRA adapter: ``peft_model.disable_adapter()`` is a context manager that
yields the base model's behaviour, and it reproduces the base log-probabilities exactly. The
reference forward pass therefore costs activations and time, but not a second copy of the
model in VRAM. That is why this module accepts four tensors and cares nothing about where
they came from — the same weights, adapter on and adapter off, are a perfectly good pi and
pi_ref.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "IGNORE_INDEX",
    "LOGPROB_CHUNK_ELEMENTS",
    "DPOOutput",
    "DPOVariant",
    "dpo_loss",
    "sequence_logprob",
]

IGNORE_INDEX: Final = -100
"""Label value excluded from the log-probability sum, matching the Hugging Face convention."""

DPOVariant = Literal["sigmoid", "ipo", "cdpo"]

_VARIANTS: Final = frozenset({"sigmoid", "ipo", "cdpo"})

# log_softmax over a bf16/fp16 logit row loses enough precision to move a sequence
# log-probability by a few hundredths of a nat, which is the same order as the margins DPO
# is trying to learn. Promote those two dtypes and leave float32/float64 alone.
_LOW_PRECISION: Final = frozenset({torch.float16, torch.bfloat16})


LOGPROB_CHUNK_ELEMENTS: Final = 64_000_000
"""Upper bound on the float32 elements one log-softmax may materialise at a time.

Sizing note, measured rather than guessed. Scoring label tokens means a softmax over the
vocabulary, and Qwen2.5's vocabulary is 151 936 entries. One DPO pair at 2 048 tokens is
therefore two rows of 2 047 x 151 936 scores: 1.16 GiB in bfloat16, and 2.32 GiB again
for each float32 copy the promotion makes. Done in one shot on an RTX 4070 that peaks at
**6.95 GiB above its inputs for a single pair**, and 13.9 GiB for two -- which is how the
first real run of this pipeline died, in the LoRA layer of the reference forward pass with
25 GiB already allocated.

Chunking the promotion and the gather brings the same computation to **3.71 GiB for one
pair and 7.19 GiB for two**, and it is not an approximation: log-softmax is taken over the
vocabulary axis independently for every position, so splitting the position axis leaves both
the values and the gradients **bit-identical**. Measured difference against the unchunked
path: 0.0 on the log-probabilities and 0.0 on the gradient with respect to the logits.

The obvious alternative is worth recording because it is wrong. Replacing the whole thing
with ``-F.cross_entropy(flat_logits, targets, reduction="none")`` is the memory-efficient
idiom, needs 2.32 GiB, and on bfloat16 logits it does its arithmetic in bfloat16: the
sequence log-probabilities came out **18.5 nats away** from the float32 answer on the same
inputs. A DPO margin lives at the scale of a fraction of a nat. That optimisation would have
been invisible in a loss curve and fatal to the result.

64 million float32 elements is 244 MiB per chunk, which is small next to a model's
activations and large enough that the loop runs a handful of times rather than hundreds.
"""


def _gathered_logprob(predicting: Tensor, index: Tensor) -> Tensor:
    """Log-probability of the indexed token at each position, in float32.

    Splits the work along the flattened position axis so no single float32 buffer exceeds
    :data:`LOGPROB_CHUNK_ELEMENTS`. Equivalent to
    ``log_softmax(predicting.float(), -1).gather(2, index.unsqueeze(2)).squeeze(2)``
    for every input, and identical bit for bit, because each position's softmax depends only
    on that position's row of scores.

    Args:
        predicting: Scores of shape ``(batch, positions, vocab)``.
        index: Token ids of shape ``(batch, positions)``, already made valid.

    Returns:
        Float32 log-probabilities of shape ``(batch, positions)``.
    """
    vocab = predicting.shape[-1]
    flat = predicting.reshape(-1, vocab)
    flat_index = index.reshape(-1)
    rows = max(1, LOGPROB_CHUNK_ELEMENTS // vocab)
    if flat.shape[0] <= rows:
        promoted = flat.float() if flat.dtype in _LOW_PRECISION else flat
        gathered = torch.log_softmax(promoted, dim=-1).gather(1, flat_index.unsqueeze(1))
        return gathered.squeeze(1).view_as(index)

    pieces = []
    for start in range(0, flat.shape[0], rows):
        piece = flat[start : start + rows]
        promoted = piece.float() if piece.dtype in _LOW_PRECISION else piece
        gathered = torch.log_softmax(promoted, dim=-1).gather(
            1, flat_index[start : start + rows].unsqueeze(1)
        )
        pieces.append(gathered.squeeze(1))
    return torch.cat(pieces).view_as(index)


def sequence_logprob(logits: Tensor, labels: Tensor, *, average: bool = False) -> Tensor:
    """Log-probability of the label tokens of each sequence under ``logits``.

    Args:
        logits: Unnormalised scores of shape ``(batch, seq, vocab)``, exactly as a causal
            language model returns them.
        labels: Token ids of shape ``(batch, seq)``. Positions holding :data:`IGNORE_INDEX`
            are excluded, which is how prompt tokens and right-padding are masked out.
        average: Divide by the number of scored tokens, returning a per-token
            log-probability instead of a sum.

    Returns:
        A tensor of shape ``(batch,)``, in float32 or better.

    Raises:
        ValueError: If the shapes disagree, the sequence is shorter than two positions,
            ``labels`` does not hold integer ids, or an id falls outside the vocabulary.

    The off-by-one, which everyone writes at least once:
        A causal model at position ``t`` predicts the token at position ``t + 1``. The score
        of ``labels[:, t]`` therefore lives in ``logits[:, t - 1]``, and the two tensors must
        be shifted against each other before the gather::

            logits[:, :-1, :]   drop the last column: it predicts a token past the end
            labels[:, 1:]       drop the first label: nothing in the sequence predicts it

        Gathering ``labels[:, t]`` from ``logits[:, t]`` instead asks for the probability of
        each token *given itself*. That version runs, does not crash, and trains. It inflates
        the log-probability, and under DPO it inflates chosen and rejected alike, so the loss
        still falls and nothing looks wrong until the evaluation numbers refuse to move. The
        test suite pins the shift down from both sides: perturbing ``labels[:, 0]`` and
        perturbing ``logits[:, -1]`` must both leave the result bit-identical.

    Sum or average:
        The summed form is the log-probability of the sequence and is what the DPO
        derivation above assumes. It is also length-biased: every additional token adds
        another negative number, so a longer response scores lower for no reason other than
        being longer, and preference training on summed log-probabilities learns to prefer
        short responses. Length-averaging removes that bias but no longer corresponds to any
        sequence probability, so it changes the objective rather than merely rescaling it.
        The choice is the caller's; both are exposed so an experiment can measure it.

    A row with no scored tokens returns ``0.0`` in both modes rather than ``nan``: an empty
    sum is zero, and the average clamps the divisor. That keeps a fully padded row in a
    ragged batch from poisoning the batch mean.
    """
    if logits.ndim != 3:
        msg = f"logits must be (batch, seq, vocab); got shape {tuple(logits.shape)}"
        raise ValueError(msg)
    if labels.ndim != 2:
        msg = f"labels must be (batch, seq); got shape {tuple(labels.shape)}"
        raise ValueError(msg)
    if logits.shape[:2] != labels.shape:
        msg = (
            f"logits and labels must agree on (batch, seq); got {tuple(logits.shape[:2])} "
            f"and {tuple(labels.shape)}"
        )
        raise ValueError(msg)
    if labels.is_floating_point() or labels.dtype == torch.bool:
        msg = f"labels must hold integer token ids; got dtype {labels.dtype}"
        raise ValueError(msg)
    if labels.shape[1] < 2:
        msg = (
            "labels must span at least two positions; after the shift a length-1 row scores nothing"
        )
        raise ValueError(msg)

    target = labels[:, 1:]
    predicting = logits[:, :-1, :]
    scored = target != IGNORE_INDEX
    # Ignored positions still travel through the gather, so give them a valid index first.
    index = target.masked_fill(~scored, 0).to(torch.long)

    vocab = predicting.shape[-1]
    # One device synchronisation per call, which is nothing next to the forward pass that
    # produced these logits, and much cheaper than the alternative: an out-of-range gather
    # on CUDA raises a device-side assert that cannot be caught and takes the process down.
    if bool(((index < 0) | (index >= vocab)).any()):
        msg = f"label ids must lie in [0, {vocab}) or equal {IGNORE_INDEX}"
        raise ValueError(msg)

    token_logp = _gathered_logprob(predicting, index)
    # masked_fill, not a multiply by the mask: a padded position whose substitute token has
    # probability zero gathers -inf, and 0 * -inf is nan. The fill also blocks the gradient
    # at ignored positions, which multiplication by a zero would do as well but less obviously.
    token_logp = token_logp.masked_fill(~scored, 0.0)

    total = token_logp.sum(dim=-1)
    if not average:
        return total
    counts = scored.sum(dim=-1).to(total.dtype)
    return total / counts.clamp(min=1.0)


@dataclass(frozen=True, slots=True)
class DPOOutput:
    """Everything one DPO step produces, loss and diagnostics together.

    The rewards are detached. They are metrics, not part of the graph: the gradient of the
    objective already flows through ``loss``, and a live graph hanging off a logged tensor
    is how a training loop ends up retaining every step's activations.

    Attributes:
        loss: Scalar batch mean, the value to call ``backward()`` on.
        losses: Per-pair losses, shaped like the inputs. Exposed so a caller can reduce
            differently — per-slice reporting, or a weighted mean over a mixed batch.
        chosen_rewards: ``beta * (policy_chosen_logp - ref_chosen_logp)``, the implicit
            reward DPO assigns to the chosen response, up to the ``beta * log Z(x)`` term
            that cancels within a pair and so is never computed.
        rejected_rewards: The same quantity for the rejected response.
        reward_margins: ``chosen_rewards - rejected_rewards``. Equal to ``beta * h``; the
            single number the sigmoid and cDPO objectives are functions of.
        reward_accuracy: Fraction of pairs the implicit reward ranks correctly. A tie counts
            as a miss, because a policy that cannot separate the pair has not learned it.
    """

    loss: Tensor
    losses: Tensor
    chosen_rewards: Tensor
    rejected_rewards: Tensor
    reward_margins: Tensor
    reward_accuracy: Tensor

    @property
    def reward_margin(self) -> Tensor:
        """Mean reward margin over the batch, as a scalar tensor."""
        return self.reward_margins.mean()

    def metrics(self) -> dict[str, float]:
        """Plain floats for a logger, with the names a DPO run is normally read by."""
        return {
            "loss": float(self.loss),
            "reward_chosen": float(self.chosen_rewards.mean()),
            "reward_rejected": float(self.rejected_rewards.mean()),
            "reward_margin": float(self.reward_margin),
            "reward_accuracy": float(self.reward_accuracy),
        }


def dpo_loss(
    policy_chosen_logp: Tensor,
    policy_rejected_logp: Tensor,
    ref_chosen_logp: Tensor,
    ref_rejected_logp: Tensor,
    *,
    beta: float = 0.1,
    variant: DPOVariant = "sigmoid",
    label_smoothing: float = 0.0,
) -> DPOOutput:
    """Direct Preference Optimisation loss over four sequence log-probabilities.

    Args:
        policy_chosen_logp: ``log pi(y_w | x)``, one entry per pair, requiring grad.
        policy_rejected_logp: ``log pi(y_l | x)``.
        ref_chosen_logp: ``log pi_ref(y_w | x)``. Constant with respect to the parameters;
            gradients into it are meaningless but harmless.
        ref_rejected_logp: ``log pi_ref(y_l | x)``.
        beta: Strength of the KL constraint in the RLHF problem this objective solves. Small
            beta lets the policy move far from the reference; large beta pins it. It is the
            scale of the implicit reward, not a learning rate.
        variant: ``"sigmoid"``, ``"ipo"`` or ``"cdpo"``, derived below.
        label_smoothing: Assumed probability that a preference label is wrong. Defined only
            for ``"cdpo"``, and must lie in ``[0, 0.5)``.

    Returns:
        A :class:`DPOOutput`.

    Raises:
        ValueError: If ``beta`` is not positive, the variant is unknown, ``label_smoothing``
            is out of range or given for a variant that has no smoothing term, or the four
            inputs disagree in shape or are not floating point.

    All three variants are functions of the same scalar per pair::

        h = (policy_chosen_logp - policy_rejected_logp)
            - (ref_chosen_logp - ref_rejected_logp)

    "sigmoid" — the original objective (Rafailov et al., 2023):
        Bradley-Terry maximum likelihood on the implicit reward,
        ``L = -log sigma(beta * h)``. Its gradient with respect to ``beta * h`` is
        ``-sigma(-beta * h)``, which decays to zero as the margin grows: a pair the model
        already gets right stops contributing. That is the intended behaviour, and also the
        weakness. With deterministic preferences — which is exactly the case here, where the
        labels come from a verifier and not from annotators — the likelihood is maximised
        only as ``h -> infinity``, so nothing in the loss bounds how far the policy drifts
        from the reference except how soon training stops.

    "ipo" — identity preference optimisation (Azar et al., 2023):
        Replaces maximum likelihood with a regression onto a fixed target,
        ``L = (h - 1 / (2 * beta))^2``. The population minimiser is ``h* = 1 / (2 * beta)``,
        a finite margin, so the objective cannot be driven down by pushing the log-ratio to
        infinity, and the KL constraint keeps biting no matter how separable the data is.
        Its gradient ``2 * (h - 1 / (2 * beta))`` grows linearly in the distance from the
        target rather than saturating, which is what "does not saturate" means in practice:
        a pair that has overshot keeps producing gradient, pulling it back.

    "cdpo" — conservative DPO, the label-smoothed mixture:
        Assumes each observed preference is flipped with probability ``eps``, and takes the
        likelihood of the noisy label::

            L = -(1 - eps) * log sigma(beta * h) - eps * log sigma(-beta * h)

        Writing ``u = beta * h``, the derivative collapses to ``dL/du = sigma(u) - (1 - eps)``.
        So the loss is minimised at ``sigma(u*) = 1 - eps``, that is
        ``h* = log((1 - eps) / eps) / beta``, and its minimum value is the binary entropy
        ``H(eps) = -(1 - eps) log(1 - eps) - eps log eps``. Two consequences worth stating:
        the target margin is finite, like IPO's but reached by a different route; and beyond
        it the gradient changes sign, so cDPO actively pulls an over-confident pair back.
        ``eps = 0`` reduces to "sigmoid" exactly, and ``eps >= 0.5`` would invert the
        preference, which is why it is rejected rather than clamped.

    Numerics:
        ``F.logsigmoid`` throughout, never ``log(sigmoid(x))``. In float32 ``sigmoid(-200)``
        underflows to exactly zero, so the naive form returns ``-inf`` and a ``nan``
        gradient, while ``logsigmoid(-200)`` returns ``-200``. Log-ratio differences of that
        size are not hypothetical late in a DPO run on separable data, which is precisely
        when the sigmoid variant is least constrained.
    """
    if not beta > 0.0:
        msg = f"beta must be positive; got {beta}"
        raise ValueError(msg)
    if variant not in _VARIANTS:
        msg = f"unknown variant {variant!r}; expected one of {sorted(_VARIANTS)}"
        raise ValueError(msg)
    if not 0.0 <= label_smoothing < 0.5:
        msg = (
            f"label_smoothing must lie in [0, 0.5); got {label_smoothing}. At 0.5 the "
            "objective is minimised at a zero margin, and above it the preference inverts."
        )
        raise ValueError(msg)
    if label_smoothing != 0.0 and variant != "cdpo":
        msg = (
            f"label_smoothing is only defined for variant='cdpo'; the {variant!r} objective "
            "has no smoothing term"
        )
        raise ValueError(msg)

    tensors = (policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp)
    names = ("policy_chosen_logp", "policy_rejected_logp", "ref_chosen_logp", "ref_rejected_logp")
    for name, tensor in zip(names, tensors, strict=True):
        if tensor.shape != policy_chosen_logp.shape:
            msg = (
                f"{name} has shape {tuple(tensor.shape)}, expected "
                f"{tuple(policy_chosen_logp.shape)}: the four inputs are aligned per pair"
            )
            raise ValueError(msg)
        if not tensor.is_floating_point():
            msg = f"{name} must be floating point; got dtype {tensor.dtype}"
            raise ValueError(msg)

    policy_logratio = policy_chosen_logp - policy_rejected_logp
    ref_logratio = ref_chosen_logp - ref_rejected_logp
    h = policy_logratio - ref_logratio

    if variant == "ipo":
        losses = (h - 1.0 / (2.0 * beta)) ** 2
    else:
        u = beta * h
        if label_smoothing == 0.0:
            # Branch rather than multiply by zero: the smoothing term is finite here, but a
            # 0.0 * logsigmoid(u) factor would produce nan the moment an input is infinite,
            # and this path is also the one that must reduce to the original objective exactly.
            losses = -F.logsigmoid(u)
        else:
            losses = -(1.0 - label_smoothing) * F.logsigmoid(u) - label_smoothing * F.logsigmoid(-u)

    chosen_rewards = beta * (policy_chosen_logp - ref_chosen_logp).detach()
    rejected_rewards = beta * (policy_rejected_logp - ref_rejected_logp).detach()
    margins = chosen_rewards - rejected_rewards
    accuracy = (chosen_rewards > rejected_rewards).to(margins.dtype).mean()

    return DPOOutput(
        loss=losses.mean(),
        losses=losses,
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        reward_margins=margins,
        reward_accuracy=accuracy,
    )
