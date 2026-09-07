"""Deterministic synthetic adviser notes and the gold advice record for each one.

The whole project rests on being able to say "the model got better" without hedging, and
that claim dies if the data is either non-deterministic or leaky. So every note here is a
pure function of a `random.Random` stream: no clock, no filesystem, no network, no set
iteration, no hash ordering. Re-running the generator a year from now on another machine
reproduces the corpus byte for byte, which is what lets an experiment record cite a data
hash and mean it.

The notes are synthetic on purpose. Real adviser file notes are client data and have no
place in a public portfolio repository. The register is Australian wealth management -
superannuation, pension phase, adviser service fees, insurance in super, ASX tickers -
because that vocabulary is what the model actually has to cope with, and because notes
written that way reproduce the real failure modes: a figure corrected mid-sentence, a date
dictated three different ways, an optional field the adviser simply never mentioned.

Difficulty is organised into `Slice` values rather than a single blended pile, so a lift in
the average can never hide a regression on, say, absent fields. One builder per slice, all
of them drawing from the same fact model, so the slices differ only in how the same kind of
fact is expressed - which is precisely the variable under test.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

from sftdpo.schemas import (
    Action,
    AdviceRecord,
    Fees,
    Recommendation,
    RiskProfile,
    Slice,
    advice_record_json_schema,
)

__all__ = ["GeneratedNote", "generate", "generate_one", "render_prompt"]


# --------------------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------------------

_FIRST_NAMES: tuple[str, ...] = (
    "Margaret",
    "Daniel",
    "Priya",
    "Tomas",
    "Alison",
    "Hamish",
    "Wei",
    "Bronwyn",
    "Ngaire",
    "Callum",
    "Rosa",
    "Desmond",
    "Imogen",
    "Farid",
    "Kylie",
    "Angus",
)

_LAST_NAMES: tuple[str, ...] = (
    "Whitlock",
    "Okafor",
    "Nguyen",
    "Kowalski",
    "Fairbairn",
    "McAllister",
    "Zhang",
    "Petrides",
    "Boland",
    "Sultana",
    "Cavanagh",
    "Rasmussen",
    "Iyer",
    "Duong",
    "Hargreaves",
    "Moloney",
)

# No product name is a substring of another, so "does this note mention product X" is an
# unambiguous question and the tests can check ordering by string position.
_PRODUCTS: tuple[str, ...] = (
    "AustralianSuper Balanced",
    "Hostplus Indexed Balanced",
    "Australian Retirement Trust Lifecycle Balanced Pool",
    "CareSuper Sustainable Balanced",
    "Colonial First State FirstChoice Wholesale Pension",
    "Netwealth Super Accelerator account-based pension",
    "BT Panorama Investments wrap",
    "MLC MasterKey Business Super",
    "Vanguard Australian Shares Index Fund",
    "Betashares Australia 200 ETF (ASX: A200)",
    "Vanguard MSCI International Shares ETF (ASX: VGS)",
    "BHP Group (ASX: BHP)",
    "CSL Limited (ASX: CSL)",
    "Telstra Group (ASX: TLS)",
    "Macquarie Cash Management Account",
    "Challenger Guaranteed Annuity (Liquid Lifetime)",
    "TPD cover held as insurance in super",
    "Term deposit ladder with ING",
)

# Objectives are copied verbatim into both the note and the gold record. That is what makes
# the absent-fields slice checkable: if the gold says the list is empty, no catalogue phrase
# may appear anywhere in the note, and a test can assert exactly that.
_OBJECTIVES: tuple[str, ...] = (
    "retire at 62 and move the superannuation balance into pension phase",
    "consolidate three legacy superannuation accounts into one fund",
    "maximise concessional contributions through salary sacrifice",
    "fund a kitchen renovation in the 2028 calendar year",
    "build a non-superannuation income stream before retirement",
    "protect the household income against illness or injury",
    "reduce the Age Pension assets test impact of the restructure",
    "leave a tax-effective inheritance to two adult children",
    "keep at least six months of living expenses in cash",
    "review insurance in super so premiums stop eroding the balance",
)

# Flags are snake_case tokens in the record and a full sentence in the note, so the model
# has to recognise the concept rather than copy a string.
_FLAG_PROSE: dict[str, str] = {
    "insurance_in_super_reduces_balance": (
        "I noted that the premiums deducted inside super will keep reducing the retirement balance."
    ),
    "transfer_balance_cap_check_required": (
        "We have to check the transfer balance cap before any pension commences."
    ),
    "adviser_service_fee_consent_due": (
        "The adviser service fee consent is due for renewal before the next anniversary."
    ),
    "centrelink_means_test_impact": (
        "I flagged the Centrelink means test consequences of the restructure."
    ),
    "capital_gains_on_switch": (
        "I warned the client that switching will realise capital gains this financial year."
    ),
    "cooling_off_period_applies": (
        "I reminded the client that a cooling-off period applies to the new product."
    ),
    "preservation_age_not_reached": (
        "I recorded that the client has not yet reached preservation age."
    ),
    "sole_purpose_test_reviewed": (
        "We reviewed the sole purpose test before agreeing to the in-specie transfer."
    ),
}

_FLAGS: tuple[str, ...] = tuple(_FLAG_PROSE)

_ACTIONS: tuple[Action, ...] = tuple(Action)
_ACTIONS_WITH_AMOUNT: tuple[Action, ...] = (Action.BUY, Action.SELL, Action.SWITCH)
_RISK_PROFILES: tuple[RiskProfile, ...] = tuple(RiskProfile)

_ACTION_VERBS: dict[Action, str] = {
    Action.BUY: "buy",
    Action.SELL: "sell",
    Action.HOLD: "hold",
    Action.SWITCH: "switch",
}

_ADVICE_FEES: tuple[float, ...] = (2200.0, 3300.0, 4400.0, 5500.0, 6600.0)
# Every value times one hundred is a whole number of basis points, which is what lets the
# mixed-formats slice quote the same fee as "seventy-seven basis points" without rounding.
_ONGOING_FEE_PCTS: tuple[float, ...] = (0.44, 0.55, 0.66, 0.77, 0.88, 1.1)
_REVIEW_MONTHS: tuple[int, ...] = (3, 6, 12, 24)

_CHITCHAT: tuple[str, ...] = (
    "Morning. The coffee machine is broken again so we met in the small room.",
    "Quick one before I forget the detail.",
    "Dictating this in the car park, apologies for the noise.",
    "Right, file note, and remind me to chase the fund about the last one.",
)

_ANECDOTES: tuple[str, ...] = (
    "He spent the first ten minutes on his son's cricket season; the under fifteens lost the "
    "grand final on the last ball and he is still sore about it.",
    "She told me at length about the renovation next door and the dispute over the fence, "
    "none of which has anything to do with the advice.",
    "We got sidetracked onto the new train line and whether it will ever reach the airport.",
    "He wanted my opinion on his brother's caravan purchase, which I declined to give.",
)

_CLOSERS: tuple[str, ...] = (
    "Also remind me to book the car in for a service.",
    "End of note. Coffee run.",
    "That is the lot, back to the inbox.",
    "Note ends. I still owe the paraplanner last week's file.",
)

# Realistic meeting narrative with no digits, no dollar figures and no percentages, so
# padding the long-context slice cannot accidentally introduce a competing fact. The total
# length is asserted by a test, because it is what guarantees the padding loop terminates.
_FILLER: tuple[str, ...] = (
    "The client arrived a few minutes early and we sat in the front meeting room with the "
    "door open because the air conditioning was struggling again.",
    "We spent the first part of the meeting catching up on how the move to the new house had "
    "gone and whether the commute has turned out to be as bad as expected.",
    "Parking near the office is still a problem, so the appointment started later than "
    "planned and we had to work through the agenda more briskly than I would have liked.",
    "Her partner joined for part of the discussion by phone and dropped off once we had "
    "finished the part that affected both of them.",
    "We talked briefly about the recent volatility in markets and why a long term plan does "
    "not get rewritten every time the news cycle turns unpleasant.",
    "I reminded the client that the statement of advice follows in writing and that nothing "
    "is actioned until it has been read and signed.",
    "There was a long conversation about aged care for a parent, which is not something we "
    "are acting on yet but which will come back to us within a year or two.",
    "The client asked again how the practice charges for its work and I walked through the "
    "fee disclosure document line by line until she was satisfied.",
    "We agreed that email is the best way to reach the client during working hours and that "
    "anything urgent should go to the mobile instead.",
    "The client mentioned that a colleague had recommended a different adviser, and we "
    "talked about what to look for and what questions are worth asking.",
    "I explained the difference between preservation age and pension phase on the whiteboard, "
    "which seemed to land better than the version in the brochure.",
    "The client wanted to understand what happens to the account if they go back to work part "
    "time for a couple of years after leaving the current employer.",
    "We reviewed the binding death benefit nominations together and agreed that they still "
    "reflect what the client wants to happen.",
    "The estate planning documents were last updated a long time ago and the client has "
    "undertaken to speak to a solicitor before the next review.",
    "I made a note to send through the product disclosure statement after the meeting so the "
    "client can read it without me sitting opposite.",
    "The client is comfortable with the level of reporting they receive and does not want "
    "anything more frequent, which suits both of us.",
    "The conversation turned to the client's plans for a long trip overseas once the youngest "
    "finishes school at the end of next year.",
    "I confirmed the contact details we hold on file and corrected the postal address, which "
    "had not been updated since the move.",
    "The client asked whether the practice has capacity to look after a sibling as well, and "
    "I said I would come back to them on that.",
    "We agreed to keep the same rhythm of contact as last year unless something material "
    "changes in the household or at work.",
    "The client's employer has changed payroll providers, which delayed the last few "
    "contributions and caused a certain amount of unnecessary anxiety.",
    "There was a short digression about the football which I will not record here in any "
    "detail beyond noting that it happened.",
    "The client finds the annual statement from the fund hard to read and much prefers the "
    "summary the practice produces after each review.",
    "We looked at the projection tool together and talked through the assumptions sitting "
    "behind it, because the output is only as good as those.",
    "The client is not interested in direct property and has said so consistently across "
    "several meetings, so I have stopped raising it.",
    "I noted that the client prefers to make decisions after sleeping on them rather than in "
    "the room, and that pushing for a decision on the day never works.",
    "The meeting ran well over time because the client had brought a list of questions "
    "written out on paper and worked through every one of them.",
    "We finished by agreeing what happens next and who is responsible for each step, which I "
    "have summarised at the end of this note.",
    "The client walked out with the paperwork under one arm and said they would call if "
    "anything in it was unclear.",
    "The weather was miserable and the client had walked up from the station, so we started "
    "with a cup of tea and let the room warm up.",
    "I mentioned that the office closes over the holiday period and gave the client the "
    "number for the on call adviser in case anything comes up.",
    "We agreed the client would forward the latest annual statement from the old fund so the "
    "paraplanner does not have to chase it a second time.",
    "The client asked how other people in a similar position tend to approach the decision, "
    "and I was careful to answer in general terms only.",
    "There was some discussion of the client's brother, who has been giving unsolicited "
    "financial advice at family gatherings for years.",
    "I explained why the practice keeps a written record of every meeting and what happens to "
    "that record if the client ever moves to another firm.",
    "The client thanked the front desk on the way out, which I mention only because the last "
    "few reviews have been considerably tenser than this one.",
)

_MONTHS: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

_WEEKDAYS: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

_ONES: tuple[str, ...] = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)

_TENS: tuple[str, ...] = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)

_ORDINAL_SUFFIXES: dict[int, str] = {1: "st", 2: "nd", 3: "rd"}

_YEAR = 2026
_LONG_CONTEXT_TARGET_WORDS = 640


# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------


def _below_thousand(value: int) -> str:
    """Spell out 0-999 in the Australian idiom ("one hundred and twenty")."""
    if value < 20:
        return _ONES[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"
    hundreds, rest = divmod(value, 100)
    head = f"{_ONES[hundreds]} hundred"
    return head if rest == 0 else f"{head} and {_below_thousand(rest)}"


def _int_to_words(value: int) -> str:
    """Spell out a non-negative integer below one million.

    Written by hand rather than pulled from a dependency because the mapping has to stay
    frozen: the word form of a figure is part of the corpus, and a library upgrade that
    changed "one hundred and twenty" to "one hundred twenty" would silently change every
    data hash in every experiment record.
    """
    if not 0 <= value < 1_000_000:
        raise ValueError(f"value must be in [0, 1000000), got {value}")
    if value < 1000:
        return _below_thousand(value)
    thousands, rest = divmod(value, 1000)
    head = f"{_below_thousand(thousands)} thousand"
    return head if rest == 0 else f"{head} and {_below_thousand(rest)}"


def _whole_dollars(amount: float) -> int:
    """Reject fractional amounts before they reach a formatter that cannot express them."""
    if not float(amount).is_integer():
        raise ValueError(f"amount must be a whole number of dollars, got {amount!r}")
    return int(amount)


def _money_plain(amount: float) -> str:
    return f"${amount:,.0f}"


def _money_k(amount: float) -> str:
    dollars = _whole_dollars(amount)
    if dollars % 1000 != 0:
        raise ValueError(f"the k form needs a whole number of thousands, got {amount!r}")
    return f"${dollars // 1000}k"


def _money_decimal(amount: float) -> str:
    return f"{amount:,.2f}"


def _money_words(amount: float) -> str:
    return f"{_int_to_words(_whole_dollars(amount))} dollars"


def _basis_points(pct: float) -> str:
    points = pct * 100.0
    if abs(points - round(points)) > 1e-9:
        raise ValueError(f"percentage is not a whole number of basis points: {pct!r}")
    return f"{round(points)} basis points"


def _ordinal(day: int) -> str:
    suffix = "th" if 11 <= day % 100 <= 13 else _ORDINAL_SUFFIXES.get(day % 10, "th")
    return f"{day}{suffix}"


def _date_long(day: dt.date) -> str:
    return f"{day.day} {_MONTHS[day.month - 1]} {day.year}"


def _date_slash(day: dt.date) -> str:
    """Australian day-first order, which is exactly the ambiguity the model has to survive."""
    return f"{day.day:02d}/{day.month:02d}/{day.year}"


def _date_relative(day: dt.date) -> str:
    """Relative phrasing that still resolves, because the month is named and the note carries
    an explicit dictation date for the year."""
    return f"next {_WEEKDAYS[day.weekday()]} the {_ordinal(day.day)} of {_MONTHS[day.month - 1]}"


_DATE_STYLES: tuple[Callable[[dt.date], str], ...] = (_date_long, _date_slash, _date_relative)
_MONEY_STYLES: tuple[Callable[[float], str], ...] = (_money_words, _money_k, _money_decimal)


def _sentence(phrase: str) -> str:
    return phrase[0].upper() + phrase[1:]


def _numbered(phrases: Sequence[str]) -> list[str]:
    return [f"{i}. {_sentence(phrase)}." for i, phrase in enumerate(phrases, start=1)]


def _word_count(text: str) -> int:
    return len(text.split())


def _risk_phrase(risk: RiskProfile) -> str:
    return risk.value.replace("_", " ")


# --------------------------------------------------------------------------------------
# The fact model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Facts:
    """The facts of one file note, before anybody decides how to phrase them.

    Keeping the facts separate from the prose is what makes the slices comparable: the
    difficulty knob changes the rendering, never the underlying record, so a score
    difference between slices is a difference in reading comprehension rather than in the
    difficulty of the answer.
    """

    client: str
    date: dt.date
    risk: RiskProfile
    objectives: tuple[str, ...]
    recommendations: tuple[Recommendation, ...]
    advice_fee: float
    ongoing_fee_pct: float
    review_months: int
    flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GeneratedNote:
    """One adviser note and the record a correct model would emit for it."""

    slice: Slice
    note: str
    gold: AdviceRecord


def _draw_amount(rng: random.Random, *, whole_thousands: bool) -> float:
    if whole_thousands:
        return float(rng.randrange(5, 250) * 1000)
    return float(rng.randrange(10, 500) * 500)


def _draw_recommendations(
    rng: random.Random,
    count: int,
    *,
    whole_thousands: bool = False,
    allow_hold: bool = True,
    ensure_amount: bool = False,
) -> tuple[Recommendation, ...]:
    """Draw distinct products, one recommendation each.

    Products are sampled without replacement so that a note never names the same product
    twice; the ordering tests depend on each product having a single position in the text.
    """
    products = rng.sample(_PRODUCTS, count)
    drawn: list[Recommendation] = []
    for index, product in enumerate(products):
        pool = _ACTIONS if allow_hold else _ACTIONS_WITH_AMOUNT
        if ensure_amount and index == 0:
            pool = _ACTIONS_WITH_AMOUNT
        action = rng.choice(pool)
        amount = (
            None if action is Action.HOLD else _draw_amount(rng, whole_thousands=whole_thousands)
        )
        drawn.append(Recommendation(product=product, action=action, amount=amount))
    return tuple(drawn)


def _draw_facts(
    rng: random.Random,
    *,
    n_recs: int,
    n_objectives: int = 2,
    n_flags: int = 1,
    whole_thousands: bool = False,
    allow_hold: bool = True,
    ensure_amount: bool = False,
) -> _Facts:
    return _Facts(
        client=f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}",
        date=dt.date(_YEAR, rng.randint(1, 12), rng.randint(1, 28)),
        risk=rng.choice(_RISK_PROFILES),
        objectives=tuple(rng.sample(_OBJECTIVES, n_objectives)),
        recommendations=_draw_recommendations(
            rng,
            n_recs,
            whole_thousands=whole_thousands,
            allow_hold=allow_hold,
            ensure_amount=ensure_amount,
        ),
        advice_fee=rng.choice(_ADVICE_FEES),
        ongoing_fee_pct=rng.choice(_ONGOING_FEE_PCTS),
        review_months=rng.choice(_REVIEW_MONTHS),
        flags=tuple(rng.sample(_FLAGS, n_flags)),
    )


def _record(facts: _Facts) -> AdviceRecord:
    return AdviceRecord(
        client_name=facts.client,
        record_date=facts.date.isoformat(),
        risk_profile=facts.risk,
        objectives=list(facts.objectives),
        recommendations=list(facts.recommendations),
        fees=Fees(advice_fee=facts.advice_fee, ongoing_fee_pct=facts.ongoing_fee_pct),
        review_months=facts.review_months,
        flags=list(facts.flags),
    )


def _rec_phrase(rec: Recommendation, money: Callable[[float], str]) -> str:
    if rec.action is Action.HOLD:
        return f"hold {rec.product} as it stands"
    if rec.amount is None:
        return (
            f"{_ACTION_VERBS[rec.action]} {rec.product}, with the dollar figure still to be "
            "confirmed with the fund"
        )
    if rec.action is Action.SWITCH:
        return f"switch {money(rec.amount)} into {rec.product}"
    return f"{_ACTION_VERBS[rec.action]} {money(rec.amount)} of {rec.product}"


def _amount_of(rec: Recommendation) -> float:
    """Narrow a recommendation drawn with `ensure_amount` back to a plain float.

    The distractor slice has to supersede a figure, so it needs a recommendation that has
    one. Rather than assert, this raises with the product name attached: if a future change
    to the drawing rules ever hands it a hold, the failure names the culprit.
    """
    if rec.amount is None:
        raise ValueError(f"recommendation for {rec.product} has no amount to supersede")
    return rec.amount


def _fee_sentence(facts: _Facts) -> str:
    return (
        f"Advice fee of {_money_plain(facts.advice_fee)} including GST and an ongoing "
        f"adviser service fee of {facts.ongoing_fee_pct} per cent of funds under advice."
    )


# --------------------------------------------------------------------------------------
# One builder per slice
# --------------------------------------------------------------------------------------


def _build_clean(rng: random.Random) -> tuple[str, AdviceRecord]:
    """Everything stated once, plainly, in schema order, with the date already normalised.

    This slice is the control: a model that fails here has a formatting problem rather than
    a comprehension problem, and the slice breakdown makes that distinction for free.
    """
    facts = _draw_facts(rng, n_recs=rng.randint(2, 3), n_objectives=rng.randint(2, 3))
    lines = [
        f"File note for {facts.client}.",
        f"Date of record: {facts.date.isoformat()}.",
        f"Agreed risk profile: {_risk_phrase(facts.risk)}.",
        f"Objectives: {'; '.join(facts.objectives)}.",
        "Recommendations:",
        *_numbered([_rec_phrase(rec, _money_plain) for rec in facts.recommendations]),
        _fee_sentence(facts),
        f"Next review in {facts.review_months} months.",
        *(_FLAG_PROSE[flag] for flag in facts.flags),
    ]
    return "\n".join(lines), _record(facts)


def _build_distractor(rng: random.Random) -> tuple[str, AdviceRecord]:
    """Chit-chat, an unrelated anecdote, and two figures the adviser corrects mid-note.

    The corrections are the point. A model that copies the first number it sees scores well
    on every other slice and fails here, which is exactly the behaviour that shows up in
    production when an adviser changes their mind halfway through a sentence.
    """
    facts = _draw_facts(
        rng, n_recs=rng.randint(2, 3), n_objectives=rng.randint(2, 3), ensure_amount=True
    )
    corrected = facts.recommendations[0]
    amount = _amount_of(corrected)
    stale_amount = amount - rng.randrange(1, 9) * 500
    stale_review = rng.choice(tuple(m for m in _REVIEW_MONTHS if m != facts.review_months))

    first = _rec_phrase(corrected.model_copy(update={"amount": stale_amount}), _money_plain)
    rest = [_rec_phrase(rec, _money_plain) for rec in facts.recommendations[1:]]
    lines = [
        rng.choice(_CHITCHAT),
        f"File note for {facts.client}, date of record {facts.date.isoformat()}.",
        rng.choice(_ANECDOTES),
        f"Risk profile stays {_risk_phrase(facts.risk)}.",
        f"Objectives: {'; '.join(facts.objectives)}.",
        "Recommendations:",
        f"1. {_sentence(first)} - actually, make that {_money_plain(amount)}.",
        *[f"{i}. {_sentence(phrase)}." for i, phrase in enumerate(rest, start=2)],
        _fee_sentence(facts),
        f"I said {stale_review} months for the review earlier; ignore that, book it for "
        f"{facts.review_months} months.",
        *(_FLAG_PROSE[flag] for flag in facts.flags),
        rng.choice(_CLOSERS),
    ]
    return "\n".join(lines), _record(facts)


def _build_mixed_formats(rng: random.Random) -> tuple[str, AdviceRecord]:
    """The same facts dictated in inconsistent surface forms; the gold is always normalised.

    Three date styles and three money styles appear in every note, and which one carries the
    record date rotates, so a model cannot learn "the record date is the one with slashes".
    """
    facts = _draw_facts(
        rng,
        n_recs=3,
        n_objectives=rng.randint(2, 3),
        whole_thousands=True,
        allow_hold=False,
    )
    anchor = facts.date - dt.timedelta(days=rng.randint(3, 9))
    follow_up = facts.date + dt.timedelta(days=rng.randint(5, 20))
    style = rng.randrange(len(_DATE_STYLES))

    phrases = [
        _rec_phrase(rec, _MONEY_STYLES[(index + style) % len(_MONEY_STYLES)])
        for index, rec in enumerate(facts.recommendations)
    ]
    lines = [
        f"Dictated {_date_long(anchor)}.",
        f"Client {facts.client}. Record date: {_DATE_STYLES[style](facts.date)}.",
        f"Risk profile {_risk_phrase(facts.risk)}.",
        f"Objectives: {'; '.join(facts.objectives)}.",
        "Recommendations:",
        *_numbered(phrases),
        f"Advice fee {_money_words(facts.advice_fee)} in total, ongoing adviser service fee "
        f"{_basis_points(facts.ongoing_fee_pct)}.",
        f"Review in {_int_to_words(facts.review_months)} months.",
        f"Follow-up appointment pencilled in for "
        f"{_DATE_STYLES[(style + 1) % len(_DATE_STYLES)](follow_up)}.",
        *(_FLAG_PROSE[flag] for flag in facts.flags),
    ]
    return "\n".join(lines), _record(facts)


def _build_absent_fields(rng: random.Random) -> tuple[str, AdviceRecord]:
    """One optional field genuinely never mentioned, so the gold is empty or null.

    The failure this slice catches is invention: a model that has learned the shape of the
    record will happily fill in a plausible objective nobody stated. Because the omitted
    material is removed from the note entirely rather than replaced by a phrase such as "no
    objectives discussed", a test can assert that no catalogue phrase survives in the text.
    """
    mode = rng.choice(("objectives", "flags", "amount"))
    facts = _draw_facts(
        rng,
        n_recs=rng.randint(2, 3),
        n_objectives=0 if mode == "objectives" else rng.randint(2, 3),
        n_flags=0 if mode == "flags" else rng.randint(1, 2),
        allow_hold=mode != "amount",
        ensure_amount=True,
    )
    if mode == "amount":
        index = rng.randrange(len(facts.recommendations))
        recs = list(facts.recommendations)
        recs[index] = recs[index].model_copy(update={"amount": None})
        facts = replace(facts, recommendations=tuple(recs))

    lines = [f"File note for {facts.client}, {facts.date.isoformat()}."]
    lines.append(f"Risk profile agreed as {_risk_phrase(facts.risk)}.")
    if facts.objectives:
        lines.append(f"Objectives: {'; '.join(facts.objectives)}.")
    lines.append("Recommendations:")
    lines.extend(_numbered([_rec_phrase(rec, _money_plain) for rec in facts.recommendations]))
    lines.append(_fee_sentence(facts))
    lines.append(f"Next review in {facts.review_months} months.")
    lines.extend(_FLAG_PROSE[flag] for flag in facts.flags)
    return "\n".join(lines), _record(facts)


def _build_long_context(rng: random.Random) -> tuple[str, AdviceRecord]:
    """The same facts buried in six hundred words of meeting narrative.

    The padding carries no digits, no dollar figures and no percentages, so length is the
    only variable being changed: anything the model gets wrong here it got wrong because the
    facts were far apart, not because the filler introduced a competing number.
    """
    facts = _draw_facts(rng, n_recs=rng.randint(3, 4), n_objectives=3, n_flags=2)
    core = [
        f"Annual review meeting with {facts.client}. Date of record: {facts.date.isoformat()}.",
        (
            f"We revisited the risk profile in some detail and the client confirmed they remain "
            f"comfortable with a {_risk_phrase(facts.risk)} profile."
        ),
        "The client set out what they want the money to do. "
        + " ".join(f"They want to {objective}." for objective in facts.objectives),
        "The recommendations I made, in the order we discussed them, were as follows. "
        + " ".join(_numbered([_rec_phrase(rec, _money_plain) for rec in facts.recommendations])),
        f"{_fee_sentence(facts)} The next review is set for {facts.review_months} months away.",
        " ".join(_FLAG_PROSE[flag] for flag in facts.flags),
    ]

    # The loop always leaves through the break, never by exhausting the catalogue: the
    # filler holds more words than the target on its own, which a test asserts. That is
    # deliberate, because running out would mean repeating a sentence inside one note.
    words = sum(_word_count(part) for part in core)
    padding: list[str] = []
    for sentence in rng.sample(_FILLER, len(_FILLER)):
        if words >= _LONG_CONTEXT_TARGET_WORDS:
            break
        padding.append(sentence)
        words += _word_count(sentence)

    # Stride slicing spreads the padding across the note without another allocation helper;
    # the filler is already shuffled, so the stride does not bias which sentence lands where.
    chunks = [" ".join(padding[offset :: len(core)]) for offset in range(len(core))]
    paragraphs = [f"{part} {chunk}".strip() for part, chunk in zip(core, chunks, strict=True)]
    return "\n\n".join(paragraphs), _record(facts)


def _build_many_items(rng: random.Random) -> tuple[str, AdviceRecord]:
    """Five to eight recommendations, which is where list handling and ordering break.

    Small models truncate long lists, merge adjacent items and reorder them. All three are
    scored as field errors by the verifier, and none of them shows up on a two-item note.
    """
    facts = _draw_facts(rng, n_recs=rng.randint(5, 8), n_objectives=3, n_flags=rng.randint(1, 2))
    lines = [
        f"File note for {facts.client}. Date of record: {facts.date.isoformat()}.",
        f"Risk profile: {_risk_phrase(facts.risk)}.",
        f"Objectives: {'; '.join(facts.objectives)}.",
        f"There are {len(facts.recommendations)} recommendations and the order matters:",
        *_numbered([_rec_phrase(rec, _money_plain) for rec in facts.recommendations]),
        _fee_sentence(facts),
        f"Next review in {facts.review_months} months.",
        *(_FLAG_PROSE[flag] for flag in facts.flags),
    ]
    return "\n".join(lines), _record(facts)


_BUILDERS: dict[Slice, Callable[[random.Random], tuple[str, AdviceRecord]]] = {
    Slice.CLEAN: _build_clean,
    Slice.DISTRACTOR: _build_distractor,
    Slice.MIXED_FORMATS: _build_mixed_formats,
    Slice.ABSENT_FIELDS: _build_absent_fields,
    Slice.LONG_CONTEXT: _build_long_context,
    Slice.MANY_ITEMS: _build_many_items,
}


# --------------------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------------------


def generate_one(note_slice: Slice, rng: random.Random) -> GeneratedNote:
    """Generate a single note for one slice.

    Args:
        note_slice: Which difficulty slice to render.
        rng: The only source of variation. Passing the stream in rather than a seed lets the
            dataset builder derive an independent stream per (split, slice, index) and so
            guarantee disjoint splits by construction.

    Returns:
        The note and the gold record that a correct model would emit for it.
    """
    note, gold = _BUILDERS[note_slice](rng)
    return GeneratedNote(slice=note_slice, note=note, gold=gold)


def generate(
    seed: int = 1, n: int = 12, slices: Sequence[Slice] | None = None
) -> list[GeneratedNote]:
    """Generate `n` notes, cycling through the slices so the mix stays balanced.

    Args:
        seed: Seed for the single `random.Random` stream used for the whole batch.
        n: How many notes to produce.
        slices: Which slices to cycle through; every slice by default.

    Returns:
        The generated notes, in slice-cycling order.

    Raises:
        ValueError: If `n` is negative or `slices` is empty.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    chosen = tuple(Slice) if slices is None else tuple(slices)
    if not chosen:
        raise ValueError("slices must not be empty")
    rng = random.Random(seed)
    return [generate_one(chosen[index % len(chosen)], rng) for index in range(n)]


@lru_cache(maxsize=1)
def _schema_text() -> str:
    """The schema block, rendered once.

    `render_prompt` runs for every example of every epoch and again for every evaluation
    sample, and re-serialising an unchanging schema thousands of times is pure waste.
    `sort_keys` keeps the text stable across pydantic versions that happen to emit keys in a
    different order, which matters because the prompt text feeds the data hash.
    """
    return json.dumps(advice_record_json_schema(), indent=2, sort_keys=True)


_INSTRUCTION = (
    "You are an Australian financial adviser's file note assistant. Read the adviser's note "
    "and return the advice record it describes as one JSON object.\n"
    "\n"
    "Rules:\n"
    "- Return exactly one JSON object and nothing else: no prose, no explanation, no markdown "
    "code fence.\n"
    "- Use only the keys in the schema. Do not invent keys and do not omit required ones.\n"
    "- record_date must be an ISO-8601 date in YYYY-MM-DD form.\n"
    "- Amounts are plain numbers of dollars: no currency symbol, no thousands separator, no "
    "words.\n"
    "- If the note never states something optional, use an empty list or null. Never guess a "
    "value the note does not contain.\n"
    "- If the adviser corrects a figure, record the corrected value and not the superseded "
    "one."
)


def render_prompt(note: str) -> str:
    """Render the instruction shown to the model for one note.

    This is the single most load-bearing string in the package. Supervised fine-tuning,
    preference sampling and evaluation all have to see character-for-character the same
    prompt, or the measured lift is partly a prompt change and the comparison is worthless.
    So it lives here, next to the data that defines the task, and is imported by the trainers
    and the evaluator rather than copied into either.

    The schema is generated from the pydantic model, so the schema the model is asked to
    follow and the schema the verifier enforces cannot drift apart.

    Args:
        note: The adviser's free-text note.

    Returns:
        The full prompt, ending with a cue that leaves the model nothing to do but emit JSON.
    """
    return (
        f"{_INSTRUCTION}\n"
        "\n"
        "JSON Schema:\n"
        f"{_schema_text()}\n"
        "\n"
        "Adviser note:\n"
        f"{note}\n"
        "\n"
        "JSON object:\n"
    )
