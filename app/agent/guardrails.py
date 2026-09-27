"""Deterministic guardrail detectors for the AI WhatsApp Support Agent.

Each detector is pure analysis: it takes text (plus, where relevant, bounded
history or structured tool results) and returns a small, frozen, serializable
result. Nothing here calls an LLM, the network, a tool, or WhatsApp, and
nothing here mutates ``ConversationState``. The caller owns state mutation;
``apply_signals_to_flags`` is the one helper that *computes* the updated
``ConversationFlags`` for it, and even that returns a new copy.

Components:

    InjectionDetector    customer text          -> InjectionResult
    AngerScorer          customer text          -> AngerResult
    RepetitionDetector   history + text         -> RepetitionResult
    HumanRequestDetector customer text          -> HumanRequestResult
    EmergencyDetector    customer text          -> EmergencyResult
    GroundingValidator   model reply + facts    -> GroundingResult

Untrusted input: customer text is data. It is normalized, bounded to
``MAX_ANALYSIS_LENGTH`` characters, matched against fixed regular expressions
with bounded quantifiers, and never executed, evaluated, or logged raw.
Results carry pattern identifiers and codes, never the customer's words.

Hindi: Python's ``\\w`` and ``\\b`` do not treat Devanagari vowel signs as
word characters, so Devanagari phrases are matched as plain substrings of the
NFKC-normalized text, never with ``\\b`` boundaries.
"""

import re
import unicodedata
from typing import Any, Dict, FrozenSet, Iterable, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from app.agent.state import ConversationFlags, HistoryMessage, MAX_MESSAGE_LENGTH, ToolInvocation
from app.knowledge import KnowledgeBase, Service, normalize_match_text, strip_punctuation

# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

MAX_ANALYSIS_LENGTH = MAX_MESSAGE_LENGTH  # analyse at most one WhatsApp message worth of text
MAX_HISTORY_TURNS_COMPARED = 10  # repetition looks at the last N *customer* turns only
MAX_SENTENCES = 60  # grounding looks at the first N sentences of a reply
MAX_FACTS = 100  # service facts considered by the grounding validator
MAX_VIOLATIONS = 20
MAX_PATTERN_HITS = 20

INJECTION_SUSPECTED_THRESHOLD = 0.5
REPETITION_SIMILARITY_THRESHOLD = 0.8
ANGER_DECAY = 0.5  # how much of last turn's anger carries into this turn's flag

_ZERO_WIDTH_RE = re.compile("[\u200b-\u200f\u2060\ufeff]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: Any) -> str:
    """Bounded, NFKC-normalized, lower-cased, whitespace-collapsed text.

    Non-string input becomes an empty string; text is cut at
    ``MAX_ANALYSIS_LENGTH`` *before* any pattern matching so every detector
    is bounded by construction.
    """
    if not isinstance(text, str):
        return ""
    bounded = text[:MAX_ANALYSIS_LENGTH]
    normalized = unicodedata.normalize("NFKC", bounded)
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip().lower()


_strip_punctuation = strip_punctuation


def _sorted_unique(values: Iterable[str]) -> List[str]:
    return sorted(set(values))


# ---------------------------------------------------------------------------
# 1. Injection detection
# ---------------------------------------------------------------------------

# (pattern_id, reason_code, weight, regex). Bounded ``.{0,N}`` gaps only; no
# nested quantifiers, so matching stays linear on adversarial input.
_INJECTION_PATTERNS: Tuple[Tuple[str, str, float, "re.Pattern[str]"], ...] = tuple(
    (pattern_id, reason_code, weight, re.compile(regex, re.IGNORECASE))
    for pattern_id, reason_code, weight, regex in (
        (
            "ignore_previous_instructions",
            "instruction_override",
            0.6,
            r"\b(ignore|disregard|forget|override|bypass|drop)\b.{0,20}?"
            r"\b(previous|prior|above|earlier|all|your|initial|original|system|these|those)\b.{0,20}?"
            r"\b(instructions?|prompts?|rules?|guidelines?|directives?|programming)\b",
        ),
        (
            "new_instructions",
            "instruction_override",
            0.5,
            r"\b(new|real|actual|true) (instructions?|rules?|system prompt)\b.{0,10}?(:|are|is)",
        ),
        (
            "reveal_system_prompt",
            "prompt_disclosure",
            0.6,
            r"\b(reveal|show|print|display|repeat|output|tell|give|share|what|dump|expose|leak|paste|recite)\b.{0,30}?"
            r"\b(system|hidden|secret|initial|original|internal|developer|confidential)\b.{0,10}?"
            r"\b(prompt|instructions?|message|rules?|config(?:uration)?)\b",
        ),
        (
            "act_as_privileged_role",
            "role_override",
            0.5,
            r"\b(act|behave|respond|operate|function) as (?:a |an |the )?"
            r"(developer|admin(?:istrator)?|root|system|jailbroken|unrestricted|unfiltered|sysadmin)\b",
        ),
        (
            "privileged_mode",
            "role_override",
            0.5,
            r"\b(developer|debug|god|admin|jailbreak|dan|maintenance|sudo) mode\b",
        ),
        (
            "you_are_now",
            "role_override",
            0.5,
            r"\byou are now (?:the |a |an )?"
            r"(system|developer|admin(?:istrator)?|root|dan|unrestricted|unfiltered|free|jailbroken)\b",
        ),
        (
            "pretend_privileged_role",
            "role_override",
            0.5,
            r"\b(pretend|imagine|assume) (?:that )?(?:you are|you're|to be) (?:the |a |an )?"
            r"(system|developer|admin(?:istrator)?|root|sysadmin)\b",
        ),
        ("jailbreak", "role_override", 0.5, r"\bjailbreak\w*\b"),
        (
            "system_override",
            "role_override",
            0.5,
            r"\b(system|safety|security) override\b|\boverride (?:mode|protocol|safety)\b",
        ),
        (
            "from_now_on",
            "role_override",
            0.3,
            r"\bfrom now on(?:,)? you (?:are|will|must|should)\b",
        ),
        (
            "reveal_credentials",
            "secret_request",
            0.7,
            r"\b(reveal|show|give|tell|share|print|leak|expose|send|what is|what's|whats|dump|display|paste|need)\b.{0,30}?"
            r"\b(api[ _-]?keys?|access tokens?|secret keys?|passwords?|credentials?|bearer tokens?|"
            r"private keys?|auth(?:entication)? tokens?|environment variables?|env vars?|\.env|"
            r"connection strings?|webhook (?:secret|token)|verify token)\b",
        ),
        (
            "which_model_or_provider",
            "internal_disclosure",
            0.4,
            r"\b(which|what|whose) (llm|language model|ai model|model|provider|api|backend|vendor) "
            r"(are you|do you use|is (?:this|behind|powering)|powers you|runs? you|are you built on)\b",
        ),
        (
            "named_provider_probe",
            "internal_disclosure",
            0.4,
            r"\b(are you|is this|using|running on|powered by|built on) "
            r"(groq|openai|chatgpt|gpt[- ]?\d|llama|gemini|claude|anthropic|mistral|deepseek)\b",
        ),
        (
            "reveal_tool_definitions",
            "internal_disclosure",
            0.4,
            r"\b(tool definitions?|function definitions?|function schemas?|tool schemas?|tool names?|"
            r"functions you can call|tools you (?:have|can (?:call|use)|are given)|available functions?)\b",
        ),
    )
)

# Reason codes that mean the customer asked for internal data rather than
# merely trying to change behaviour.
_INTERNAL_DATA_CODES: FrozenSet[str] = frozenset({"secret_request", "prompt_disclosure", "internal_disclosure"})


class InjectionResult(BaseModel):
    """Prompt-injection signal for one customer message.

    ``hits`` are pattern identifiers, never customer text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    suspected: bool = False
    hits: List[str] = Field(default_factory=list)
    score: float = Field(0.0, ge=0.0, le=1.0)
    reason_codes: List[str] = Field(default_factory=list)
    secrets_requested: bool = False
    internal_data_requested: bool = False

    @property
    def hit_count(self) -> int:
        return len(self.hits)


class InjectionDetector:
    """Deterministic, offline prompt-injection detector.

    Not a content moderator: it only recognises attempts to override the
    instruction hierarchy, assume a privileged role, or extract prompts,
    credentials, or provider/tool internals. A weak signal (e.g. asking
    which model powers the bot) is recorded as a hit but does not by itself
    cross ``INJECTION_SUSPECTED_THRESHOLD``.
    """

    def __init__(self, threshold: float = INJECTION_SUSPECTED_THRESHOLD):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self._threshold = threshold

    def detect(self, text: Any) -> InjectionResult:
        normalized = normalize_text(text)
        if not normalized:
            return InjectionResult()

        hits: List[str] = []
        reason_codes: Set[str] = set()
        score = 0.0
        for pattern_id, reason_code, weight, regex in _INJECTION_PATTERNS:
            if regex.search(normalized):
                hits.append(pattern_id)
                reason_codes.add(reason_code)
                score += weight
                if len(hits) >= MAX_PATTERN_HITS:
                    break

        score = min(1.0, round(score, 3))
        suspected = score >= self._threshold
        return InjectionResult(
            suspected=suspected,
            hits=hits,  # pattern-table order: deterministic
            score=score,
            reason_codes=_sorted_unique(reason_codes),
            secrets_requested="secret_request" in reason_codes,
            internal_data_requested=bool(reason_codes & _INTERNAL_DATA_CODES),
        )


# ---------------------------------------------------------------------------
# 2. Anger scoring
# ---------------------------------------------------------------------------

_PROFANITY_RE = re.compile(
    r"\b(damn|dammit|hell|crap|shit\w*|fuck\w*|wtf|bullshit|bloody|bastards?|assholes?|"
    r"idiots?|stupid|morons?|screw(?:ed)?|sucks?|piss(?:ed)?|bollocks|frigging|effing)\b"
)
_ANGER_PHRASE_RE = re.compile(
    r"\b(this is ridiculous|ridiculous|terrible service|worst service|horrible service|awful service|"
    r"useless|fix this now|fix it now|i am angry|i'm angry|im angry|i am furious|furious|"
    r"i've had enough|ive had enough|had enough|fed up|unacceptable|pathetic|disgusting|"
    r"sick of|waste of (?:my )?time|scam(?:mers?)?|outrageous|never again|worst experience|"
    r"absolutely (?:not|no) (?:ok|okay|acceptable)|so frustrated|very frustrated|extremely frustrated)\b"
)
_COMPLAINT_RE = re.compile(
    r"\b(not working|doesn't work|does not work|isn't working|still waiting|still not|still no|"
    r"not received|never arrived|hasn't arrived|has not arrived|didn't arrive|wrong order|wrong item|"
    r"damaged|broken|leaking|stale|refund|complaint|complain|no one (?:replied|responded|answered)|"
    r"nobody (?:replied|responded|answered)|third time|fourth time|again and again|"
    r"i already told you|how many times|no response|ignored|charged twice|overcharged)\b"
)
_EXCLAMATION_RUN_RE = re.compile(r"!{2,}")
_CAPS_WORD_RE = re.compile(r"\b[A-Z]{3,}\b")

_PROFANITY_FIRST_WEIGHT = 0.3
_PROFANITY_EXTRA_WEIGHT = 0.1
_PROFANITY_CAP = 0.5
_ANGER_PHRASE_WEIGHT = 0.3
_ANGER_PHRASE_CAP = 0.6
_COMPLAINT_WEIGHT = 0.15
_COMPLAINT_CAP = 0.3
_EXCLAMATION_RUN_WEIGHT = 0.15
_EXCLAMATION_MANY_WEIGHT = 0.1
_ALL_CAPS_WEIGHT = 0.25
_CAPS_WORDS_WEIGHT = 0.1
_MAX_COUNTED_MATCHES = 10


class AngerResult(BaseModel):
    """Weighted anger/frustration signal in ``[0, 1]``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: float = Field(0.0, ge=0.0, le=1.0)
    hit_count: int = Field(0, ge=0)
    reason_codes: List[str] = Field(default_factory=list)

    @property
    def complaint_language(self) -> bool:
        return "complaint_language" in self.reason_codes


class AngerScorer:
    """Deterministic anger scoring from weighted lexical and typographic signals.

    A single mild negative word ("bad", "disappointing") scores 0. One
    profanity alone scores 0.3 — well under the escalation threshold — so
    a customer is never treated as abusive for one strong word.
    """

    def score(self, text: Any) -> AngerResult:
        normalized = normalize_text(text)
        if not normalized:
            return AngerResult()

        raw = text[:MAX_ANALYSIS_LENGTH] if isinstance(text, str) else ""
        score = 0.0
        hit_count = 0
        reason_codes: Set[str] = set()

        profanity = min(len(_PROFANITY_RE.findall(normalized)), _MAX_COUNTED_MATCHES)
        if profanity:
            hit_count += profanity
            reason_codes.add("profanity")
            score += min(_PROFANITY_CAP, _PROFANITY_FIRST_WEIGHT + (profanity - 1) * _PROFANITY_EXTRA_WEIGHT)

        anger_phrases = min(len(_ANGER_PHRASE_RE.findall(normalized)), _MAX_COUNTED_MATCHES)
        if anger_phrases:
            hit_count += anger_phrases
            reason_codes.add("anger_phrase")
            score += min(_ANGER_PHRASE_CAP, anger_phrases * _ANGER_PHRASE_WEIGHT)

        complaints = min(len(_COMPLAINT_RE.findall(normalized)), _MAX_COUNTED_MATCHES)
        if complaints:
            hit_count += complaints
            reason_codes.add("complaint_language")
            score += min(_COMPLAINT_CAP, complaints * _COMPLAINT_WEIGHT)

        exclamation_runs = len(_EXCLAMATION_RUN_RE.findall(normalized))
        exclamations = normalized.count("!")
        if exclamation_runs:
            hit_count += 1
            reason_codes.add("repeated_exclamation")
            score += _EXCLAMATION_RUN_WEIGHT
        if exclamations >= 3:
            hit_count += 1
            reason_codes.add("many_exclamations")
            score += _EXCLAMATION_MANY_WEIGHT

        letters = [c for c in raw if c.isalpha()]
        if len(letters) >= 8:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio >= 0.7:
                hit_count += 1
                reason_codes.add("all_caps")
                score += _ALL_CAPS_WEIGHT
            elif len(_CAPS_WORD_RE.findall(raw)) >= 2:
                hit_count += 1
                reason_codes.add("caps_words")
                score += _CAPS_WORDS_WEIGHT

        return AngerResult(
            score=min(1.0, round(score, 3)),
            hit_count=hit_count,
            reason_codes=_sorted_unique(reason_codes),
        )


# ---------------------------------------------------------------------------
# 3. Repetition detection
# ---------------------------------------------------------------------------

_STOPWORDS: FrozenSet[str] = frozenset(
    """
    a an the is are am was were be been do does did can could would will shall should
    of to for in on at by with from about as into and or but if then so than too very
    i me my mine we us our you your yours it its this that these those there here
    what whats which who whom how when where why please pls plz hi hello hey ok okay
    thanks thank u ya yeah yes no not tell know want need just also again still much
    get give would like any some
    """.split()
)

_MIN_CONTENT_TOKENS = 2
_MAX_COMPARE_LENGTH = 1000

HistoryLike = Union[HistoryMessage, str, Mapping[str, Any]]


class RepetitionResult(BaseModel):
    """Whether the current message repeats an earlier customer turn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repeated: bool = False
    similarity_score: float = Field(0.0, ge=0.0, le=1.0)
    matched_turn: Optional[int] = Field(None, ge=0)
    reason: str = "no_match"


def _content_tokens(normalized: str) -> List[str]:
    tokens = _strip_punctuation(normalized)[:_MAX_COMPARE_LENGTH].split()
    return [t for t in tokens if t not in _STOPWORDS]


def _dice(a: Sequence[str], b: Sequence[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return 0.0
    return round(2.0 * len(set_a & set_b) / (len(set_a) + len(set_b)), 3)


def _iter_user_history(history: Sequence[HistoryLike]) -> List[Tuple[int, str]]:
    """``(turn, content)`` for the last ``MAX_HISTORY_TURNS_COMPARED`` customer turns."""
    turns: List[Tuple[int, str]] = []
    for index, item in enumerate(history):
        if isinstance(item, HistoryMessage):
            if item.role != "user":
                continue
            turns.append((item.turn, item.content))
        elif isinstance(item, str):
            turns.append((index, item))
        elif isinstance(item, Mapping):
            if item.get("role", "user") != "user":
                continue
            turn = item.get("turn", index)
            content = item.get("content", "")
            turns.append((turn if isinstance(turn, int) and turn >= 0 else index, str(content)))
    return turns[-MAX_HISTORY_TURNS_COMPARED:]


class RepetitionDetector:
    """Lightweight lexical repetition detector over bounded customer history.

    Similarity is 1.0 for an exact match after normalization, else the Dice
    coefficient over content tokens (stopwords removed). Very short messages
    ("ok", "yes") never count as a repeated *question*.
    """

    def __init__(self, threshold: float = REPETITION_SIMILARITY_THRESHOLD):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self._threshold = threshold

    def detect(self, text: Any, history: Sequence[HistoryLike] = ()) -> RepetitionResult:
        current = _strip_punctuation(normalize_text(text))[:_MAX_COMPARE_LENGTH]
        if not current:
            return RepetitionResult(reason="empty")
        current_tokens = _content_tokens(current)
        if len(current_tokens) < _MIN_CONTENT_TOKENS:
            return RepetitionResult(reason="too_short")

        previous = _iter_user_history(history)
        if not previous:
            return RepetitionResult(reason="no_history")

        best_score = 0.0
        best_turn: Optional[int] = None
        best_reason = "no_match"
        for turn, content in previous:  # oldest -> newest; ties resolve to the newest
            candidate = _strip_punctuation(normalize_text(content))[:_MAX_COMPARE_LENGTH]
            if not candidate:
                continue
            if candidate == current:
                score, reason = 1.0, "exact_match"
            else:
                score = _dice(current_tokens, _content_tokens(candidate))
                reason = "near_match" if score >= self._threshold else "no_match"
            if score >= best_score:
                best_score, best_turn, best_reason = score, turn, reason

        repeated = best_score >= self._threshold
        return RepetitionResult(
            repeated=repeated,
            similarity_score=best_score,
            matched_turn=best_turn if repeated else None,
            reason=best_reason if repeated else "no_match",
        )


# ---------------------------------------------------------------------------
# 4. Human-request detection (input to escalation rule A)
# ---------------------------------------------------------------------------

_HUMAN_REQUEST_RE = re.compile(
    r"\b(?:(?:talk|speak|chat) (?:to|with) (?:a |an |the )?(?:human|person|real person|someone|somebody|agent|"
    r"representative|manager|your team|support team|staff|operator)|"
    r"(?:human|real|live|actual) (?:agent|person|being|support|operator|representative)|"
    r"agent please|human please|"
    r"connect me (?:to|with)|transfer me (?:to|with)?|put me through|"
    r"(?:i want|i need|get me|give me) (?:a |an )?(?:human|person|real person|agent|representative|manager|someone)|"
    r"not a bot|no bots?|stop the bot|"
    r"customer (?:care|service|support) (?:executive|representative|agent|team|number)|"
    r"call me (?:back|please|now)|escalate (?:this|me|it))\b"
)


class HumanRequestResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    requested: bool = False
    reason_codes: List[str] = Field(default_factory=list)


class HumanRequestDetector:
    """Detects an explicit request to talk to a person. Deterministic, offline."""

    def detect(self, text: Any) -> HumanRequestResult:
        normalized = normalize_text(text)
        if normalized and _HUMAN_REQUEST_RE.search(normalized):
            return HumanRequestResult(requested=True, reason_codes=["explicit_human_request"])
        return HumanRequestResult()


# ---------------------------------------------------------------------------
# 5. Dental emergency detection (pain, bleeding, swelling, trauma)
# ---------------------------------------------------------------------------

PatientLanguage = Literal["en", "hi", "hinglish"]

_DEVANAGARI_RE = re.compile("[ऀ-ॿ]")
# Common Hindi words written in Roman script. Two or more in one message is
# a reliable Hinglish signal; single words ("hai") appear in English chats too.
_HINGLISH_MARKERS: FrozenSet[str] = frozenset(
    """
    hai hain he ho hoga hota raha rahi rha rhi mein mai mera meri mere mujhe hum aap ka ki ke kya kyu kyun
    nahi nhi bahut bohot bahot bhi aur kar karo kare karna se toh gaya gayi gya gyi tha thi chahiye kitna
    kitne kitni lagega lagegi lagta daant dant dard kal parso subah shaam abhi jaldi ji haan theek
    kahan kaha kidhar kab kaise karwana sakta sakti chahta chahti wala wali
    """.split()
)


def detect_language(text: Any) -> PatientLanguage:
    """Which language/script the patient wrote in: Devanagari Hindi, Hinglish, or English."""
    normalized = normalize_text(text)
    if _DEVANAGARI_RE.search(normalized):
        return "hi"
    tokens = _strip_punctuation(normalized).split()
    if sum(1 for token in tokens if token in _HINGLISH_MARKERS) >= 2:
        return "hinglish"
    return "en"


def _compile_substrings(phrases: Iterable[str]) -> "re.Pattern[str]":
    """Devanagari phrases: NFKC-normalized plain substrings, no ``\\b``."""
    normalized = sorted({unicodedata.normalize("NFKC", p).lower() for p in phrases}, key=lambda p: (-len(p), p))
    return re.compile("|".join(re.escape(p) for p in normalized))


# (code, script, pattern). ``latin`` patterns run on the normalized text with
# ``\b`` boundaries; ``devanagari`` patterns are substring alternations.
# Questions about a procedure ("is RCT painful?", "RCT mein dard hota hai
# kya?") deliberately do not match: the patterns need a first-person or
# present-tense report ("I have tooth pain", "dard ho raha hai").
_EMERGENCY_PATTERNS: Tuple[Tuple[str, str, "re.Pattern[str]"], ...] = (
    # --- English ---------------------------------------------------------
    ("pain", "en", re.compile(r"\b(?:tooth ?ache|jaw ?ache)s?\b")),
    (
        "pain",
        "en",
        re.compile(
            r"\b(?:severe|bad|terrible|horrible|unbearable|extreme|intense|sharp|throbbing|constant|excruciating)"
            r" (?:tooth |teeth |jaw |gum |dental )?(?:pain|ache)\b"
        ),
    ),
    (
        "pain",
        "en",
        re.compile(
            r"\b(?:i have|i've got|i got|i'm having|im having|having|got) (?:a |some |so much |a lot of |lots of )?"
            r"(?:tooth|teeth|jaw|gum|dental|mouth) (?:pain|ache)\b"
        ),
    ),
    (
        "pain",
        "en",
        re.compile(
            r"\b(?:tooth|teeth|jaw|gums?|mouth|molar|wisdom tooth) (?:is |are )?(?:really |very |so |still )?"
            r"(?:hurting|paining|aching|killing me|throbbing)\b"
        ),
    ),
    ("pain", "en", re.compile(r"\b(?:my|the) (?:tooth|teeth|jaw|gums?) (?:really |still )?hurts?\b")),
    ("pain", "en", re.compile(r"\b(?:i am|i'm|im) in (?:so much |a lot of |severe |terrible )?pain\b")),
    ("pain", "en", re.compile(r"\bcan'?t (?:sleep|eat|chew|bite)\b.{0,30}\bpain\b|\bpain\b.{0,30}\bcan'?t (?:sleep|eat|chew|bite)\b")),
    ("bleeding", "en", re.compile(r"\b(?:bleeding|bleeds? (?:a lot|non ?stop|continuously))\b")),
    (
        "bleeding",
        "en",
        re.compile(r"\b(?:lot of|lots of|so much|too much) blood\b|\bblood (?:is )?(?:coming|flowing|not stopping|won'?t stop)\b|\bspitting blood\b"),
    ),
    ("swelling", "en", re.compile(r"\b(?:swollen|abscess|pus)\b")),
    (
        "swelling",
        "en",
        re.compile(
            r"\b(?:my|have|has|got|there is|there's|face|cheek|jaw|gum|gums) (?:a |some |big |lot of )?swelling\b"
            r"|\bswelling (?:in|on|of) (?:my|the)\b|\bswelling (?:is )?(?:increasing|getting worse|spreading)\b"
        ),
    ),
    (
        "trauma",
        "en",
        re.compile(
            r"\b(?:broke|broken|chipped|cracked|knocked out|knocked|lost|fractured) (?:my |his |her |a |the |one |two )?"
            r"(?:front |back )?(?:tooth|teeth|jaw)\b"
        ),
    ),
    (
        "trauma",
        "en",
        re.compile(r"\b(?:tooth|teeth) (?:got |is |are |has |have )?(?:broke|broken|chipped|cracked|knocked out|fell out|came out|fallen out|loose)\b"),
    ),
    ("trauma", "en", re.compile(r"\b(?:accident|fell|injury|injured|punched|hit)\b.{0,40}\b(?:tooth|teeth|mouth|jaw|face|lip)\b")),
    ("trauma", "en", re.compile(r"\bjaw (?:is )?(?:locked|stuck|dislocated)\b|\bcan'?t (?:open|close) my (?:mouth|jaw)\b")),
    # --- Hinglish (Hindi in Roman script) ----------------------------------
    ("pain", "hinglish", re.compile(r"\b(?:bahut|bohot|bohut|bahot|bht|tez|tej|zyada|jyada|bhayankar|asahniya) dard\b")),
    ("pain", "hinglish", re.compile(r"\bdard (?:ho )?(?:raha|rha|rahi|rhi)\b|\bdard (?:hai|he|h)\b|\bdard se\b|\bdard ke maa?re\b")),
    ("pain", "hinglish", re.compile(r"\b(?:daant|dant|daat|daad|jabde|masude|masudo|masoodhe) (?:me|mein|mai|main|mei) (?:bahut |bohot |tez )?dard\b")),
    ("pain", "hinglish", re.compile(r"\bdukh (?:raha|rha|rahi|rhi)\b")),
    ("bleeding", "hinglish", re.compile(r"\b(?:khoon|khun) (?:aa|nikal|beh|bah) ?(?:raha|rha|rahi|rhi)\b|\b(?:khoon|khun) (?:band|ruk) (?:nahi|nhi)\b")),
    ("bleeding", "hinglish", re.compile(r"\b(?:bahut|bohot|zyada|jyada) (?:khoon|khun)\b")),
    ("swelling", "hinglish", re.compile(r"\b(?:sujan|soojan|sujaan|soojhan|sujhan|mawad|mavaad)\b")),
    ("swelling", "hinglish", re.compile(r"\b(?:suj|sooj|sujh|phool) (?:gaya|gayi|gya|gyi|gaye|raha|rahi)\b")),
    ("trauma", "hinglish", re.compile(r"\b(?:daant|dant|daat|daanth)\b.{0,20}\b(?:toot|tut|tuut|hil|gir) ?(?:gaya|gya|gaye|gye|gayi|raha|rha)\b")),
    ("trauma", "hinglish", re.compile(r"\bchot (?:lag|lagi|aayi|aai)\b")),
    ("trauma", "hinglish", re.compile(r"\b(?:accident|gir gaya|gir gayi|gir gya|gira)\b.{0,40}\b(?:daant|dant|muh|munh|mooh|jabda)\b")),
    # --- Hindi (Devanagari) -------------------------------------------------
    (
        "pain",
        "hi",
        _compile_substrings(
            [
                "बहुत दर्द", "तेज दर्द", "तेज़ दर्द", "असहनीय दर्द", "दर्द हो रहा", "दर्द हो रही", "दर्द है", "दर्द से",
                "दांत में दर्द", "दाँत में दर्द", "दाढ़ में दर्द", "मसूड़े में दर्द", "दुख रहा", "दुख रही",
            ]
        ),
    ),
    (
        "bleeding",
        "hi",
        _compile_substrings(["खून आ रहा", "खून निकल रहा", "खून बह रहा", "खून बंद नहीं", "खून रुक नहीं", "बहुत खून", "ब्लीडिंग"]),
    ),
    ("swelling", "hi", _compile_substrings(["सूजन", "सूज गया", "सूज गई", "सूज गयी", "फूल गया", "फूल गई", "मवाद"])),
    (
        "trauma",
        "hi",
        _compile_substrings(
            ["दांत टूट", "दाँत टूट", "टूट गया", "टूट गई", "टूट गयी", "चोट", "गिर गया", "गिर गई", "गिर गयी", "दांत हिल", "दाँत हिल", "एक्सीडेंट"]
        ),
    ),
)


class EmergencyResult(BaseModel):
    """Whether the patient reports a possible dental emergency.

    ``reason_codes`` are symptom categories (``pain``, ``bleeding``,
    ``swelling``, ``trauma``), never the patient's words. ``language`` is the
    patient's language/script, used to pick the deterministic reply.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    detected: bool = False
    hit_count: int = Field(0, ge=0)
    reason_codes: List[str] = Field(default_factory=list)
    language: PatientLanguage = "en"


class EmergencyDetector:
    """Flags pain, bleeding, swelling or dental trauma in English, Hindi and Hinglish.

    Deterministic and offline. Deliberately biased towards escalating: a
    false positive costs a staff callback, a false negative leaves a patient
    in pain talking to a bot.
    """

    def detect(self, text: Any) -> EmergencyResult:
        normalized = normalize_text(text)
        if not normalized:
            return EmergencyResult()

        reason_codes: Set[str] = set()
        scripts: Set[str] = set()
        hits = 0
        for code, script, pattern in _EMERGENCY_PATTERNS:
            if pattern.search(normalized):
                hits += 1
                reason_codes.add(code)
                scripts.add(script)
                if hits >= MAX_PATTERN_HITS:
                    break

        language = detect_language(normalized)
        if language == "en" and "hinglish" in scripts:
            language = "hinglish"
        return EmergencyResult(
            detected=bool(reason_codes),
            hit_count=hits,
            reason_codes=_sorted_unique(reason_codes),
            language=language,
        )


# ---------------------------------------------------------------------------
# 6. Grounding validation
# ---------------------------------------------------------------------------


_PARENTHETICAL_RE = re.compile(r"\(([^)]{0,60})\)")


class ServiceFact(BaseModel):
    """A clinic service and its price range, treated as ground truth.

    ``None`` prices mean "unknown", which is not the same as "any price": a
    price claim about a service with an unknown range is unverifiable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=120)
    aliases: List[str] = Field(default_factory=list, description="Normalized match terms, longest first")
    price_min_inr: Optional[float] = Field(None, ge=0)
    price_max_inr: Optional[float] = Field(None, ge=0)

    @classmethod
    def from_service(cls, service: Service) -> "ServiceFact":
        return cls(
            id=service.id,
            name=service.name,
            aliases=service.match_aliases(),
            price_min_inr=service.price_min_inr,
            price_max_inr=service.price_max_inr,
        )

    @classmethod
    def from_tool_item(cls, item: Mapping[str, Any]) -> Optional["ServiceFact"]:
        """A ``ClinicFact`` dict of kind ``service`` -> fact; ``None`` otherwise."""
        if item.get("kind") != "service":
            return None
        fact_id, title = item.get("id"), item.get("title")
        if not isinstance(fact_id, str) or not isinstance(title, str) or not fact_id or not title:
            return None

        def price(key: str) -> Optional[float]:
            value = item.get(key)
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

        title = title[:120]
        # "Root canal treatment (RCT)" -> the full title, "root canal treatment" and "rct".
        terms = {title, _PARENTHETICAL_RE.sub(" ", title), *_PARENTHETICAL_RE.findall(title)}
        aliases = sorted({a for a in (normalize_match_text(t) for t in terms) if a}, key=lambda t: (-len(t), t))
        try:
            return cls(
                id=fact_id[:64],
                name=title,
                aliases=aliases,
                price_min_inr=price("price_min_inr"),
                price_max_inr=price("price_max_inr"),
            )
        except ValueError:
            return None

    def merged_with(self, other: "ServiceFact") -> "ServiceFact":
        """Fill unknown prices and add aliases from ``other`` (same ID)."""
        aliases = sorted(set(self.aliases) | set(other.aliases), key=lambda t: (-len(t), t))
        return self.model_copy(
            update={
                "aliases": aliases,
                "price_min_inr": self.price_min_inr if self.price_min_inr is not None else other.price_min_inr,
                "price_max_inr": self.price_max_inr if self.price_max_inr is not None else other.price_max_inr,
            }
        )

    def allows(self, amount: float) -> Optional[bool]:
        """Whether ``amount`` is inside this service's range; ``None`` if unknown."""
        if self.price_min_inr is None or self.price_max_inr is None:
            return None
        return self.price_min_inr <= amount <= self.price_max_inr


class GroundingResult(BaseModel):
    """Whether a model reply's clinic claims are supported by known facts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    grounded: bool = True
    violations: List[str] = Field(default_factory=list)
    matched_services: List[str] = Field(default_factory=list, description="Service IDs mentioned in the reply")
    reason_codes: List[str] = Field(default_factory=list)
    facts_available: bool = False
    claims_checked: int = Field(0, ge=0)


_CURRENCY = r"(?:₹|rs\.?|inr|rupees?|rupaye|rupaiye|रुपये|रुपए|रु\.?)"
_CURRENCY_AFTER = r"(?:inr|rupees?|rs\.?|rupaye|rupaiye|रुपये|रुपए|रु)"
_AMOUNT = r"(\d[\d,]{0,12}(?:\.\d{1,2})?)"
_PRICE_RE = re.compile(
    rf"{_CURRENCY}\s*{_AMOUNT}|{_AMOUNT}\s*{_CURRENCY_AFTER}(?![a-z])",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+|\n+")
# Roman-script mentions only: a Devanagari transliteration cannot be compared
# against the English names on file, so it is not checked.
_DENTIST_MENTION_RE = re.compile(r"\bdr\.?\s+([a-z][a-z'-]{1,30})")
# Medicines, doses and home remedies. The assistant must never suggest these;
# naming them at all in a reply is treated as unverified medical advice.
_MEDICAL_ADVICE_RE = re.compile(
    r"\b(?:paracetamol|acetaminophen|ibuprofen|aspirin|diclofenac|nimesulide|amoxicillin|augmentin|"
    r"metronidazole|azithromycin|combiflam|dolo|crocin|calpol|antibiotics?|painkillers?|pain killers?|"
    r"analgesics?|clove oil|salt water rinse|warm salt water|\d+\s?mg)\b"
    r"|पैरासिटामोल|एंटीबायोटिक|पेनकिलर|दर्द की गोली|दर्द निवारक"
)


def _parse_amount(raw: str) -> Optional[float]:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _amounts_in(text: str) -> List[float]:
    amounts: List[float] = []
    for match in _PRICE_RE.finditer(text):
        value = _parse_amount(match.group(1) or match.group(2) or "")
        if value is not None:
            amounts.append(value)
        if len(amounts) >= MAX_VIOLATIONS:
            break
    return amounts


def _format_amount(value: float) -> str:
    return f"{value:g}"


class GroundingValidator:
    """Conservative, deterministic claim checker for outgoing replies.

    Service facts come from explicit ``facts``, the current-turn
    ``tool_results`` (``clinic_faq_lookup`` results *and* suggestions), and
    the ``KnowledgeBase`` (every clinic service is trusted data). Checks:

    - every rupee amount must fall inside the range of the service(s) the
      sentence names, or of some known service if the sentence names none;
    - a price with no service facts at all is unverifiable;
    - any "Dr. <name>" must be one of the clinic's dentists (when known);
    - medicine names, doses and home remedies are never allowed.

    No LLM, no network.
    """

    # -- Fact collection ----------------------------------------------------

    def collect_facts(
        self,
        tool_results: Sequence[Union[ToolInvocation, Mapping[str, Any]]] = (),
        knowledge: Optional[KnowledgeBase] = None,
        facts: Sequence[ServiceFact] = (),
    ) -> List[ServiceFact]:
        by_id: Dict[str, ServiceFact] = {}

        def add(fact: Optional[ServiceFact]) -> None:
            if fact is None or len(by_id) >= MAX_FACTS and fact.id not in by_id:
                return
            by_id[fact.id] = fact.merged_with(by_id[fact.id]) if fact.id in by_id else fact

        for fact in facts:
            add(fact)
        for invocation in tool_results:
            result = invocation.result if isinstance(invocation, ToolInvocation) else invocation
            if not isinstance(result, Mapping):
                continue
            for key in ("results", "suggestions"):
                items = result.get(key)
                if not isinstance(items, list):
                    continue
                for item in items[:MAX_FACTS]:
                    if isinstance(item, Mapping):
                        add(ServiceFact.from_tool_item(item))
        if knowledge is not None:
            for service in knowledge.clinic.services:
                add(ServiceFact.from_service(service))
        return list(by_id.values())

    @staticmethod
    def _dentist_name_terms(
        tool_results: Sequence[Union[ToolInvocation, Mapping[str, Any]]], knowledge: Optional[KnowledgeBase]
    ) -> Optional[Set[str]]:
        """Normalized first/last names of known dentists, or ``None`` if none are known."""
        names: List[str] = []
        if knowledge is not None:
            names.extend(d.name for d in knowledge.clinic.dentists)
        for invocation in tool_results:
            result = invocation.result if isinstance(invocation, ToolInvocation) else invocation
            if isinstance(result, Mapping) and isinstance(result.get("results"), list):
                names.extend(
                    item["title"]
                    for item in result["results"]
                    if isinstance(item, Mapping) and item.get("kind") == "dentist" and isinstance(item.get("title"), str)
                )
        if not names:
            return None
        terms: Set[str] = set()
        for name in names:
            terms.update(normalize_match_text(name).split()[1:])  # drop the "dr" prefix
        return terms

    # -- Validation ---------------------------------------------------------

    def validate(
        self,
        response_text: Any,
        tool_results: Sequence[Union[ToolInvocation, Mapping[str, Any]]] = (),
        knowledge: Optional[KnowledgeBase] = None,
        facts: Sequence[ServiceFact] = (),
    ) -> GroundingResult:
        raw = response_text[:MAX_ANALYSIS_LENGTH] if isinstance(response_text, str) else ""
        normalized = normalize_text(raw)
        if not normalized:
            return GroundingResult(reason_codes=["empty_response"])

        known = self.collect_facts(tool_results, knowledge, facts)
        violations: List[str] = []
        claims = 0

        def add_violation(code: str) -> None:
            if code not in violations and len(violations) < MAX_VIOLATIONS:
                violations.append(code)

        def services_in(sentence: str) -> List[ServiceFact]:
            padded = f" {normalize_match_text(sentence)} "
            # A trailing "s" covers plurals ("root canals", "check ups").
            return [f for f in known if any(f" {a} " in padded or f" {a}s " in padded for a in f.aliases)]

        sentences = [s for s in _SENTENCE_SPLIT_RE.split(normalized) if s.strip()][:MAX_SENTENCES]
        all_matched: Dict[str, ServiceFact] = {}
        for sentence in sentences:
            for fact in services_in(sentence):
                all_matched.setdefault(fact.id, fact)

        # Prices ------------------------------------------------------------
        for sentence in sentences:
            focus = services_in(sentence) or list(all_matched.values())
            for amount in _amounts_in(sentence):
                claims += 1
                if not known:
                    add_violation(f"price_without_facts:{_format_amount(amount)}")
                    continue
                verdicts = [f.allows(amount) for f in (focus or known)]
                if any(v is True for v in verdicts):
                    continue
                if all(v is None for v in verdicts):
                    add_violation(f"unverifiable_price:{_format_amount(amount)}")
                else:
                    add_violation(f"unsupported_price:{_format_amount(amount)}")

        # Dentists ----------------------------------------------------------
        dentist_terms = self._dentist_name_terms(tool_results, knowledge)
        if dentist_terms is not None:
            for match in _DENTIST_MENTION_RE.finditer(normalized):
                claims += 1
                mentioned = normalize_match_text(match.group(1))
                if mentioned and mentioned.split()[0] not in dentist_terms:
                    add_violation("unknown_dentist")

        # Medical advice ----------------------------------------------------
        if _MEDICAL_ADVICE_RE.search(normalized):
            claims += 1
            add_violation("medical_advice")

        reason_codes = _sorted_unique(v.split(":", 1)[0] for v in violations)
        return GroundingResult(
            grounded=not violations,
            violations=violations,
            matched_services=list(all_matched),
            reason_codes=reason_codes,
            facts_available=bool(known),
            claims_checked=claims,
        )


# ---------------------------------------------------------------------------
# Signal bundle + flag helper
# ---------------------------------------------------------------------------


class GuardrailSignals(BaseModel):
    """Everything the escalation policy needs from the detectors for one turn.

    Any component may be ``None`` when its detector did not run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    injection: Optional[InjectionResult] = None
    anger: Optional[AngerResult] = None
    repetition: Optional[RepetitionResult] = None
    human_request: Optional[HumanRequestResult] = None
    emergency: Optional[EmergencyResult] = None
    grounding: Optional[GroundingResult] = None

    def with_grounding(self, grounding: GroundingResult) -> "GuardrailSignals":
        return self.model_copy(update={"grounding": grounding})


def analyze_customer_message(
    text: Any,
    history: Sequence[HistoryLike] = (),
    injection: Optional[InjectionDetector] = None,
    anger: Optional[AngerScorer] = None,
    repetition: Optional[RepetitionDetector] = None,
    human_request: Optional[HumanRequestDetector] = None,
    emergency: Optional[EmergencyDetector] = None,
) -> GuardrailSignals:
    """Run the input-side detectors on one customer message (no grounding yet)."""
    return GuardrailSignals(
        injection=(injection or InjectionDetector()).detect(text),
        anger=(anger or AngerScorer()).score(text),
        repetition=(repetition or RepetitionDetector()).detect(text, history),
        human_request=(human_request or HumanRequestDetector()).detect(text),
        emergency=(emergency or EmergencyDetector()).detect(text),
    )


def apply_signals_to_flags(flags: ConversationFlags, signals: GuardrailSignals) -> ConversationFlags:
    """Compute the ``ConversationFlags`` after this turn's signals. Pure.

    Returns a new object; the caller assigns it (``state.flags = ...``).
    Semantics:
    - ``injection_suspected`` is sticky once set; ``injection_hits`` counts
      matched patterns across all suspected turns.
    - ``anger_score`` cools: ``max(this turn, ANGER_DECAY * previous)``.
    - ``repeated_question_count`` counts *consecutive* repeats; a new
      question resets it.
    - ``grounding_violations`` counts replies the validator rejected.
    ``unanswered_asks``, ``declines``, ``off_topic_count`` and
    ``tool_failures_this_turn`` are owned by other components and untouched.
    """
    updates: Dict[str, Any] = {}
    if signals.injection is not None and signals.injection.suspected:
        updates["injection_suspected"] = True
        updates["injection_hits"] = flags.injection_hits + len(signals.injection.hits)
    if signals.anger is not None:
        updates["anger_score"] = min(1.0, max(signals.anger.score, round(flags.anger_score * ANGER_DECAY, 3)))
    if signals.repetition is not None and signals.repetition.reason not in ("empty", "too_short"):
        updates["repeated_question_count"] = flags.repeated_question_count + 1 if signals.repetition.repeated else 0
    if signals.grounding is not None and not signals.grounding.grounded:
        updates["grounding_violations"] = flags.grounding_violations + 1
    return ConversationFlags.model_validate({**flags.model_dump(), **updates})


__all__ = [
    "ANGER_DECAY",
    "AngerResult",
    "AngerScorer",
    "EmergencyDetector",
    "EmergencyResult",
    "GroundingResult",
    "GroundingValidator",
    "GuardrailSignals",
    "HumanRequestDetector",
    "HumanRequestResult",
    "INJECTION_SUSPECTED_THRESHOLD",
    "InjectionDetector",
    "InjectionResult",
    "MAX_ANALYSIS_LENGTH",
    "MAX_HISTORY_TURNS_COMPARED",
    "PatientLanguage",
    "REPETITION_SIMILARITY_THRESHOLD",
    "RepetitionDetector",
    "RepetitionResult",
    "ServiceFact",
    "analyze_customer_message",
    "apply_signals_to_flags",
    "detect_language",
    "normalize_text",
]
