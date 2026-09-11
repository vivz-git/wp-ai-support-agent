"""Deterministic guardrail detectors for the AI WhatsApp Support Agent.

Each detector is pure analysis: it takes text (plus, where relevant, bounded
history or structured tool results) and returns a small, frozen, serializable
result. Nothing here calls an LLM, the network, a tool, or WhatsApp, and
nothing here mutates ``ConversationState``. The caller owns state mutation;
``apply_signals_to_flags`` is the one helper that *computes* the updated
``ConversationFlags`` for it, and even that returns a new copy.

Components (Milestone 2, Slice 9):

    InjectionDetector   customer text          -> InjectionResult
    AngerScorer         customer text          -> AngerResult
    RepetitionDetector  history + text         -> RepetitionResult
    HumanRequestDetector customer text         -> HumanRequestResult
    GroundingValidator  model reply + facts    -> GroundingResult

Untrusted input: customer text is data. It is normalized, bounded to
``MAX_ANALYSIS_LENGTH`` characters, matched against fixed regular expressions
with bounded quantifiers, and never executed, evaluated, or logged raw.
Results carry pattern identifiers and codes, never the customer's words.

Not wired into ``app.main`` or the orchestrator; the Milestone 1 webhook is
untouched.
"""

import re
import unicodedata
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from app.agent.state import ConversationFlags, HistoryMessage, MAX_MESSAGE_LENGTH, ToolInvocation
from app.knowledge import KnowledgeBase, Product, build_business_digest

# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

MAX_ANALYSIS_LENGTH = MAX_MESSAGE_LENGTH  # analyse at most one WhatsApp message worth of text
MAX_HISTORY_TURNS_COMPARED = 10  # repetition looks at the last N *customer* turns only
MAX_SENTENCES = 60  # grounding looks at the first N sentences of a reply
MAX_FACTS = 100  # product facts considered by the grounding validator
MAX_VIOLATIONS = 20
MAX_PATTERN_HITS = 20

INJECTION_SUSPECTED_THRESHOLD = 0.5
REPETITION_SIMILARITY_THRESHOLD = 0.8
ANGER_DECAY = 0.5  # how much of last turn's anger carries into this turn's flag

_ZERO_WIDTH_RE = re.compile("[\u200b-\u200f\u2060\ufeff]")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)


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


def _strip_punctuation(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", _PUNCTUATION_RE.sub(" ", text)).strip()


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
# 5. Grounding validation
# ---------------------------------------------------------------------------


class ProductFact(BaseModel):
    """Structured product facts the validator may treat as ground truth.

    ``None`` means "unknown", which is *not* the same as "false": a claim
    about an unknown attribute is reported as unverifiable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sku: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=120)
    price_inr: Optional[float] = Field(None, ge=0)
    in_stock: Optional[bool] = None
    origin: Optional[str] = Field(None, max_length=120)
    tasting_notes: Optional[List[str]] = None

    @classmethod
    def from_product(cls, product: Product) -> "ProductFact":
        return cls(
            sku=product.sku,
            name=product.name,
            price_inr=product.price_inr,
            in_stock=product.in_stock,
            origin=product.origin,
            tasting_notes=list(product.tasting_notes),
        )

    @classmethod
    def from_tool_item(cls, item: Mapping[str, Any]) -> Optional["ProductFact"]:
        """A ``ProductResult``-shaped dict -> fact; ``None`` if it is malformed."""
        sku, name = item.get("sku"), item.get("name")
        if not isinstance(sku, str) or not isinstance(name, str) or not sku or not name:
            return None
        price = item.get("price_inr")
        in_stock = item.get("in_stock")
        origin = item.get("origin")
        notes = item.get("tasting_notes")
        try:
            return cls(
                sku=sku[:64],
                name=name[:120],
                price_inr=float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else None,
                in_stock=in_stock if isinstance(in_stock, bool) else None,
                origin=origin[:120] if isinstance(origin, str) else None,
                tasting_notes=[str(n)[:60] for n in notes[:20]] if isinstance(notes, list) else None,
            )
        except ValueError:
            return None

    def merged_with(self, other: "ProductFact") -> "ProductFact":
        """Fill this fact's unknown attributes from ``other`` (same SKU)."""
        return self.model_copy(
            update={
                "price_inr": self.price_inr if self.price_inr is not None else other.price_inr,
                "in_stock": self.in_stock if self.in_stock is not None else other.in_stock,
                "origin": self.origin if self.origin is not None else other.origin,
                "tasting_notes": self.tasting_notes if self.tasting_notes is not None else other.tasting_notes,
            }
        )


class GroundingResult(BaseModel):
    """Whether a model reply's product claims are supported by known facts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    grounded: bool = True
    violations: List[str] = Field(default_factory=list)
    matched_products: List[str] = Field(default_factory=list, description="SKUs mentioned in the reply")
    reason_codes: List[str] = Field(default_factory=list)
    facts_available: bool = False
    claims_checked: int = Field(0, ge=0)


_PRICE_RE = re.compile(
    r"(?:₹|rs\.?|inr)\s*(\d[\d,]{0,12}(?:\.\d{1,2})?)|(\d[\d,]{0,12}(?:\.\d{1,2})?)\s*(?:inr|rupees|rs\.?)(?![a-z])",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_OUT_OF_STOCK_RE = re.compile(
    r"\b(out of stock|sold out|unavailable|not available|not in stock|currently out|no longer (?:available|stocked))\b"
)
_IN_STOCK_RE = re.compile(
    r"\b(in stock|available now|currently available|is available|are available|available today|"
    r"we have (?:it|them|this|these) (?:in stock|available)|ready to ship)\b"
)
_ORIGIN_CUE_RE = re.compile(
    r"\b(from|grown|sourced|origin|originat\w*|estate|farm\w*|region|beans?|coffee|single origin)\b"
)
_TASTING_CUE_RE = re.compile(
    r"\b(notes?|tasting|flavou?rs?|hints?|tastes?|profile|palate|aroma|finish|acidity|sweetness|body)\b"
)
# The unknown-product check looks at the *original-case* reply for a proper-noun
# product-ish phrase ("our Midnight Roast") and checks it against known aliases.
_PRODUCT_PHRASE_RE = re.compile(
    r"\b(?:our|the)\s+((?:[A-Z][\w'-]{1,30}\s+){1,3}"
    r"(?:Blend|Roast|Reserve|Espresso|Decaf|Subscription|Grinder|Dripper|Scale|Filters?|Beans|Coffee))\b"
)

_COMMON_ORIGINS: FrozenSet[str] = frozenset(
    {
        "ethiopia", "yirgacheffe", "sidamo", "guji", "harrar", "kenya", "kirinyaga", "nyeri",
        "colombia", "huila", "narino", "brazil", "india", "coorg", "chikmagalur", "araku",
        "guatemala", "antigua", "jamaica", "blue mountain", "hawaii", "kona", "vietnam",
        "indonesia", "sumatra", "java", "sulawesi", "bali", "yemen", "rwanda", "burundi",
        "tanzania", "uganda", "peru", "honduras", "costa rica", "panama", "nicaragua",
        "el salvador", "mexico", "papua new guinea", "bolivia", "ecuador", "nepal", "thailand",
    }
)
_COMMON_TASTING_TERMS: FrozenSet[str] = frozenset(
    {
        "chocolate", "dark chocolate", "milk chocolate", "cocoa", "caramel", "toffee", "butterscotch",
        "citrus", "lemon", "lime", "orange", "grapefruit", "bergamot", "floral", "jasmine", "rose",
        "berry", "berries", "blueberry", "strawberry", "raspberry", "cherry", "black currant",
        "blackcurrant", "stone fruit", "apricot", "peach", "plum", "nutty", "hazelnut", "almond",
        "walnut", "peanut", "honey", "vanilla", "molasses", "brown sugar", "maple", "tobacco",
        "smoky", "smoke", "earthy", "spice", "spicy", "cinnamon", "clove", "winey", "wine",
        "tropical", "mango", "pineapple", "papaya", "apple", "grape", "malt", "malty", "cereal",
        "biscuit", "cream", "creamy", "buttery", "herbal", "tea-like", "black tea", "green apple",
    }
)
_ORIGIN_GENERIC_TOKENS: FrozenSet[str] = frozenset({"rotating", "single", "origins", "origin", "and", "the"})


def _product_aliases(name: str) -> List[str]:
    """Normalized name plus short, distinctive prefixes for loose matching."""
    base = re.sub(r"\([^)]{0,60}\)", " ", name)
    normalized = _strip_punctuation(normalize_text(base))
    tokens = normalized.split()
    aliases = [normalized] if normalized else []
    if len(tokens) >= 2:
        aliases.append(" ".join(tokens[:2]))
    if tokens and len(tokens[0]) >= 7:
        aliases.append(tokens[0])
    return [a for a in aliases if a]


def _term_regex(terms: Iterable[str]) -> Optional["re.Pattern[str]"]:
    ordered = sorted({t for t in terms if t}, key=lambda t: (-len(t), t))
    if not ordered:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in ordered) + r")\b")


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

    Facts come from (in order of authority) explicit ``facts``, the
    current-turn ``tool_results`` (``product_lookup`` results *and*
    suggestions), and — only to enrich those by SKU with origin/tasting
    notes, and to whitelist business amounts such as a free-shipping
    threshold — an optional ``KnowledgeBase``. With ``catalog_as_facts=True``
    every catalog product counts as a fact, for deployments whose prompt
    exposes the whole catalog.

    It checks, sentence by sentence: prices, availability, origins and
    tasting notes stated about a product the reply names; unknown
    product-like names; and any price stated when no product facts exist.
    An attribute the facts do not carry is reported as unverifiable rather
    than approved. No LLM, no network.
    """

    def __init__(self, catalog_as_facts: bool = False):
        self._catalog_as_facts = catalog_as_facts

    # -- Fact collection ----------------------------------------------------

    def collect_facts(
        self,
        tool_results: Sequence[Union[ToolInvocation, Mapping[str, Any]]] = (),
        knowledge: Optional[KnowledgeBase] = None,
        facts: Sequence[ProductFact] = (),
    ) -> List[ProductFact]:
        by_sku: Dict[str, ProductFact] = {}

        def add(fact: Optional[ProductFact]) -> None:
            if fact is None or len(by_sku) >= MAX_FACTS and fact.sku not in by_sku:
                return
            by_sku[fact.sku] = fact.merged_with(by_sku[fact.sku]) if fact.sku in by_sku else fact

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
                        add(ProductFact.from_tool_item(item))
        if knowledge is not None:
            for product in knowledge.catalog.products:
                if self._catalog_as_facts or product.sku in by_sku:
                    add(ProductFact.from_product(product))
        return list(by_sku.values())

    @staticmethod
    def business_amounts(knowledge: Optional[KnowledgeBase]) -> Set[float]:
        """Currency amounts stated in the business digest (e.g. a shipping threshold)."""
        if knowledge is None:
            return set()
        return set(_amounts_in(build_business_digest(knowledge)))

    # -- Validation ---------------------------------------------------------

    def validate(
        self,
        response_text: Any,
        tool_results: Sequence[Union[ToolInvocation, Mapping[str, Any]]] = (),
        knowledge: Optional[KnowledgeBase] = None,
        facts: Sequence[ProductFact] = (),
    ) -> GroundingResult:
        raw = response_text[:MAX_ANALYSIS_LENGTH] if isinstance(response_text, str) else ""
        normalized = normalize_text(raw)
        if not normalized:
            return GroundingResult(reason_codes=["empty_response"])

        known = self.collect_facts(tool_results, knowledge, facts)
        allowed_amounts = self.business_amounts(knowledge)
        violations: List[str] = []
        matched_skus: List[str] = []
        claims = 0

        aliases: List[Tuple[str, ProductFact]] = [(alias, fact) for fact in known for alias in _product_aliases(fact.name)]
        alias_regex = _term_regex(alias for alias, _ in aliases)
        origin_terms = set(_COMMON_ORIGINS)
        tasting_terms = set(_COMMON_TASTING_TERMS)
        for fact in known:
            if fact.origin:
                origin_terms.update(
                    t for t in _strip_punctuation(normalize_text(fact.origin)).split()
                    if len(t) >= 4 and t not in _ORIGIN_GENERIC_TOKENS
                )
            for note in fact.tasting_notes or []:
                tasting_terms.add(_strip_punctuation(normalize_text(note)))
        origin_regex = _term_regex(origin_terms)
        tasting_regex = _term_regex(tasting_terms)

        def add_violation(code: str) -> None:
            if code not in violations and len(violations) < MAX_VIOLATIONS:
                violations.append(code)

        def products_in(sentence: str) -> List[ProductFact]:
            if alias_regex is None:
                return []
            found: Dict[str, ProductFact] = {}
            for match in alias_regex.finditer(sentence):
                for alias, fact in aliases:
                    if alias == match.group(1):
                        found.setdefault(fact.sku, fact)
            return list(found.values())

        def without_names(sentence: str) -> str:
            return alias_regex.sub(" ", sentence) if alias_regex is not None else sentence

        sentences = [s for s in _SENTENCE_SPLIT_RE.split(normalized) if s.strip()][:MAX_SENTENCES]
        all_matched: Dict[str, ProductFact] = {}
        for sentence in sentences:
            for fact in products_in(sentence):
                all_matched.setdefault(fact.sku, fact)
        matched_skus = list(all_matched)

        for sentence in sentences:
            in_sentence = products_in(sentence)
            focus = in_sentence or list(all_matched.values())
            body = without_names(sentence)

            # Prices --------------------------------------------------------
            for amount in _amounts_in(sentence):
                claims += 1
                if amount in allowed_amounts:
                    continue
                if not known:
                    add_violation(f"price_without_facts:{_format_amount(amount)}")
                    continue
                candidates = focus or known
                if not any(f.price_inr == amount for f in candidates):
                    add_violation(f"unsupported_price:{_format_amount(amount)}")

            # Availability ----------------------------------------------------
            # A follow-on sentence ("It is out of stock.") is attributed to
            # the reply's product only when exactly one product was named.
            says_out = bool(_OUT_OF_STOCK_RE.search(body))
            says_in = bool(_IN_STOCK_RE.search(body))
            availability_targets = in_sentence or (list(all_matched.values()) if len(all_matched) == 1 else [])
            if says_out or says_in:
                for fact in availability_targets:
                    claims += 1
                    if fact.in_stock is None:
                        add_violation(f"unverifiable_availability:{fact.sku}")
                    elif (says_out and fact.in_stock) or (says_in and not fact.in_stock and not says_out):
                        add_violation(f"unsupported_availability:{fact.sku}")

            if not in_sentence:
                # Origin terms with a cue but no product context: grounded only
                # if some known product actually has that origin.
                if known and origin_regex is not None and _ORIGIN_CUE_RE.search(body):
                    for match in origin_regex.finditer(body):
                        term = match.group(1)
                        claims += 1
                        if not any(f.origin and term in normalize_text(f.origin) for f in known):
                            add_violation(f"unsupported_origin:{term}")
                continue

            # Origin ----------------------------------------------------------
            if origin_regex is not None:
                for match in origin_regex.finditer(body):
                    term = match.group(1)
                    claims += 1
                    if any(f.origin is None for f in in_sentence):
                        add_violation(f"unverifiable_origin:{in_sentence[0].sku}")
                    elif not any(term in normalize_text(f.origin or "") for f in in_sentence):
                        add_violation(f"unsupported_origin:{term}")

            # Tasting notes ---------------------------------------------------
            if tasting_regex is not None:
                terms = [m.group(1) for m in tasting_regex.finditer(body)][:MAX_VIOLATIONS]
                if terms and (_TASTING_CUE_RE.search(body) or len(terms) >= 2):
                    for term in terms:
                        claims += 1
                        if any(f.tasting_notes is None for f in in_sentence):
                            add_violation(f"unverifiable_tasting_note:{in_sentence[0].sku}")
                        elif not any(
                            term in {normalize_text(n) for n in (f.tasting_notes or [])} for f in in_sentence
                        ):
                            add_violation(f"unsupported_tasting_note:{term}")

        # Unknown product-like names (original case) ----------------------------
        if known:
            for match in _PRODUCT_PHRASE_RE.finditer(raw):
                phrase = _strip_punctuation(normalize_text(match.group(1)))
                claims += 1
                if alias_regex is None or not alias_regex.search(phrase):
                    add_violation("unknown_product_name")

        reason_codes = _sorted_unique(v.split(":", 1)[0] for v in violations)
        return GroundingResult(
            grounded=not violations,
            violations=violations,
            matched_products=matched_skus,
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
) -> GuardrailSignals:
    """Run the input-side detectors on one customer message (no grounding yet)."""
    return GuardrailSignals(
        injection=(injection or InjectionDetector()).detect(text),
        anger=(anger or AngerScorer()).score(text),
        repetition=(repetition or RepetitionDetector()).detect(text, history),
        human_request=(human_request or HumanRequestDetector()).detect(text),
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
    "ProductFact",
    "REPETITION_SIMILARITY_THRESHOLD",
    "RepetitionDetector",
    "RepetitionResult",
    "analyze_customer_message",
    "apply_signals_to_flags",
    "normalize_text",
]
