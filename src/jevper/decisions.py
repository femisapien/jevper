"""OpenAI's Decisions request and response schema over this client.

``client.decisions.create(input=…, questions=[…])`` takes the documented ``DecisionRequest`` and
returns its ``DecisionResponse``: ``model``, the ``answers`` in question order, and ``usage``.

Nothing here is a new way to ask a model. The request is translated into the same machine
``system_one`` drives — one prompt per question, put to the model the way the client is configured
to ask — and the answer is translated back into the wire's own fields. That is what makes the
wrapper useful where the endpoint itself does not exist: a provider that has no ``POST /v1/decisions``
(a local server, a gateway, an Anthropic-compatible endpoint) answers the same questions, and moving
to the endpoint later is a client swap rather than a rewrite.

Where the wire carries something jevper cannot ask, the request is refused before it is sent rather
than answered from evidence the model never saw: an ``input_image`` part, for instance, on surfaces
that render text alone. See ``docs/decisions.md`` for the field-by-field mapping and the differences
that remain.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from .errors import InvalidQuestionError, JevperError
from .prompts import render_content
from .reasoning import ReasoningConfig
from .types import (
    CHOICE_MAX_OPTIONS,
    SCORE_MAX_LEVELS,
    SCORE_MIN_LEVELS,
    Api,
    Choice,
    ChoiceAnswer,
    MethodSelection,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
    _validation_message,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .client import AsyncSystemOneClient, Examples, SystemOneClient

_MAX_QUESTIONS = 200


class _InputText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["input_text"] = "input_text"
    text: str


class _InputImage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["input_image"] = "input_image"
    image_url: str
    detail: str | None = None


class _InputMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user"] = "user"
    content: str | list[Annotated[_InputText | _InputImage, Field(discriminator="type")]]
    type: Literal["message"] = "message"


class _ChoiceOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str | bool
    description: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _string_or_boolean(cls, value: Any) -> Any:
        """The wire's ``str | boolean``, refused rather than coerced into it.

        Pydantic reads a number as a boolean in lax mode, so an ``int`` option left to the union
        validates as ``True``/``False`` and the answer carries a value the caller never wrote — and
        an option key, which is what that value becomes here, is the last place a silent type change
        belongs. Anything else is refused by name, one question-indexed line.
        """
        if isinstance(value, (str, bool)):
            return value
        raise ValueError(
            f"must be a string or a boolean, got {type(value).__name__} {value!r}"
        )


class _ScoreLevel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    description: str | None = None


class _PredicateQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["predicate"] = "predicate"
    name: str | None = None
    instructions: str


class _ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    name: str | None = None
    instructions: str
    choices: list[_ChoiceOption] = Field(min_length=2, max_length=CHOICE_MAX_OPTIONS)


class _ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    name: str | None = None
    instructions: str
    levels: list[_ScoreLevel] = Field(min_length=SCORE_MIN_LEVELS, max_length=SCORE_MAX_LEVELS)


_QUESTION = TypeAdapter(
    Annotated[
        _PredicateQuestion | _ChoiceQuestion | _ScoreQuestion, Field(discriminator="type")
    ]
)
_INPUT_MESSAGE = TypeAdapter(_InputMessage)


def _validate(adapter: TypeAdapter[Any], payload: Any, *, where: str, error: type[JevperError]) -> Any:
    """One payload through one adapter, reported the way the rest of jevper reports local mistakes.

    ``_validation_message`` is the same formatter the question models use, so a request refused here
    reads like a question refused by ``system_one``: ``field.path: what is wrong``.
    """
    try:
        return adapter.validate_python(payload)
    except ValidationError as exc:
        raise error(f"{where}: {_validation_message(exc)}") from None


@dataclass(frozen=True)
class _Asked:
    """One question of the request, plus the parts of it the wire echoes but jevper does not ask with.

    ``values`` is the caller's own option value per option key, and ``labels`` their own score level
    labels, lowest first — the echo the wrapper can produce because it still holds the request. See
    ``decision_for`` and ``to_decision`` for what happens without them.
    """

    name: str | None
    question: Question
    values: dict[str, str | bool]
    labels: tuple[str, ...]


def _message_text(message: _InputMessage, index: int) -> str:
    if isinstance(message.content, str):
        return message.content
    texts: list[str] = []
    for part in message.content:
        if isinstance(part, _InputImage):
            raise JevperError(
                f"input message {index}: input_image parts are not supported — jevper answers from "
                "text on every surface it speaks (chat_completions, responses, messages), so an "
                "image would be evidence the model never saw"
            )
        texts.append(part.text)
    return "\n".join(texts)


def _state_from_input(input: Any) -> Any:
    """The wire's ``input`` as the ``state`` jevper asks about.

    A string is a string either way. Messages collapse to text turns, because a single text block is
    a state shape every surface renders identically — and the wire allows no role but ``user``, which
    jevper's chat-message state does carry, so the two agree on what this means.
    """
    if isinstance(input, str):
        return input
    if isinstance(input, (str, bytes, bytearray, memoryview)) or not isinstance(input, Sequence):
        raise JevperError(
            f"input must be a string or a list of user messages, got {type(input).__name__}"
        )
    if not input:
        raise JevperError(
            "input messages must be a non-empty list: jevper cannot carry an empty conversation"
        )
    messages: list[dict[str, str]] = []
    for index, raw in enumerate(input):
        message = _validate(_INPUT_MESSAGE, raw, where=f"input message {index}", error=JevperError)
        messages.append({"role": "user", "content": _message_text(message, index)})
    return messages


def _choice_asked(question: _ChoiceQuestion, where: str) -> _Asked:
    """A choice question as jevper asks it, remembering which option key stands for which value.

    jevper's option keys are text, which the wire's values are not: it allows a string and a boolean
    ("a string and a boolean with the same text are distinct"), and two options that would collapse
    onto one key are two options this wrapper cannot ask about, so they are refused rather than
    silently merged.
    """
    values: dict[str, str | bool] = {}
    criteria: dict[str, str | None] = {}
    for option in question.choices:
        key = option.value if isinstance(option.value, str) else ("true" if option.value else "false")
        if key in values:
            raise InvalidQuestionError(
                f"{where}: choices carry both {values[key]!r} and {option.value!r}, which jevper "
                "cannot tell apart — its option keys are text, so those two would be one option; "
                "give them distinct values"
            )
        values[key] = option.value
        criteria[key] = option.description
    return _Asked(
        name=question.name,
        question=Choice(instructions=question.instructions, criteria=criteria),
        values=values,
        labels=(),
    )


def _questions_from(questions: Any) -> tuple[_Asked, ...]:
    if isinstance(questions, (str, bytes)) or not isinstance(questions, Sequence):
        raise InvalidQuestionError(
            f"questions must be a list of question objects, got {type(questions).__name__}"
        )
    if not 1 <= len(questions) <= _MAX_QUESTIONS:
        raise InvalidQuestionError(
            f"questions must carry 1..{_MAX_QUESTIONS} questions, got {len(questions)}"
        )
    asked: list[_Asked] = []
    for index, raw in enumerate(questions):
        where = f"question {index}"
        parsed = _validate(_QUESTION, raw, where=where, error=InvalidQuestionError)
        if isinstance(parsed, _ChoiceQuestion):
            asked.append(_choice_asked(parsed, where))
        elif isinstance(parsed, _ScoreQuestion):
            asked.append(
                _Asked(
                    name=parsed.name,
                    question=Score(
                        instructions=parsed.instructions,
                        criteria=[
                            f"{level.label}: {level.description}" if level.description else level.label
                            for level in parsed.levels
                        ],
                    ),
                    values={},
                    labels=tuple(level.label for level in parsed.levels),
                )
            )
        else:
            asked.append(
                _Asked(
                    name=parsed.name,
                    question=Noul(instructions=parsed.instructions),
                    values={},
                    labels=(),
                )
            )
    return tuple(asked)


def _build_request(input: Any, questions: Any) -> tuple[Any, tuple[_Asked, ...]]:
    """The whole request validated: the state to ask about, and the questions in ask order.

    jevper's ids here are positional, because the wire's ``name`` is optional and need not be unique:
    two unnamed questions, or two questions named alike, are questions the wire answers separately,
    and keying them by name would invent a collision it does not have. The caller's own name travels
    beside the question instead, and comes back on the answer.
    """
    return _state_from_input(input), _questions_from(questions)


def _unique_ids_by_name(asked: Sequence[_Asked]) -> dict[str, str]:
    """The id each uniquely-named question travels as; a name used twice is left out.

    ``examples`` is read by id, and this call's ids are positional, so a name is the only key a
    caller's own examples can be recognised by. A name two questions share identifies neither, so it
    is not a key this translation can move.
    """
    names = [item.name for item in asked if item.name is not None]
    return {
        item.name: str(index)
        for index, item in enumerate(asked)
        if item.name is not None and names.count(item.name) == 1
    }


def _client_examples(configured: Examples, asked: Sequence[_Asked]) -> Examples:
    """The examples the client was built with, keyed by name like the per-call ones.

    ``system_one`` reads an examples mapping by the id a question travels as, and a client built with
    ``SystemOneClient(..., examples={"intent": […]})`` — which guides that question on ``system_one`` —
    would find nothing matching here and drop them in silence, because a wrapper call's ids are
    positional. So the names are translated to this call's ids, and every other key is left as it is:
    a client may carry examples for questions this call does not ask, or for ids of its own.
    """
    if not isinstance(configured, Mapping) or not configured:
        return configured
    by_name = _unique_ids_by_name(asked)
    return {
        by_name.get(str(key), key) if isinstance(key, str) else key: value
        for key, value in configured.items()
    }


def _examples_by_name(examples: Examples, asked: Sequence[_Asked]) -> Examples:
    """The caller's examples, keyed the way ``system_one`` reads them: by the id a question travels as.

    The ids are positional here, so a mapping is keyed by question *name* — the one identifier the
    caller wrote — and this is where a name becomes an id. A sequence is handed through untouched: it
    applies to every question, as ``system_one`` documents.
    """
    if not isinstance(examples, Mapping):
        return examples
    by_name: dict[str, str] = {}
    for index, item in enumerate(asked):
        if item.name is None:
            continue
        if item.name in by_name:
            raise InvalidQuestionError(
                f"examples is keyed by question name, and the name {item.name!r} names more than one "
                "question; pass a sequence of Example objects instead, which applies to every question"
            )
        by_name[item.name] = str(index)
    unknown = sorted(str(key) for key in examples if key not in by_name)
    if unknown:
        raise InvalidQuestionError(
            "examples is keyed by question name; no question is named "
            + ", ".join(repr(name) for name in unknown)
        )
    return {by_name[key]: value for key, value in examples.items()}


def _examples_for(
    client: SystemOneClient | AsyncSystemOneClient, examples: Examples, asked: Sequence[_Asked]
) -> Examples:
    """The examples this call asks with: the caller's per-call mapping over the client's own.

    ``system_one`` reads the per-call examples before the constructor's for every question id, which is
    the order these two arrive in once the ids have been re-keyed — so the precedence a caller knows
    from ``system_one`` is the precedence here, id by id. A *sequence* of examples speaks for every
    question and therefore outranks any id-keyed mapping, as it does there; and the client's own mapping
    arrives under this call's ids even when the caller passed none, because otherwise it would arrive
    keyed by a name no question of this call travels as, and be dropped without a word.
    """
    per_call = _examples_by_name(examples, asked)
    configured = _client_examples(client.examples, asked)
    if not isinstance(configured, Mapping) or not configured:
        return per_call
    if isinstance(per_call, Mapping):
        return {**configured, **per_call}
    if per_call:
        return per_call
    return configured


class DecisionChoiceProbability(BaseModel):
    """One option's probability, with the value the caller wrote for it."""

    value: str | bool
    probability: float = Field(allow_inf_nan=False)


class DecisionScoreProbability(BaseModel):
    """One level's probability, with its index and the caller's own label for it."""

    value: int
    label: str
    probability: float = Field(allow_inf_nan=False)


class DecisionPredicateAnswer(BaseModel):
    type: Literal["predicate"] = "predicate"
    name: str | None
    """The question's name, ``None`` when it was asked without one — the wire echoes the request."""
    probability: float = Field(allow_inf_nan=False)
    """The Jev ``noul``, passed through: the wire's type is an unbounded number, and the service's
    own regression output is not a number jevper would round on its way to the caller."""


class DecisionChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    name: str | None
    choice: str | bool
    probabilities: list[DecisionChoiceProbability]
    confidence: float = Field(allow_inf_nan=False)
    """The answer's own confidence, unbounded as the wire's ``number`` is: what jevper produced is
    bounded where it is computed (``ChoiceAnswer`` and ``ScoreAnswer`` hold a confidence in ``[0, 1]``,
    as ``normalize``'s formulas allow only that), so a decision from this wrapper is in range either
    way, while a number a service reported is the service's own and not jevper's to refuse — the rule
    every probability beside it already follows."""


class DecisionScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    name: str | None
    score: float = Field(allow_inf_nan=False)
    probabilities: list[DecisionScoreProbability]
    confidence: float = Field(allow_inf_nan=False)
    """Unbounded like the choice answer's, and for the same reason: see ``DecisionChoiceAnswer``."""


class DecisionRefusalAnswer(BaseModel):
    """The wire's ``refusal``: the model declined this one question.

    jevper never emits one — a refusal it met raises ``ModelRefusalError``, because on a prompt
    surface a refusal is the model failing the call rather than answering a question of it. The type
    completes the documented union so a payload from the endpoint itself still validates.
    """

    type: Literal["refusal"] = "refusal"
    name: str | None


DecisionAnswer = Annotated[
    DecisionPredicateAnswer | DecisionChoiceAnswer | DecisionScoreAnswer | DecisionRefusalAnswer,
    Field(discriminator="type"),
]


class DecisionInputTokensDetails(BaseModel):
    cached_tokens: int | None = None
    """Prompt tokens the provider read from its cache; ``None`` when it reported nothing."""
    cache_write_tokens: int | None = None
    """Always ``None``: jevper reads no cache-write counter from any surface it speaks, and a
    reported ``0`` would claim the provider counted one."""


class DecisionOutputTokensDetails(BaseModel):
    reasoning_tokens: int | None = None


class DecisionUsage(BaseModel):
    input_tokens: int | None
    input_tokens_details: DecisionInputTokensDetails
    output_tokens: int | None
    output_tokens_details: DecisionOutputTokensDetails
    total_tokens: int | None
    """``input_tokens + output_tokens``, or ``None`` when either is unknown: a sum of unknowns is not
    a number, and ``usage`` leaves a count ``None`` exactly when a constituent call omitted it."""


class Decision(BaseModel):
    """One Decisions response: ``model``, the answers in question order, and the usage.

    Dumps carry those three keys and nothing else — ``raw`` is excluded from every dump, so a payload
    here is the shape the endpoint itself would have returned.
    """

    model: str
    answers: list[DecisionAnswer]
    usage: DecisionUsage
    raw: SystemOneResponse | None = Field(default=None, exclude=True)
    """The jevper response behind this one, for the fields the wire has no room for: ``system_one``'s
    own ``reasoning`` and ``debug``, and each answer's ``laya`` extras where the service sent them."""


def _decision_usage(usage: Usage) -> DecisionUsage:
    total: int | None = None
    if usage.input_tokens is not None and usage.output_tokens is not None:
        total = usage.input_tokens + usage.output_tokens
    return DecisionUsage(
        input_tokens=usage.input_tokens,
        input_tokens_details=DecisionInputTokensDetails(
            cached_tokens=usage.cached_tokens, cache_write_tokens=None
        ),
        output_tokens=usage.output_tokens,
        output_tokens_details=DecisionOutputTokensDetails(reasoning_tokens=usage.reasoning_tokens),
        total_tokens=total,
    )


def _choice_answer(
    answer: ChoiceAnswer, name: str | None, values: Mapping[str, str | bool]
) -> DecisionChoiceAnswer:
    """One choice answer, with the caller's value for every option key.

    ``values`` covers every key of ``answer.probabilities`` — both producers build it from the
    question's own criteria, and a response keeps exactly those keys — and the chosen option is one of
    them wherever jevper read the answer itself: a service's choice must name an option the question
    offered, and its distribution must carry every one of them. A response assembled by hand is bound
    by neither rule, though (the answer models do not tie ``choice`` to ``probabilities``), so the
    chosen key is handed back as it arrived when there is no value recorded for it — the rule the
    legend fallback follows — instead of a lookup that would raise on a caller's own object.
    """
    return DecisionChoiceAnswer(
        name=name,
        choice=values.get(answer.choice, answer.choice),
        probabilities=[
            DecisionChoiceProbability(value=values[key], probability=probability)
            for key, probability in answer.probabilities.items()
        ],
        confidence=answer.confidence,
    )


def _legend_label(answer: ScoreAnswer, level: int) -> str:
    """A level's own text from the response's legend, or the index when the legend does not name it.

    ``legend`` is what a response remembers of the rubric it was asked with. A response that reports a
    level without one is still answering with that level, and the index is the spelling the answer
    uses for it everywhere else (``parse_answer`` keys levels by index, and the debug readout spells
    them the same way), so the conversion stays total rather than raising on a missing key.
    """
    if level in answer.legend:
        return render_content(answer.legend[level])
    return str(level)


def _score_answer(
    answer: ScoreAnswer, name: str | None, labels: Sequence[str] | None
) -> DecisionScoreAnswer:
    return DecisionScoreAnswer(
        name=name,
        score=answer.score,
        probabilities=[
            DecisionScoreProbability(
                value=level,
                label=labels[level] if labels is not None else _legend_label(answer, level),
                probability=probability,
            )
            for level, probability in answer.probabilities.items()
        ],
        confidence=answer.confidence,
    )


def decision_for(response: SystemOneResponse, asked: Sequence[_Asked]) -> Decision:
    """The wire's answer to a request this module still holds, so the echo is the caller's own.

    A choice's value is what the caller wrote for that option — a string or a boolean — and a score
    level's label is the caller's own, rather than the key or the rendered criteria ``to_decision``
    has to fall back on. Requests and answers are matched by position: ``_build_request`` numbered
    the questions, and a response answers its questions in the order they were asked.
    """
    answers: list[DecisionAnswer] = []
    for index, item in enumerate(asked):
        answer = response.answers[str(index)]
        if isinstance(answer, NoulAnswer):
            answers.append(DecisionPredicateAnswer(name=item.name, probability=answer.noul))
        elif isinstance(answer, ChoiceAnswer):
            answers.append(_choice_answer(answer, item.name, item.values))
        else:
            answers.append(_score_answer(answer, item.name, item.labels))
    return Decision(
        model=response.model,
        answers=answers,
        usage=_decision_usage(response.usage),
        raw=response,
    )


def to_decision(response: SystemOneResponse) -> Decision:
    """A ``system_one`` response as the Decisions schema, with only the echo the response carries.

    A question's ``name`` is its id, a choice's value is the option key it was asked with, and a score
    level's label is that level's criteria entry as text — which is everything a response remembers
    about the request that produced it. ``client.decisions.create`` echoes the caller's own names,
    labels and values instead, because it still holds the request; this is what a caller has after
    ``system_one`` returned.
    """
    answers: list[DecisionAnswer] = []
    for question_id, answer in response.answers.items():
        if isinstance(answer, NoulAnswer):
            answers.append(DecisionPredicateAnswer(name=question_id, probability=answer.noul))
        elif isinstance(answer, ChoiceAnswer):
            answers.append(
                _choice_answer(answer, question_id, {key: key for key in answer.probabilities})
            )
        else:
            answers.append(_score_answer(answer, question_id, None))
    return Decision(
        model=response.model,
        answers=answers,
        usage=_decision_usage(response.usage),
        raw=response,
    )


class Decisions:
    """``client.decisions.create(...)``: OpenAI's Decisions schema over this client.

    The request is the documented one — ``input``, ``questions``, ``model`` — and the response is the
    documented one. How the question reaches the model is the client's own: ``method=``, ``api=``,
    ``reasoning=``, ``temperature=``, ``examples=``, ``prompt_cache_key=``, ``extras=`` and
    ``keep_alive=`` are ``system_one``'s options, here per call, and mean what they mean there.

    Two things the wire carries have no counterpart here. An ``input_image`` part is refused before a
    request is made, because no jevper surface renders images and the model would be answering without
    the evidence the caller pointed at; a refusal is a ``ModelRefusalError`` rather than the wire's
    ``refusal`` answer, because on a prompt surface a refusal is the model failing the call.
    ``safety_identifier`` is not a parameter at all — an unknown keyword is a ``TypeError``, as
    everywhere else in jevper.
    """

    def __init__(self, client: SystemOneClient) -> None:
        self._client = client

    def create(
        self,
        *,
        input: str | Sequence[Mapping[str, Any]],
        questions: Sequence[Mapping[str, Any]],
        examples: Examples = (),
        model: str | None = None,
        method: MethodSelection | None = None,
        api: Api | None = None,
        reasoning: ReasoningConfig | None = None,
        temperature: float | None = None,
        prompt_cache_key: str | None = None,
        extras: Sequence[str] = (),
        keep_alive: str | int | None = None,
    ) -> Decision:
        """Answer ``questions`` about ``input``, answered and returned in the Decisions schema.

        Answers come back in question order, each carrying the name its question was asked with —
        ``None`` when it was asked without one. Everything beyond ``input`` and ``questions`` is
        ``system_one``'s own option, with the same meaning and the same default.
        """
        state, asked = _build_request(input, questions)
        response = self._client.system_one(
            state=state,
            questions={str(index): item.question for index, item in enumerate(asked)},
            examples=_examples_for(self._client, examples, asked),
            model=model,
            method=method,
            api=api,
            reasoning=reasoning,
            temperature=temperature,
            prompt_cache_key=prompt_cache_key,
            extras=extras,
            keep_alive=keep_alive,
        )
        return decision_for(response, asked)


class AsyncDecisions:
    """``client.decisions.create(...)`` for ``AsyncSystemOneClient``; identical in every other way."""

    def __init__(self, client: AsyncSystemOneClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        input: str | Sequence[Mapping[str, Any]],
        questions: Sequence[Mapping[str, Any]],
        examples: Examples = (),
        model: str | None = None,
        method: MethodSelection | None = None,
        api: Api | None = None,
        reasoning: ReasoningConfig | None = None,
        temperature: float | None = None,
        prompt_cache_key: str | None = None,
        extras: Sequence[str] = (),
        keep_alive: str | int | None = None,
    ) -> Decision:
        """Answer ``questions`` about ``input``, answered and returned in the Decisions schema."""
        state, asked = _build_request(input, questions)
        response = await self._client.system_one(
            state=state,
            questions={str(index): item.question for index, item in enumerate(asked)},
            examples=_examples_for(self._client, examples, asked),
            model=model,
            method=method,
            api=api,
            reasoning=reasoning,
            temperature=temperature,
            prompt_cache_key=prompt_cache_key,
            extras=extras,
            keep_alive=keep_alive,
        )
        return decision_for(response, asked)
