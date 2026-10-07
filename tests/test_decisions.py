"""The Decisions wrapper: OpenAI's request and response schema over jevper's own machinery.

Four things are pinned here. The request the wrapper takes is the documented one — a string or user
messages, and ``predicate``/``choice``/``score`` questions with the wire's limits — and a request it
cannot carry is refused before anything is sent, rather than answered from evidence the model never
saw. The response is the documented one too: ``model``, ``answers`` in question order, ``usage``, and
nothing else in a dump. The echo is the caller's own: an option's string or boolean value, a level's
label, a question's name including a null one. And an answer reads the same on every surface, because
the wrapper is a translation of ``system_one`` rather than a second way to ask a model.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from typing import Any

import pytest
from fakes import (
    StubServer,
    anthropic_client,
    async_openai_client,
    chat_body,
    messages_body,
    openai_client,
    responses_body,
)

from jevper import (
    AsyncSystemOneClient,
    ChoiceAnswer,
    Decision,
    DecisionRefusalAnswer,
    Example,
    InvalidQuestionError,
    JevperError,
    ModelRefusalError,
    NoulAnswer,
    ScoreAnswer,
    SystemOneClient,
    SystemOneResponse,
    Usage,
)

CHOICE = {
    "type": "choice",
    "name": "department",
    "instructions": "Which department should handle this complaint?",
    "choices": [{"value": "billing"}, {"value": "sales"}],
}
PREDICATE = {"type": "predicate", "name": "billing", "instructions": "Is this about a charge?"}
SCORE = {
    "type": "score",
    "name": "severity",
    "instructions": "How bad is it?",
    "levels": [{"label": "low"}, {"label": "high"}],
}


def client_for(stub: StubServer, **kwargs: Any) -> SystemOneClient:
    kwargs.setdefault("api", "chat_completions")
    kwargs.setdefault("method", "structured")
    return SystemOneClient(openai_client(stub), model="m", **kwargs)


def option_keys(body: Mapping[str, Any]) -> list[str]:
    """The option keys the request asked with, in the order it asked them."""
    schema = body["response_format"]["json_schema"]["schema"]
    return list(schema["properties"]["probabilities"]["properties"])


def answering_with(answers: Mapping[str, Any]) -> Callable[[dict[str, Any]], tuple[int, Any]]:
    """A System One script: the service's own answer objects, keyed by the id the request used."""

    def answer(_body: dict[str, Any]) -> tuple[int, Any]:
        return 200, {"model": "m", "answers": dict(answers), "usage": {"input_tokens": 4, "output_tokens": 1}}

    return answer


def chat_script(
    *, probabilities: Mapping[str, float] | None = None, noul: float = 0.93
) -> Callable[[dict[str, Any]], tuple[int, Any]]:
    """A chat script that answers whatever schema arrives: a noul value, or a distribution.

    The distribution is read back in the keys the request itself used, so a script is written once
    and works for a choice, a score and a boolean option alike — which is also how it proves the
    wrapper asked with the keys it did.
    """

    def answer(body: dict[str, Any]) -> tuple[int, Any]:
        schema = body["response_format"]["json_schema"]
        if schema["name"] == "jevper_noul":
            return 200, chat_body(content=json.dumps({"noul": noul}))
        keys = option_keys(body)
        given = probabilities or {}
        distribution = {key: given.get(key, 1.0 if index == 0 else 0.0) for index, key in enumerate(keys)}
        return 200, chat_body(content=json.dumps({"probabilities": distribution}))

    return answer


def test_the_wrapper_answers_in_the_decisions_schema_on_chat_completions(stub_server: Any) -> None:
    stub = stub_server(chat=chat_script(probabilities={"billing": 0.9, "sales": 0.1}))
    client = client_for(stub)
    decision = client.decisions.create(
        input="I was charged twice for my order.",
        questions=[{**CHOICE, "choices": [{"value": "billing"}, {"value": "sales"}]}],
    )

    assert json.loads(decision.model_dump_json()) == {
        "model": "m",
        "answers": [
            {
                "type": "choice",
                "name": "department",
                "choice": "billing",
                "probabilities": [
                    {"value": "billing", "probability": 0.9},
                    {"value": "sales", "probability": 0.1},
                ],
                "confidence": 0.8,
            }
        ],
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": None, "cache_write_tokens": None},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 13,
        },
    }
    # The response behind it is kept, and kept out of every dump.
    assert decision.raw is not None
    assert "raw" not in decision.model_dump()

    # It left as a jevper call: the schema asks for the option keys, and the state as a document.
    schema = stub.requests[0]["response_format"]["json_schema"]
    assert set(schema["schema"]["properties"]["probabilities"]["properties"]) == {"billing", "sales"}
    assert "I was charged twice for my order." in json.dumps(stub.requests[0]["messages"])


def test_the_wrapper_echoes_boolean_choice_values(stub_server: Any) -> None:
    stub = stub_server(chat=chat_script(probabilities={"true": 0.8, "false": 0.2}))
    decision = client_for(stub).decisions.create(
        input="I was charged twice.",
        questions=[
            {
                "type": "choice",
                "name": "billing",
                "instructions": "Is this a billing question?",
                "choices": [{"value": True, "description": "Payments."}, {"value": False}],
            }
        ],
    )

    (answer,) = decision.answers
    assert answer.choice is True
    assert [entry.value for entry in answer.probabilities] == [True, False]
    assert [entry.probability for entry in answer.probabilities] == [0.8, 0.2]


def test_the_wrapper_echoes_score_labels_and_asks_with_the_descriptions(stub_server: Any) -> None:
    stub = stub_server(chat=chat_script(probabilities={"0": 0.1, "1": 0.7, "2": 0.2}))
    decision = client_for(stub).decisions.create(
        input="The app crashes on launch.",
        questions=[
            {
                **SCORE,
                "levels": [
                    {"label": "Cosmetic", "description": "Appearance only."},
                    {"label": "Workaround available"},
                    {"label": "Fully blocked", "description": "No workaround."},
                ],
            }
        ],
    )

    (answer,) = decision.answers
    assert answer.score == 1.1
    assert [entry.model_dump() for entry in answer.probabilities] == [
        {"value": 0, "label": "Cosmetic", "probability": 0.1},
        {"value": 1, "label": "Workaround available", "probability": 0.7},
        {"value": 2, "label": "Fully blocked", "probability": 0.2},
    ]
    # The prompt says what each level means, and says nothing extra for a level that means itself.
    prompt = json.dumps(stub.requests[0]["messages"])
    assert "Cosmetic: Appearance only." in prompt
    assert "Workaround available" in prompt
    assert "Workaround available: " not in prompt


def test_the_wrapper_echoes_a_null_name_and_keeps_question_order(stub_server: Any) -> None:
    stub = stub_server(chat=chat_script())
    decision = client_for(stub).decisions.create(
        input="I was charged twice for my order.",
        questions=[
            {"type": "predicate", "instructions": "Is this about a charge?"},
            {**SCORE, "name": "q"},
            {"type": "choice", "instructions": "Which department?", "choices": [{"value": "billing"}, {"value": "sales"}]},
        ],
    )

    assert [answer.type for answer in decision.answers] == ["predicate", "score", "choice"]
    assert [answer.name for answer in decision.answers] == [None, "q", None]


@pytest.mark.parametrize(
    ("input", "match"),
    [
        pytest.param([], "non-empty list", id="empty-messages"),
        pytest.param([{"role": "system", "content": "hi"}], "input message 0", id="non-user-role"),
        pytest.param(
            [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA"}]}],
            "input_image parts are not supported",
            id="image-part",
        ),
        pytest.param(
            [{"role": "user", "content": [{"type": "input_file", "file_url": "file-1"}]}],
            "input_file",
            id="unknown-part",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": "hi"}]},
            "input must be a string or a list of user messages",
            id="not-a-list",
        ),
        pytest.param(
            bytearray(b"hi"),
            "input must be a string or a list of user messages, got bytearray",
            id="bytes-like",
        ),
    ],
)
def test_the_wrapper_refuses_input_it_cannot_carry(
    stub_server: Any, input: Any, match: str
) -> None:
    stub = stub_server(chat=chat_script())
    with pytest.raises(JevperError, match=match):
        client_for(stub).decisions.create(input=input, questions=[PREDICATE])
    assert stub.requests == []


@pytest.mark.parametrize(
    ("questions", "match"),
    [
        pytest.param([{"type": "predicate", "name": "x"}], "Field required", id="no-instructions"),
        pytest.param([{**PREDICATE, "unexpected": 1}], "Extra inputs are not permitted", id="unknown-key"),
        pytest.param([{**CHOICE, "choices": [{"value": "a"}]}], "at least 2 items", id="one-choice"),
        pytest.param(
            [{**CHOICE, "choices": [{"value": f"o{index}"} for index in range(256)]}],
            "at most 255 items",
            id="256-choices",
        ),
        pytest.param([{**SCORE, "levels": [{"label": "low"}]}], "at least 2 items", id="one-level"),
        pytest.param(
            [{**SCORE, "levels": [{"label": f"l{index}"} for index in range(11)]}],
            "at most 10 items",
            id="eleven-levels",
        ),
        pytest.param([PREDICATE] * 201, r"1\.\.200", id="201-questions"),
        pytest.param(
            [{**CHOICE, "choices": [{"value": "billing"}, {"value": "billing"}]}],
            "cannot tell apart",
            id="duplicate-value",
        ),
        pytest.param(
            [{**CHOICE, "choices": [{"value": "true"}, {"value": True}]}],
            "cannot tell apart",
            id="string-and-boolean",
        ),
        pytest.param(
            [{**CHOICE, "choices": [{"value": "billing"}, {"value": 1}]}],
            r"choices\.1\.value: .*must be a string or a boolean, got int 1",
            id="integer-value",
        ),
        pytest.param(
            [{**CHOICE, "choices": [{"value": "billing"}, {"value": 0.0}]}],
            r"choices\.1\.value: .*must be a string or a boolean, got float 0\.0",
            id="float-value",
        ),
    ],
)
def test_the_wrapper_refuses_questions_the_wire_refuses(
    stub_server: Any, questions: Any, match: str
) -> None:
    stub = stub_server(chat=chat_script())
    with pytest.raises(InvalidQuestionError, match=match):
        client_for(stub).decisions.create(input="charged twice", questions=questions)
    assert stub.requests == []


def test_a_clients_own_examples_follow_a_wrapper_call(stub_server: Any) -> None:
    """Examples set on the client work here the way they work on ``system_one``.

    The constructor's ``examples`` is keyed by the id a question travels as, and this call's ids are
    positional — so a client configured with ``{"department": [...]}``, which guides the very same
    question under ``system_one``, would otherwise match nothing and drop them without a word. The
    names are translated to the ids this call travels as; a per-call mapping still wins over the
    client's own, and a key that names no question here is left exactly as it was.
    """

    stub = stub_server(chat=chat_script(probabilities={"billing": 0.9, "sales": 0.1}))
    client = client_for(
        stub,
        examples={"department": [Example(state="a request for a refund", answer="billing")]},
    )

    client.decisions.create(input="charged twice", questions=[CHOICE])
    assert "a request for a refund" in json.dumps(stub.requests[0]["messages"])

    client.decisions.create(
        input="charged twice",
        questions=[CHOICE],
        examples={"department": [Example(state="a cancelled subscription", answer="sales")]},
    )
    assert "a cancelled subscription" in json.dumps(stub.requests[1]["messages"])
    assert "a request for a refund" not in json.dumps(stub.requests[1]["messages"])

    # A client whose examples belong to another call — or to positional ids — is not misread.
    client = client_for(stub, examples={"other": [Example(state="unrelated", answer="billing")]})
    client.decisions.create(input="charged twice", questions=[CHOICE])
    assert "unrelated" not in json.dumps(stub.requests[2]["messages"])


def test_each_answer_echoes_its_own_questions_options(stub_server: Any) -> None:
    """Two questions of one kind travel as two ids, and each answer keeps its own options.

    The ids are positional and the questions may look alike, so this is where a translation that keyed
    answers by something other than position — or that reused one question's values for another —
    would show up as an answer carrying the wrong question's option values.
    """

    def script(body: dict[str, Any]) -> tuple[int, Any]:
        keys = option_keys(body)
        distribution = (
            {"billing": 0.6, "sales": 0.4}
            if keys == ["billing", "sales"]
            else {"true": 0.3, "false": 0.7}
        )
        return 200, chat_body(content=json.dumps({"probabilities": distribution}))

    stub = stub_server(chat=script)
    decision = client_for(stub).decisions.create(
        input="charged twice",
        questions=[
            {
                "type": "choice",
                "name": "department",
                "instructions": "Which department?",
                "choices": [{"value": "billing"}, {"value": "sales"}],
            },
            {
                "type": "choice",
                "name": "urgency",
                "instructions": "Is this urgent?",
                "choices": [{"value": True, "description": "needed today"}, {"value": False}],
            },
        ],
    )

    assert [answer.name for answer in decision.answers] == ["department", "urgency"]
    assert decision.answers[0].choice == "billing"
    assert [(entry.value, entry.probability) for entry in decision.answers[0].probabilities] == [
        ("billing", 0.6),
        ("sales", 0.4),
    ]
    assert decision.answers[1].choice is False
    assert [(entry.value, entry.probability) for entry in decision.answers[1].probabilities] == [
        (True, 0.3),
        (False, 0.7),
    ]


def test_examples_are_keyed_by_question_name(stub_server: Any) -> None:
    stub = stub_server(chat=chat_script(probabilities={"billing": 0.9, "sales": 0.1}))
    client = client_for(stub)
    client.decisions.create(
        input="I was charged twice for my order.",
        questions=[CHOICE],
        examples={"department": [Example(state="a request for a refund", answer="billing")]},
    )
    assert "a request for a refund" in json.dumps(stub.requests[0]["messages"])

    # A sequence needs no key at all: it applies to every question, as it does on ``system_one``.
    client.decisions.create(
        input="charged twice",
        questions=[CHOICE],
        examples=[Example(state="a refund request", answer="billing")],
    )
    assert "a refund request" in json.dumps(stub.requests[1]["messages"])

    # A named question beside an unnamed one still receives the mapping's example: the key matches a
    # name, and the question that carries none is asked without one rather than given someone else's.
    client.decisions.create(
        input="charged twice",
        questions=[{"type": "predicate", "instructions": "Is it urgent?"}, CHOICE],
        examples={"department": [Example(state="a refund request", answer="billing")]},
    )
    sent = [json.dumps(request["messages"]) for request in stub.requests[2:]]
    choice_sent = [text for text in sent if "Which department" in text]
    predicate_sent = [text for text in sent if "Is it urgent" in text]
    assert choice_sent and "a refund request" in choice_sent[0]
    assert predicate_sent and "a refund request" not in predicate_sent[0]

    with pytest.raises(InvalidQuestionError, match="no question is named 'other'"):
        client.decisions.create(input="charged twice", questions=[CHOICE], examples={"other": []})
    with pytest.raises(InvalidQuestionError, match="names more than one question"):
        client.decisions.create(input="charged twice", questions=[CHOICE, CHOICE], examples={"department": []})


def test_one_decisions_call_reads_the_same_on_every_surface(stub_server: Any) -> None:
    # The three bodies report the same usage on purpose: a surface's own default would otherwise be
    # the only difference in the payload, and this test is about the payload being the same.
    chat = stub_server(chat=lambda _: (200, chat_body(content='{"noul": 0.93}', reasoning_tokens=None)))
    responses = stub_server(
        responses=lambda _: (
            200,
            responses_body(text='{"noul": 0.93}', reasoning_tokens=None, cached_tokens=None),
        )
    )
    messages = stub_server(messages=lambda _: (200, messages_body(text='{"noul": 0.93}')))

    clients = [
        SystemOneClient(openai_client(chat), model="m", api="chat_completions", method="structured"),
        SystemOneClient(openai_client(responses), model="m", api="responses", method="structured"),
        SystemOneClient(anthropic_client(messages), model="m", api="messages", method="structured"),
    ]
    payloads = [
        client.decisions.create(input="charged twice", questions=[PREDICATE]).model_dump()
        for client in clients
    ]

    assert payloads[0] == payloads[1] == payloads[2]
    assert payloads[0] == {
        "model": "m",
        "answers": [{"type": "predicate", "name": "billing", "probability": 0.93}],
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": None, "cache_write_tokens": None},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": None},
            "total_tokens": 13,
        },
    }


def test_the_wrapper_answers_over_the_system_one_surface_too(stub_server: Any) -> None:
    """The native surface is the client's own option, and the wrapper rides it unchanged.

    One request carries every question, keyed by position because the wire's ``name`` need not be
    there, and the service's own numbers come back as they were sent — this is the one path where the
    ``score`` and the ``confidence`` are read rather than computed.
    """

    def script(body: dict[str, Any]) -> tuple[int, Any]:
        assert list(body["questions"]) == ["0", "1"]
        return 200, {
            "model": "native",
            "answers": {
                "0": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": {"billing": 0.7, "sales": 0.3},
                    "confidence": 0.4,
                },
                "1": {
                    "type": "score",
                    "score": 1.0,
                    "legend": {"0": "low", "1": "severe"},
                    "probabilities": {"0": 0.4, "1": 0.6},
                    "confidence": 0.2,
                },
            },
            "usage": {"input_tokens": 7, "output_tokens": 2},
        }

    stub = stub_server(systemone=script)
    client = SystemOneClient(openai_client(stub), model="native", api="systemone")
    decision = client.decisions.create(
        input="charged twice",
        questions=[
            {
                "type": "choice",
                "name": "department",
                "instructions": "Which department?",
                "choices": [{"value": "billing"}, {"value": "sales"}],
            },
            {
                "type": "score",
                "name": "severity",
                "instructions": "How bad is it?",
                "levels": [{"label": "low"}, {"label": "severe"}],
            },
        ],
    )

    assert decision.model == "native"
    assert [answer.name for answer in decision.answers] == ["department", "severity"]
    assert decision.answers[0].choice == "billing"
    assert [entry.value for entry in decision.answers[0].probabilities] == ["billing", "sales"]
    assert decision.answers[0].confidence == 0.4
    assert [entry.label for entry in decision.answers[1].probabilities] == ["low", "severe"]
    assert decision.answers[1].score == 1.0
    assert decision.usage.total_tokens == 9


def test_the_async_wrapper_answers_the_same(stub_server: Any) -> None:
    stub = stub_server(chat=chat_script(probabilities={"billing": 0.9, "sales": 0.1}))

    async def ask() -> Decision:
        client = AsyncSystemOneClient(
            async_openai_client(stub), model="m", api="chat_completions", method="structured"
        )
        try:
            return await client.decisions.create(input="charged twice", questions=[CHOICE])
        finally:
            await client.aclose()

    answered = asyncio.run(ask())
    sync = client_for(stub).decisions.create(input="charged twice", questions=[CHOICE])

    assert answered.model_dump() == sync.model_dump()


def test_a_system_one_response_converts_to_a_decision() -> None:
    response = SystemOneResponse(
        model="m",
        answers={
            "intent": ChoiceAnswer(
                choice="billing", probabilities={"billing": 0.9, "technical": 0.1}, confidence=0.8
            ),
            "sentiment": ScoreAnswer(
                score=1.0,
                legend={0: "calm", 1: "angry"},
                probabilities={0: 0.1, 1: 0.9},
                confidence=0.8,
            ),
            "needs_human": NoulAnswer(noul=0.4),
        },
        usage=Usage(input_tokens=10, output_tokens=3),
    )
    decision = response.to_decision()

    assert decision.raw is response
    assert decision.model_dump() == {
        "model": "m",
        "answers": [
            {
                "type": "choice",
                "name": "intent",
                "choice": "billing",
                "probabilities": [
                    {"value": "billing", "probability": 0.9},
                    {"value": "technical", "probability": 0.1},
                ],
                "confidence": 0.8,
            },
            {
                "type": "score",
                "name": "sentiment",
                "score": 1.0,
                "probabilities": [
                    {"value": 0, "label": "calm", "probability": 0.1},
                    {"value": 1, "label": "angry", "probability": 0.9},
                ],
                "confidence": 0.8,
            },
            {"type": "predicate", "name": "needs_human", "probability": 0.4},
        ],
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": None, "cache_write_tokens": None},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": None},
            "total_tokens": 13,
        },
    }

    # The union the wire documents is the union this model reads: a refusal answer from the endpoint
    # itself parses, although nothing jevper produces is one.
    payload = Decision.model_validate(
        {
            "model": "m",
            "answers": [{"type": "refusal", "name": None}],
            "usage": {
                "input_tokens": 1,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 1,
            },
        }
    )
    assert isinstance(payload.answers[0], DecisionRefusalAnswer)


def test_a_score_level_the_legend_does_not_name_keeps_its_index() -> None:
    """``legend`` is what a response remembers of the rubric it was asked with.

    A response that reports a level its legend does not name still converts: the level index is how
    the response spells its own levels, and a label nobody recorded is not a reason to lose the
    answer — or to hand the caller a ``KeyError`` from a public converter.
    """
    response = SystemOneResponse(
        model="m",
        answers={
            "sentiment": ScoreAnswer(
                score=1.0,
                legend={0: "calm"},
                probabilities={0: 0.2, 1: 0.8},
                confidence=0.6,
            )
        },
    )

    answer = response.to_decision().answers[0]
    assert [entry.label for entry in answer.probabilities] == ["calm", "1"]
    assert [entry.value for entry in answer.probabilities] == [0, 1]


def test_a_payload_from_the_endpoint_is_read_as_the_endpoint_wrote_it() -> None:
    """These models are the wire's, not jevper's own stricter reading of it.

    ``probability``, ``probabilities[].probability``, ``score`` and ``confidence`` are bare ``number``
    in the schema, unbounded; the endpoint's numbers are its own and jevper never questions them (the
    guide states a 0..1 estimate for a predicate's ``probability``, nothing for ``confidence``). A
    caller who reads a real ``/v1/decisions`` payload through ``Decision`` gets that payload, not a
    validation error over a bound the schema does not state. What jevper *produces* is bounded where it
    is computed, so a decision from this wrapper stays in ``[0, 1]`` either way.
    """
    payload = Decision.model_validate(
        {
            "model": "gpt-6-luna",
            "answers": [
                {"type": "predicate", "name": "billing", "probability": 0.92},
                {
                    "type": "choice",
                    "name": "department",
                    "choice": "billing",
                    "probabilities": [
                        {"value": "billing", "probability": 0.95},
                        {"value": "sales", "probability": 0.02},
                    ],
                    "confidence": 0.93,
                },
                {
                    "type": "score",
                    "name": "severity",
                    "score": 1.1,
                    "probabilities": [
                        {"value": 0, "label": "cosmetic", "probability": 0.1},
                        {"value": 1, "label": "workaround", "probability": 0.7},
                        {"value": 2, "label": "blocked", "probability": 0.2},
                    ],
                    "confidence": 0.55,
                },
                {"type": "refusal", "name": "where_is_it"},
            ],
            "usage": {
                "input_tokens": 312,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 24,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 336,
            },
        }
    )

    assert [answer.type for answer in payload.answers] == [
        "predicate",
        "choice",
        "score",
        "refusal",
    ]
    # The one name the wire always echoes is read back even where a question asked for none.
    assert payload.answers[0].name == "billing"
    assert payload.answers[1].choice == "billing"
    assert payload.answers[2].score == 1.1

    # Synthetic, and here for one reason: nothing in the schema bounds a `confidence`, so a payload
    # is read as the endpoint wrote it — the same stance the probabilities and the predicate's
    # `probability` already take, and the reason a decision this wrapper produced can only be in
    # [0, 1]: jevper computes that number under a bound, a service reports its own.
    loose = Decision.model_validate(
        {
            "model": "gpt-6-luna",
            "answers": [
                {
                    "type": "choice",
                    "name": None,
                    "choice": "billing",
                    "probabilities": [
                        {"value": "billing", "probability": 0.95},
                        {"value": "sales", "probability": 0.02},
                    ],
                    "confidence": 1.5,
                }
            ],
            "usage": {
                "input_tokens": 1,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 2,
            },
        }
    )
    assert loose.answers[0].confidence == 1.5


def test_a_choice_the_distribution_does_not_include_still_converts() -> None:
    """A response is converted as it answered, never re-decided.

    A response assembled by hand can name a chosen option its own ``probabilities`` do not carry — the
    answer models tie the two together only as far as each field's own type, and the live surfaces
    enforce the link before an answer is built. The conversion therefore hands back the service's own
    value rather than looking it up in the distribution and failing: the same rule a level the legend
    does not name already follows, and the reason neither converter can raise a ``KeyError``.
    """
    response = SystemOneResponse(
        model="m",
        answers={
            "intent": ChoiceAnswer(
                choice="billing", probabilities={"technical": 0.9}, confidence=0.8
            )
        },
    )

    answer = response.to_decision().answers[0]
    assert answer.choice == "billing"
    assert [(entry.value, entry.probability) for entry in answer.probabilities] == [("technical", 0.9)]


def test_a_message_list_input_reaches_the_model_as_turns(stub_server: Any) -> None:
    """The wire's second ``input`` form: one turn per message, its parts joined into that turn.

    Only refusals of this branch were pinned, so a regression that dropped the turns or collapsed them
    into one state would have shipped green — and this is the form a caller reaches for when the input
    is a conversation rather than one block of text.
    """
    turns_in = [
        {"role": "user", "content": "first turn"},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "part one"},
                {"type": "input_text", "text": "part two"},
            ],
        },
    ]
    stub = stub_server(chat=chat_script(probabilities={"billing": 0.9, "sales": 0.1}))
    decision = client_for(stub).decisions.create(input=turns_in, questions=[CHOICE])

    assert decision.answers[0].choice == "billing"
    turns = [
        (message["role"], message["content"])
        for message in stub.requests[0]["messages"]
        if message["role"] == "user"
    ]
    # The question turn comes first, as it does for every state; the input's turns follow it in order.
    assert turns[-2:] == [("user", "first turn"), ("user", "part one\npart two")]

    # The same input on the Messages surface, where the turns are content blocks: the texts still
    # arrive in order, one message per turn.
    messages = stub_server(
        messages=lambda _: (200, messages_body(text='{"probabilities": {"billing": 0.9, "sales": 0.1}}'))
    )
    client = SystemOneClient(
        anthropic_client(messages), model="m", api="messages", method="structured"
    )
    client.decisions.create(input=turns_in, questions=[CHOICE])

    sent = json.dumps(messages.requests[0]["messages"])
    assert sent.index("first turn") < sent.index("part one") < sent.index("part two")


@pytest.mark.parametrize(
    ("questions", "match"),
    [
        pytest.param("a question", "must be a list of question objects", id="not-a-list"),
        pytest.param({"q": PREDICATE}, "must be a list of question objects", id="id-keyed-mapping"),
        pytest.param(5, "must be a list of question objects", id="a-number"),
    ],
)
def test_the_wrapper_refuses_questions_it_cannot_take(
    stub_server: Any, questions: Any, match: str
) -> None:
    """``system_one`` takes a mapping of id to question; this takes the wire's list, and a caller who
    arrived with the other shape is told which one is expected rather than answered from a guess."""
    stub = stub_server(chat=chat_script())
    with pytest.raises(InvalidQuestionError, match=match):
        client_for(stub).decisions.create(input="charged twice", questions=questions)
    assert stub.requests == []


def test_questions_that_share_a_name_are_both_answered(stub_server: Any) -> None:
    """The wire lets two questions carry one name, and both of them are questions.

    Keying the questions by name — the obvious design, and the one the ids exist instead of — would
    turn this legal request into a refusal or an overwrite; the ids are positional, and the name is
    only the echo.
    """
    stub = stub_server(chat=chat_script(probabilities={"billing": 0.9, "sales": 0.1}))
    decision = client_for(stub).decisions.create(
        input="charged twice",
        questions=[{**CHOICE, "name": "q"}, {**SCORE, "name": "q"}],
    )

    assert [answer.name for answer in decision.answers] == ["q", "q"]
    assert [answer.type for answer in decision.answers] == ["choice", "score"]
    assert decision.answers[0].choice == "billing"
    assert [entry.label for entry in decision.answers[1].probabilities] == ["low", "high"]


def test_option_values_are_matched_to_the_distribution_by_key(stub_server: Any) -> None:
    """A distribution is read by option key, never by the order it arrived in.

    The live service answers out of ask order — its recorded replies shuffle the options — so a
    mapping built by zipping the response with the request would label every probability with the
    wrong option there, while passing every test whose distribution happens to come back in order.
    """
    shuffled = {
        "0": {
            "type": "choice",
            "choice": "technical",
            "confidence": 0.4,
            "probabilities": {"sales": 0.3, "billing": 0.4, "technical": 0.3},
        }
    }
    stub = stub_server(systemone=answering_with(shuffled))
    client = SystemOneClient(openai_client(stub), model="m", api="systemone")
    decision = client.decisions.create(
        input="charged twice",
        questions=[
            {
                "type": "choice",
                "name": "department",
                "instructions": "Which department?",
                "choices": [
                    {"value": "billing"},
                    {"value": "technical"},
                    {"value": "sales"},
                ],
            }
        ],
    )

    answer = decision.answers[0]
    assert [(entry.value, entry.probability) for entry in answer.probabilities] == [
        ("sales", 0.3),
        ("billing", 0.4),
        ("technical", 0.3),
    ]
    assert answer.choice == "technical"


def test_the_async_wrapper_propagates_a_refusal(stub_server: Any) -> None:
    """The async accessor's failure path is the sync one: a refusal is the model failing the call."""
    stub = stub_server(
        chat=lambda _: (200, chat_body(content=None, refusal="I can't help with that"))
    )

    async def ask() -> None:
        client = AsyncSystemOneClient(
            async_openai_client(stub), model="m", api="chat_completions", method="structured"
        )
        try:
            await client.decisions.create(input="charged twice", questions=[PREDICATE])
        finally:
            await client.aclose()

    with pytest.raises(ModelRefusalError, match="the model refused to answer"):
        asyncio.run(ask())


def test_a_call_level_method_reaches_the_wire(stub_server: Any) -> None:
    """``method=`` is the caller's per call, as it is on ``system_one``.

    The readout it selects is where the probabilities come from, so the answer follows the call rather
    than only the client it was made through.
    """
    stub = stub_server(
        chat=lambda _: (200, chat_body(content="A", logprobs=[("A", -0.12), ("B", -2.47)]))
    )
    client = client_for(stub, method="structured")
    decision = client.decisions.create(
        input="charged twice", questions=[CHOICE], method="logprobs"
    )

    assert stub.requests[0]["logprobs"] is True
    answer = decision.answers[0]
    assert answer.choice == "billing"
    assert [entry.value for entry in answer.probabilities] == ["billing", "sales"]
    assert answer.probabilities[0].probability > answer.probabilities[1].probability


def test_a_refusal_raises_instead_of_becoming_a_refusal_answer(stub_server: Any) -> None:
    stub = stub_server(
        chat=lambda _: (200, chat_body(content=None, refusal="I can't help with that"))
    )
    with pytest.raises(ModelRefusalError, match="the model refused to answer"):
        client_for(stub).decisions.create(input="charged twice", questions=[PREDICATE])


def test_a_provider_that_reports_no_usage_leaves_the_numbers_null(stub_server: Any) -> None:
    stub = stub_server(
        chat=lambda _: (
            200,
            chat_body(content='{"noul": 0.5}', input_tokens=None, output_tokens=None),
        )
    )
    decision = client_for(stub).decisions.create(input="charged twice", questions=[PREDICATE])

    assert decision.usage.model_dump() == {
        "input_tokens": None,
        "input_tokens_details": {"cached_tokens": None, "cache_write_tokens": None},
        "output_tokens": None,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": None,
    }
    assert json.loads(decision.model_dump_json())["usage"]["total_tokens"] is None
