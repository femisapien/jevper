# Decisions API compatibility

*Independent implementation of OpenAI's documented Decisions API schema — not affiliated with, endorsed by, or supported by OpenAI.*

`client.decisions.create(input=…, questions=[…])` takes the Decisions API's documented request and
returns its documented response: `model`, the `answers` in question order, and one `usage` for the
call. The questions are answered by the model behind whatever surface the client is configured with —
Chat Completions, Responses, or the Anthropic-compatible Messages API.

## Why it exists

[`POST /v1/decisions`](https://developers.openai.com/api/docs/guides/decisions) is one route on one
endpoint, and it asks three kinds of question: `predicate`, `choice` and `score`. jevper asks the same
three, so an endpoint that does not serve the route — a local server, a gateway, a provider whose
Decisions route has not shipped — can answer the same questions here, in the same schema. Moving to
the endpoint itself later is a client swap, not a rewrite.

## A call

```python
from openai import OpenAI
from jevper import SystemOneClient

client = SystemOneClient(OpenAI(), model="gpt-5.6-terra")

decision = client.decisions.create(
    input="I was charged twice for the same subscription this month.",
    questions=[
        {
            "type": "choice",
            "name": "department",
            "instructions": "Which department should handle this complaint?",
            "choices": [
                {"value": "billing", "description": "Payments, invoices, and refunds."},
                {"value": "technical", "description": "Problems using the product."},
                {"value": "sales", "description": "Pricing, plans and upgrades."},
            ],
        }
    ],
)

answer = decision.answers[0]
print(answer.name, answer.choice, answer.confidence)
print([(entry.value, entry.probability) for entry in answer.probabilities])
print(decision.usage.total_tokens)
```

`AsyncSystemOneClient` carries the same accessor, with `await`. A dump is the wire's own payload and
nothing else — `decision.model_dump()` and `decision.model_dump_json()` carry exactly `model`,
`answers` and `usage`:

```json
{"model":"gpt-5.6-terra",
 "answers":[{"type":"choice","name":"department","choice":"billing",
             "probabilities":[{"value":"billing","probability":0.82},
                              {"value":"technical","probability":0.12},
                              {"value":"sales","probability":0.06}],
             "confidence":0.73}],
 "usage":{"input_tokens":312,"input_tokens_details":{"cached_tokens":null,"cache_write_tokens":null},
          "output_tokens":24,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":336}}
```

## The request

| Wire field | Accepted here | Carried as |
| --- | --- | --- |
| `model` | A string | `model=`, defaulting to the client's own model |
| `input`, as a string | A string | `state`, asked as one quoted document turn |
| `input`, as messages | A list of `{"role": "user", "content": …}` | The state as turns: one per message, in order, with a message's `input_text` parts joined by a newline into its text |
| `questions` | 1..200 question objects | One question per entry, keyed by position |
| `predicate` | `type`, `instructions`, optional `name` | `Noul(instructions=…)` |
| `choice` | `type`, `instructions`, `choices[2..255]` | `Choice(instructions=…, criteria=…)`, one criterion per option: its `description`, or `None` |
| `choice.choices[].value` | A string or a boolean | The criterion key; a boolean keeps its value on the answer (see below) |
| `score` | `type`, `instructions`, `levels[2..10]` | `Score(instructions=…, criteria=…)`, one criterion per level: `"<label>: <description>"`, or the label alone |

An option's `description`, a level's `description` and a question's `instructions` are what the model
is asked with, and the labels, names and values are echoed back on the answers, a boolean value among
them. What a criterion key cannot be is text that reads two ways: the wire counts a string `"true"`
and a boolean `true` as distinct options, and jevper asks with one key per option, so a request whose
options would collapse onto one key — a repeated value, or a string beside a boolean that reads like
it — is refused, naming both values, rather than merged.

Everything else on the call is `system_one`'s own option, with the same meaning and the same default:
`examples`, `model`, `method`, `api`, `reasoning`, `temperature`, `prompt_cache_key`, `extras` and
`keep_alive`. See [`system_one`](api.md#system_one) for each of them, and [Few-shot examples](few-shot.md)
for how `examples` is keyed here: by question **name**, since names are what the caller wrote, while
the ids the questions travel as are positional — and examples set on the client, which `system_one`
reads by id, are recognised by name here the same way, so a client built with
`SystemOneClient(..., examples={"department": […]})` still guides a question named `department`.

## What is not carried

| Wire field | What happens |
| --- | --- |
| An `input_image` part | Refused with a `JevperError` before any request: no jevper surface renders images, so an image would be evidence the model never saw |
| `safety_identifier` | No parameter. An unknown keyword is a `TypeError`, as everywhere in jevper |
| An answer of type `refusal` | A refusal the model gives the whole question raises `ModelRefusalError`, naming what was refused — see [Failures](troubleshooting.md). The type is in the `Decision` model so a payload from the endpoint itself still validates; jevper never produces one |
| Token counts the provider omitted | `null`, not `0` — a count nobody reported is not a count. `input_tokens + output_tokens` is `null` when either is, and `cache_write_tokens` is always `null`: jevper reads no cache-write counter from any surface |

## The answers

Answers come back in question order, and each carries the `name` its question was asked with —
`null` when it was asked without one, exactly as the wire documents. One answer object per question
kind:

| Answer | Fields | From |
| --- | --- | --- |
| `predicate` | `name`, `probability` | The Jev `noul`, passed through unchanged |
| `choice` | `name`, `choice`, `probabilities[].value`, `probabilities[].probability`, `confidence` | The winning option's own value, every option's own value, and the distribution read back entry by entry, each labelled with the option it belongs to — the order a service sends its options in is its own. `confidence` is the same number `ChoiceAnswer` carries |
| `score` | `name`, `score`, `probabilities[].value`, `probabilities[].label`, `probabilities[].probability`, `confidence` | The Jev `score` (the probability-weighted level index), each level's index and the label the caller wrote for it, and the Jev distribution. `confidence` is the same number `ScoreAnswer` carries |
| `refusal` | `name` | Nothing here; it exists for payloads from the endpoint itself |

The numbers a decision carries are jevper's where jevper computed them and the model's or the service's
own where it did not: a `confidence` is in `[0, 1]` when it comes from this wrapper (that number is
computed under a bound), while a `probability`, a distribution or a `score` is read back as it arrived,
unbounded as the wire's `number` is — which is also how a payload from the real endpoint parses through
`Decision` whatever numbers it carries. The wire's own type is what sets the shape here: one `number`,
no range.

One thing the `usage` shows that the endpoint's own does not: on the prompt surfaces each question is
its own provider call — the native `systemone` surface asks them all in one request — so the token
counts here are summed over those calls. A ten-question rubric is ten calls' worth of tokens and
latency; the answers are the same, the bill is per question.

## Converting a `system_one` response

A response you already have converts with `SystemOneResponse.to_decision()`:

```python
response = client.system_one(state=…, questions={"intent": Choice(criteria=…)})
decision = response.to_decision()
```

Only what the response itself remembers can be echoed: a question's `name` is its id, a `choice`
value is the option key it was asked with, and a `score` level's `label` is that level's criteria
entry rendered as text — or, for a level the response's own legend does not name, its index, which is
how a response spells its levels everywhere else. `client.decisions.create` echoes the caller's own
names, values and labels instead, because it still holds the request.

Either way, `decision.raw` keeps the whole jevper response behind the conversion — `reasoning`,
`debug`, and each answer's `laya` extras where the service sent them. It is excluded from every
dump, so a payload here stays the wire's own shape.
