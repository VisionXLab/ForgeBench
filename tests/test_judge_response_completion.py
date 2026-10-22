from types import SimpleNamespace
import json

import pytest

from eval import llm_judge_openai as judge


def response(content, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish)])


@pytest.mark.parametrize("bad", [response(""), response('{"score":75}', "length"),
                                SimpleNamespace(choices=[])])
def test_strict_transport_retries_empty_or_truncated_response(monkeypatch, bad):
    monkeypatch.setitem(judge.CONFIG, "strict_json", True)
    monkeypatch.setitem(judge.CONFIG, "base_delay", 0)
    good = response('{"score":75}')
    replies = iter([bad, good])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: next(replies))))
    assert judge._call_with_backoff(client, "test", "system", []) is good


def test_repeated_empty_responses_fail_instead_of_producing_score(monkeypatch):
    monkeypatch.setitem(judge.CONFIG, "strict_json", True)
    monkeypatch.setitem(judge.CONFIG, "base_delay", 0)
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return response("")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(ValueError, match="judge_empty_response"):
        judge._call_with_backoff(client, "test", "system", [])
    assert len(calls) == judge.CONFIG["max_retries"] + 1


def test_strict_mode_does_not_rescue_invalid_json_with_scalar_score(monkeypatch):
    monkeypatch.setitem(judge.CONFIG, "strict_json", True)
    assert judge._parse_score_0_100('{"score":75}') is None
    assert judge._parse_score_0_100('75') is None


def test_calibrated_gpt_omits_temperature(monkeypatch):
    monkeypatch.setitem(judge.CONFIG, "strict_json", True)
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return response('{"score":75}')
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    judge._call_with_backoff(client, "gpt-5.6-sol", "system", [])
    assert "temperature" not in calls[0]


def test_industrial_prompt_supplies_exact_ids_alongside_questions(monkeypatch):
    monkeypatch.setitem(judge.CONFIG, "strict_json", True)
    monkeypatch.setattr(judge, "_get_client", lambda: None)
    questions = [{"id": "erob_105_q1", "text": "Is the path clear?"},
                 {"id": "erob_105_q2", "text": "Does the AGV wait?"}]
    def call(client, **kwargs):
        prompt = kwargs["messages"][0]["content"][-1]["text"]
        for q in questions:
            assert f'[id: {json.dumps(q["id"])}] {q["text"]}' in prompt
        assert json.dumps([q["id"] for q in questions]) in prompt
        return response(json.dumps({"judgments": [
            {"id": q["id"], "answer": "yes", "visible_evidence": "visible scene evidence"}
            for q in questions]}))
    monkeypatch.setattr(judge, "_call_with_backoff", call)
    result = judge.judge_sample_industrial_logic_and_fact_alignment([], questions, {})
    assert result["answers"] == {q["id"]: "yes" for q in questions}
