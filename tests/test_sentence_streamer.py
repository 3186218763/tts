import pytest

from dialogue.sentence_streamer import SentenceStreamer


def test_single_sentence_across_tokens():
    streamer = SentenceStreamer()
    result = []
    for token in ["你好", "呀。"]:
        result.extend(streamer.add_token(token))
    assert result == ["你好呀。"]


def test_multiple_sentences_one_token():
    streamer = SentenceStreamer()
    result = streamer.add_token("你好。我是花音。")
    assert result == ["你好。我是花音。"]


def test_no_ending_no_output():
    streamer = SentenceStreamer()
    assert streamer.add_token("你好呀") == []


def test_flush_returns_remaining():
    streamer = SentenceStreamer()
    streamer.add_token("还没说完")
    assert streamer.flush() == "还没说完"


def test_flush_empty_returns_none():
    streamer = SentenceStreamer()
    assert streamer.flush() is None


def test_exclamation_and_question():
    streamer = SentenceStreamer()
    result = streamer.add_token("好棒！真的吗？")
    assert result == ["好棒！真的吗？"]


def test_newline_as_ending():
    streamer = SentenceStreamer()
    result = streamer.add_token("第一行\n第二行")
    assert result == []
    assert streamer.flush() == "第一行\n第二行"


def test_force_split_at_comma():
    streamer = SentenceStreamer(max_chars=5)
    result = streamer.add_token("你好呀，我最近很开心呢")
    assert result == ["你好呀，"]
    assert streamer.flush() == "我最近很开心呢"


def test_force_split_no_comma():
    streamer = SentenceStreamer(max_chars=10, min_chars=2)
    result = streamer.add_token("abcdefghijklmnopqrstu")
    # No natural boundary: wait until max_chars + 10, then split once.
    assert result == ["abcdefghijklmnopqrst"]
    assert streamer.flush() == "u"


def test_does_not_split_on_punctuation_inside_stage_direction():
    streamer = SentenceStreamer()
    assert streamer.add_token("（心里想着：你好。") == []
    assert streamer.add_token("）今天也要加油！") == ["（心里想着：你好。）今天也要加油！"]


def test_does_not_force_split_a_long_stage_direction():
    streamer = SentenceStreamer(max_chars=5)
    assert streamer.add_token("（心里想着今天见到你真的非常开心") == []
    assert streamer.add_token("）你好。") == ["（心里想着今天见到你真的非常开心）你好。"]


def test_ignores_a_comma_inside_a_closed_stage_direction():
    streamer = SentenceStreamer(max_chars=5)
    assert streamer.add_token("（轻轻点头，露出微笑）你好呀") == []


def test_short_acknowledgement_waits_for_following_sentence():
    streamer = SentenceStreamer(max_chars=50, min_chars=4)
    assert streamer.add_token("嗯。") == []
    assert streamer.add_token("我记住啦！") == ["嗯。我记住啦！"]


def test_long_sentence_prefers_a_later_natural_boundary():
    streamer = SentenceStreamer(max_chars=10, min_chars=2)
    result = streamer.add_token("前半段没有停顿直到这里，后半段继续")
    assert result == ["前半段没有停顿直到这里，"]
    assert streamer.flush() == "后半段继续"


@pytest.mark.parametrize(
    "kwargs",
    [{"max_chars": 0}, {"max_chars": 3, "min_chars": 4}, {"min_chars": 0}],
)
def test_rejects_invalid_sentence_limits(kwargs):
    with pytest.raises(ValueError):
        SentenceStreamer(**kwargs)
