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
    assert result == ["你好。", "我是花音。"]


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
    assert result == ["好棒！", "真的吗？"]


def test_newline_as_ending():
    streamer = SentenceStreamer()
    result = streamer.add_token("第一行\n第二行")
    assert result == ["第一行\n"]


def test_force_split_at_comma():
    streamer = SentenceStreamer(max_chars=5)
    result = streamer.add_token("你好呀，我最近很开心呢")
    # "你好呀，" 在逗号处切分（长度 4 <= max_chars=5）
    assert "你好呀，" in result


def test_force_split_no_comma():
    streamer = SentenceStreamer(max_chars=3)
    result = streamer.add_token("abcdefgh")
    # 无标点无逗号，每 max_chars 字符强制切
    assert result[0] == "abc"
    assert result[1] == "def"
    # 剩余不足 max_chars 的内容留在缓冲区，可用 flush 取出
    assert streamer.flush() == "gh"
