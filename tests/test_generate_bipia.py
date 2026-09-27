"""Unit tests for the NewsQA parsing in generate_bipia, which BIPIA's md5 can only check end to end."""

from generate_bipia import newsqa_answers, normalize_story, read_newsqa_csv


def test_answers_from_char_ranges():
    story = "Hello, world. Foo bar!"
    # Crowdsourcers separated by "|", multiple selections by ","; "None" means no answer.
    assert newsqa_answers(story, "0:6|7:13,14:22|None") == ["Foo bar", "Hello", "world"]


def test_answers_deduplicated_and_sorted():
    assert newsqa_answers("abc def", "4:7|0:3|4:7") == ["abc", "def"]
    assert newsqa_answers("abc", "None|None") == []


def test_normalize_story_collapses_blank_lines():
    assert normalize_story("a\n \t\nb\n\n\n\nc\nd") == "a\n\nb\n\nc\nd"


def test_read_csv_matches_hf_loader(tmp_path):
    path = tmp_path / "combined.csv"
    path.write_text(
        '"story_id","question","answer_char_ranges","is_answer_absent","is_question_bad","validated_answers","story_text"\n'
        '"./cnn/stories/a.story","Who?","0:3|None","0.5","0.0","{}","Bob ran.\n\nHe won."\n'
        "\n"
        '"./cnn/stories/b.story", "Where?","4:9","0.0","0.0","","In Paris"\n',
        encoding="utf-8",
    )
    rows = read_newsqa_csv(path)
    assert len(rows) == 2  # header and blank line skipped
    assert rows[0][1] == "Who?" and rows[0][2] == "0:3|None"
    assert rows[0][-1] == "Bob ran.\n\nHe won."
    assert rows[1][1] == "Where?"  # skipinitialspace
