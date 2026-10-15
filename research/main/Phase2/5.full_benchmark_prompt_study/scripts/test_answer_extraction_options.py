from __future__ import annotations

from prepare_answer_extraction import extract_option_labels


def test_extract_ascii_option_lines():
    query = "Question?\n(A) Alpha\nB. Beta\nC) Gamma"
    assert extract_option_labels(query) == ["A", "B", "C"]


def test_extract_fullwidth_chinese_parentheses():
    query = (
        "这张图片显示了自车的前视图。\n"
        "（A） 有许多汽车、两辆卡车和三名行人。\n"
        "（B） 有两辆摩托车和许多汽车。\n"
        "（C） 有一辆公共汽车、许多汽车、一辆摩托车和一个交通锥。\n"
        "（D） 有一辆卡车、一名行人和许多汽车。\n"
        "（E） 以上所有答案都是错误的。\n"
        "最佳答案为："
    )
    assert extract_option_labels(query) == ["A", "B", "C", "D", "E"]


def test_does_not_treat_answer_instruction_as_an_option():
    query = "请从 A、B、C 中选择。\n最佳答案为："
    assert extract_option_labels(query) == []


def test_extract_compact_options_without_space_before_cjk_text():
    query = "问题\n(A)蓝色\n(B)红色\n(C)白色\n(D)黑色\n(E)没有目标"
    assert extract_option_labels(query) == ["A", "B", "C", "D", "E"]
