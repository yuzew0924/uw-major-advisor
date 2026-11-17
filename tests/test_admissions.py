from rag.admissions import admissions, normalize_course
from rag.chain import answer_question


def test_structured_answer_before_api(monkeypatch):
    monkeypatch.setattr("rag.chain.require_api_key", lambda: (_ for _ in ()).throw(AssertionError()))
    for course, requirement in (("cse-123", "Computer Programming"), ("CSE312", "Statistics")):
        answer, chunks = answer_question(f"{course} 可以满足informatics的prerequisite吗")
        assert requirement in answer
        assert "ischool.uw.edu" in answer
        assert chunks == []


def test_options_keep_course_combinations_and_campus():
    db = admissions()
    partial = db.answer("Does STAT 221 satisfy an Informatics prerequisite?")
    assert "还需 STAT 321" in partial
    full = db.answer("Do STAT 221 and STAT 321 satisfy an Informatics prerequisite?")
    assert "满足该项的课程选项" in full
    assert "还需" not in full
    assert "Bothell" in db.answer("Does CSS 142 satisfy an Informatics prerequisite?")
    assert normalize_course("cse-123a") == "CSE 123A"


def test_admissions_routing_and_pathway():
    db = admissions()
    assert "Social Sciences" in db.answer("info prerequisite")
    assert "入读后修读" in db.answer("Does INFO 200 count toward the Informatics prerequisite for transfer applicants?")
    assert "1/1" in db.answer("CSE 123 可以满足哪些 major 的 prerequisite")
    assert db.answer("CSE 123 的 prerequisite 是什么") is None
    assert db.answer("我可以选 CSE 123 吗") is None
    assert db.answer("Does CSE 123A satisfy the Informatics prerequisite?") is None
    assert "你想查" in db.answer("INFO 200 prerequisite")


def test_streamlit_displays_structured_answer():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("../app.py").run()
    app.chat_input[0].set_value("cse 123 可以满足informatics的prerequisite吗").run()
    assert not app.exception
    assert any("Computer Programming" in item.value for item in app.markdown)
