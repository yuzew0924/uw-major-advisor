import re
import json
import requests
from bs4 import BeautifulSoup

scrape_url = "https://uapp.engr.washington.edu/essay_questions.php"

canonical_urls = {
    "Computer Science & Engineering": "https://www.cs.washington.edu/academics/ugrad/admissions",
    "Civil & Environmental Engineering": "https://www.ce.washington.edu/future/undergrad",
    "Human Centered Design & Engineering": "https://www.hcde.washington.edu/bs/admissions",
    "Electrical & Computer Engineering": "https://www.ece.uw.edu/academics/bachelor-of-science/bs-admissions-requirements/",
}

html = requests.get(scrape_url).text
soup = BeautifulSoup(html, "html.parser")

results = []


def clean_text(text):
    return "\n".join(
        line.strip()
        for line in text.replace("\xa0", " ").splitlines()
        if line.strip()
    )


def normalize_question_title(title):
    title = re.sub(r"Computer Engineering Question", "Question", title)
    title = re.sub(r"Computer Science Question", "Question", title)
    return title.strip()


for accordion in soup.select("div.accordion"):
    major_tag = accordion.find("h4")

    if not major_tag:
        continue

    major = major_tag.get_text(" ", strip=True)
    items = accordion.select(".accordion-item")

    questions = []
    questions_by_number = {}

    for item in items:
        question_title_tag = item.select_one("button.accordion-button")
        body_tag = item.select_one(".card-body")

        if not question_title_tag or not body_tag:
            continue

        question_title = clean_text(
            question_title_tag.get_text(" ", strip=True)
        )

        prompt_text = clean_text(
            body_tag.get_text("\n", strip=True)
        )

        track = major
        question_number = None

        match = re.search(r"#(\d+)", question_title)
        if match:
            question_number = int(match.group(1))
        else:
            question_number = len(questions) + 1

        # -----------------------------
        # Track handling
        # -----------------------------

        if major == "Computer Science & Engineering":
            if question_title.startswith("Computer Engineering"):
                track = "Computer Engineering"
            elif question_title.startswith("Computer Science"):
                track = "Computer Science"

        elif major == "Civil & Environmental Engineering":
            if question_title.startswith("Civil Engineering"):
                track = "Civil Engineering"
            elif question_title.startswith("Environmental Engineering"):
                track = "Environmental Engineering"

        # -----------------------------
        # Handle duplicated CS prompt text
        # -----------------------------

        if (
            major == "Computer Science & Engineering"
            and track == "Computer Science"
            and "Same as Computer Engineering above" in prompt_text
            and question_number in questions_by_number
        ):
            questions_by_number[question_number]["tracks"].append("Computer Science")
            continue

        # -----------------------------
        # Generic question object
        # -----------------------------

        question_data = {
            "question_title": normalize_question_title(question_title)
            if major == "Computer Science & Engineering"
            else question_title,
            "tracks": [track],
            "question_number": question_number,
            "text": prompt_text
        }

        questions.append(question_data)

        if major == "Computer Science & Engineering":
            questions_by_number[question_number] = question_data

    if questions:
        results.append({
            "major": major,
            "source_type": "official",
            "page_type": "engineering_essay_prompts",
            "canonical_url": canonical_urls.get(major),
            "scrape_url": scrape_url,
            "questions": questions
        })

with open("engineering_essay_prompts.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(json.dumps(results[:3], ensure_ascii=False, indent=2))