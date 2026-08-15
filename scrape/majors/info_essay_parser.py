import re
import json
import requests
from bs4 import BeautifulSoup

url = "https://ischool.uw.edu/programs/informatics/admissions/apply#essay"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

questions = []
seen = set()

prompt1 = soup.find("h3", string=lambda s: s and "Prompt 1" in s)

current_question = None
current_text = []
question_number = None

# include prompt1 itself
for tag in [prompt1] + list(prompt1.find_all_next()):

    # stop at next major section
    if tag.name == "h2":
        break

    # -----------------------------
    # New Prompt
    # -----------------------------
    if tag.name == "h3":

        # save previous prompt
        if current_question:
            questions.append({
                "question_title": current_question,
                "track": "Informatics",
                "question_number": question_number,
                "text": "\n\n".join(current_text).strip()
            })

        current_question = tag.get_text(strip=True)

        match = re.search(r"Prompt\s+(\d+)", current_question)

        question_number = int(match.group(1)) if match else None

        current_text = []

    # -----------------------------
    # Prompt Content
    # -----------------------------
    elif tag.name == "p":

        text = tag.get_text(" ", strip=True).replace("\xa0", " ")

        if text and text not in seen:
            current_text.append(text)
            seen.add(text)

# save final prompt
if current_question:
    questions.append({
        "question_title": current_question,
        "track": "Informatics",
        "question_number": question_number,
        "text": "\n\n".join(current_text).strip()
    })

result = {
    "major": "Informatics",
    "source_type": "official",
    "page_type": "essay_prompts",
    "url": url,
    "questions": questions
}

with open("info_essay_prompts.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(json.dumps(result, ensure_ascii=False, indent=2))