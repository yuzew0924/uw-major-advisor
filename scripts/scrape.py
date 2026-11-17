from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
import urllib3
from bs4 import BeautifulSoup, Tag
from urllib3.exceptions import InsecureRequestWarning

urllib3.disable_warnings(category=InsecureRequestWarning)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from rag.admissions_scope import keep_section
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "pages.json"

USER_AGENT = (
    "uw-major-advisor/0.1 "
    "(educational project; contact: https://github.com/)"
)

MIN_SECTION_CHARS = 80


@dataclass(frozen=True)
class PageSpec:
    major: str
    page_type: str
    url: str


PAGES: tuple[PageSpec, ...] = (
    PageSpec(
        "INFO",
        "admissions",
        "https://ischool.uw.edu/programs/informatics/admissions/current-students",
    ),
    PageSpec(
        "INFO",
        "requirements",
        "https://ischool.uw.edu/programs/informatics/curriculum",
    ),
    PageSpec(
        "INFO",
        "admissions_prerequisites",
        "https://ischool.uw.edu/programs/informatics/admissions/prerequisites",
    ),
    PageSpec(
        "INFO",
        "transfer_course_policy",
        "https://ischool.uw.edu/programs/informatics/admissions/informatics-transfer-course-policy",
    ),
    PageSpec(
        "CSE",
        "admissions",
        "https://www.cs.washington.edu/academics/undergraduate/admissions/",
    ),
    PageSpec(
        "CSE",
        "current_uw_admissions",
        "https://www.cs.washington.edu/academics/undergraduate/admissions/current-uw/",
    ),
    PageSpec(
        "CSE",
        "transfer_admissions",
        "https://www.cs.washington.edu/academics/undergraduate/admissions/transfers/",
    ),
    PageSpec(
        "CSE",
        "requirements",
        "https://www.cs.washington.edu/academics/undergraduate/degree-requirements/",
    ),
    PageSpec(
        "CSE",
        "non_major_options",
        "https://www.cs.washington.edu/academics/undergraduate/non-major-options/",
    ),
    PageSpec("ACMS", "admissions", "https://acms.washington.edu/admissions"),
    PageSpec(
        "ACMS",
        "requirements",
        "https://acms.washington.edu/program-requirements-and-information",
    ),
    PageSpec(
        "HCDE",
        "admissions",
        "https://www.hcde.washington.edu/bs/admissions/",
    ),
    PageSpec(
        "HCDE",
        "requirements",
        "https://www.hcde.washington.edu/bs/requirements/2024",
    ),
    PageSpec(
        "STATS",
        "admissions",
        "https://stat.uw.edu/academics/undergraduate/statistics-bs/application",
    ),
    PageSpec(
        "STATS",
        "requirements",
        "https://stat.uw.edu/academics/undergraduate/statistics-bs/major",
    ),
    PageSpec(
        "AMATH",
        "admissions",
        "https://amath.washington.edu/undergraduate-admissions",
    ),
    PageSpec(
        "AMATH",
        "requirements",
        "https://amath.washington.edu/undergraduate-major-applied-mathematics",
    ),
    PageSpec(
        "MATH",
        "admissions",
        "https://math.washington.edu/math-major-declaration",
    ),
    PageSpec(
        "MATH",
        "requirements",
        "https://math.washington.edu/bs-mathematics-major-requirements-0",
    ),
    PageSpec(
        "UW",
        "double_major_policy",
        "https://artsci.washington.edu/academics/declaring-a-major",
    ),
    PageSpec(
        "UW",
        "double_degree_policy",
        "https://registrar.washington.edu/staff-faculty/dars-advisers/exc-double-triple-degree/",
    ),
)


NOISE_SELECTORS = (
    "nav",
    "footer",
    "header",
    "aside",
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "form",
    "[role='navigation']",
    "[role='banner']",
    "[role='contentinfo']",
    ".breadcrumb",
    ".breadcrumbs",
    ".menu",
    ".navbar",
    ".navigation",
    ".pager",
    ".sidebar",
    ".site-footer",
    ".region-sidebar",
    ".uw-thinstrip",
)

TEXT_TAGS = {"p", "li", "td", "th", "caption", "blockquote"}
SECTION_HEADING_TAGS = {"h2", "h3"}
IGNORED_TEXT = {
    "back to top",
    "jump to a section",
    "expand all",
    "collapse all",
}


def clean_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_title(raw_title: str) -> str:
    parts = [part.strip() for part in raw_title.split("|") if part.strip()]
    if len(parts) > 1 and "University of Washington" in parts[-1]:
        return " | ".join(parts[:-1])
    return raw_title.strip()


def remove_noise(soup: BeautifulSoup) -> None:
    for selector in NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()


def content_root(soup: BeautifulSoup) -> Tag:
    for selector in ("main", "article", "[role='main']", "#main-content", ".main-content", ".content"):
        root = soup.select_one(selector)
        if root:
            return root
    if soup.body:
        return soup.body
    return soup


def get_page_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1:
        return clean_whitespace(h1.get_text(" ", strip=True))
    if soup.title and soup.title.string:
        return clean_title(clean_whitespace(soup.title.string))
    return "Untitled"


def table_to_text(table: Tag) -> str:
    rows = []
    for row in table.find_all("tr"):
        cells = [
            clean_whitespace(cell.get_text(" ", strip=True))
            for cell in row.find_all(["th", "td"])
        ]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def tag_text(tag: Tag) -> str:
    if tag.name == "table":
        return table_to_text(tag)
    return clean_whitespace(tag.get_text(" ", strip=True))


def is_ignored_text(text: str) -> bool:
    normalized = clean_whitespace(text).lower()
    return normalized in IGNORED_TEXT or normalized.startswith("button:")


def meaningful_sections(sections: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    results = []
    seen = set()

    for section in sections:
        heading = clean_whitespace(section["heading"])
        text = clean_whitespace(section["text"])

        if len(text) < MIN_SECTION_CHARS:
            continue

        dedupe_key = (heading.lower(), text[:250].lower())
        if dedupe_key in seen:
            continue

        results.append({"heading": heading, "text": text})
        seen.add(dedupe_key)

    return results


def extract_sections(root: Tag, title: str) -> list[dict[str, str]]:
    sections = []
    current_heading = title
    current_parts: list[str] = []
    seen_text = set()

    def flush() -> None:
        nonlocal current_parts
        text = "\n\n".join(current_parts)
        sections.append({"heading": current_heading, "text": text})
        current_parts = []

    for tag in root.find_all(["h2", "h3", "p", "li", "table", "blockquote"], recursive=True):
        if tag.name in SECTION_HEADING_TAGS:
            if current_parts:
                flush()
            current_heading = tag_text(tag)
            continue

        # Avoid duplicating list items that are already represented in their parent table.
        if tag.name in TEXT_TAGS and tag.find_parent("table"):
            continue

        text = tag_text(tag)
        if not text or is_ignored_text(text) or text in seen_text:
            continue

        current_parts.append(text)
        seen_text.add(text)

    if current_parts:
        flush()

    return meaningful_sections(sections)


def fetch_html(session: requests.Session, url: str) -> str:
    try:
        response = session.get(url, timeout=30)
    except requests.exceptions.SSLError:
        print(f"Warning: TLS verification failed for {url}; retrying without verification.")
        response = session.get(url, timeout=30, verify=False)

    response.raise_for_status()
    return response.text


def scrape_page(session: requests.Session, spec: PageSpec) -> dict[str, object]:
    html = fetch_html(session, spec.url)
    soup = BeautifulSoup(html, "lxml")
    remove_noise(soup)

    title = get_page_title(soup)
    root = content_root(soup)
    sections = [
        section for section in extract_sections(root, title)
        if keep_section(spec.page_type, section["heading"])
    ]

    return {
        "major": spec.major,
        "url": spec.url,
        "title": title,
        "page_type": spec.page_type,
        "sections": sections,
        "source_type": "official",
    }


def main() -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    records = [scrape_page(session, spec) for spec in PAGES]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    total_sections = sum(len(record["sections"]) for record in records)
    print(f"Wrote {len(records)} pages and {total_sections} sections to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
