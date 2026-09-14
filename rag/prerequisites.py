from __future__ import annotations

from typing import Any


INFORMATICS_REQUIREMENTS = {
    "required_courses": {
        "INFO200": "INFO 200",
    },
    "option_groups": {
        "programming": {
            "label": "one accepted programming course",
            "options": {
                "INFO180",
                "INFO201",
                "CSE121",
                "CSE122",
                "CSE123",
                "CSE142",
                "CSE143",
                "CSE154",
                "CSE160",
                "CSE163",
                "CSE180",
                "STAT180",
            },
        },
        "statistics": {
            "label": "one accepted statistics course",
            "options": {
                "STAT220",
                "STAT221",
                "BIOST310",
                "CSE312",
                "EDPSY490",
                "INDE315",
                "MATH390",
                "MATH394",
                "PSYCH315",
                "PSYCH317",
                "QMETH201",
                "QSCI381",
                "STAT290",
                "STAT311",
                "STAT390",
            },
        },
    },
    "non_course_requirements": {
        "social_sciences": "one Social Sciences (SSc) course other than INFO 200",
    },
}


MAJOR_REQUIREMENTS = {
    "INFO": INFORMATICS_REQUIREMENTS,
}


def normalize_course_code(course_code: str) -> str:
    return "".join(course_code.upper().split())


def display_course_code(course_code: str) -> str:
    normalized = normalize_course_code(course_code)
    for index, char in enumerate(normalized):
        if char.isdigit():
            return f"{normalized[:index]} {normalized[index:]}"
    return normalized


def check_requirements(
    major: str,
    completed_courses: set[str],
) -> dict[str, Any] | None:
    requirements = MAJOR_REQUIREMENTS.get(major)
    if not requirements:
        return None

    completed = {normalize_course_code(course) for course in completed_courses}
    satisfied: dict[str, list[str]] = {}
    missing: list[str] = []

    for course_code, label in requirements["required_courses"].items():
        if course_code in completed:
            satisfied[course_code] = [display_course_code(course_code)]
        else:
            missing.append(label)

    for group_name, group in requirements["option_groups"].items():
        matched_courses = sorted(completed & group["options"])
        if matched_courses:
            satisfied[group_name] = [
                display_course_code(course_code)
                for course_code in matched_courses
            ]
        else:
            missing.append(group["label"])

    missing.extend(requirements["non_course_requirements"].values())

    return {
        "major": major,
        "completed": [
            display_course_code(course_code)
            for course_code in sorted(completed)
        ],
        "satisfied": satisfied,
        "missing": missing,
    }
