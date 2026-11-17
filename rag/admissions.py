from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, validator


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "admissions"
COURSE = re.compile(
    r"(?<![a-z0-9])([a-z]{1,8}(?:[ ]+[a-z]{1,5})?)[ -]*(1xx|\d{3}[a-z]?)(?![a-z0-9])",
    re.I,
)
Campus = Literal["Seattle", "Bothell", "Tacoma"]


def normalize_course(code: str) -> str:
    match = COURSE.fullmatch(code.strip())
    if not match:
        raise ValueError(f"Invalid course code: {code}")
    return f"{' '.join(match[1].upper().split())} {match[2].upper()}"


class Course(BaseModel):
    course: str
    campus: Campus

    _normalize = validator("course", allow_reuse=True)(normalize_course)


class Option(BaseModel):
    all_of: list[Course]

    @validator("all_of")
    def nonempty(cls, courses):
        if not courses:
            raise ValueError("A course option must contain at least one course")
        return courses


class Requirement(BaseModel):
    id: str
    label: str
    options: list[Option] = Field(default_factory=list)
    designation: str | None = None
    excluded_courses: list[str] = Field(default_factory=list)
    applicant_types: list[str] | None = None

    _normalize = validator("excluded_courses", each_item=True, allow_reuse=True)(normalize_course)


class Policy(BaseModel):
    applicant_types: list[str]
    effective_from: date | None = None
    effective_until: date | None = None
    source_url: HttpUrl
    requirements: list[Requirement]


class PolicyFile(BaseModel):
    major_id: str
    major_campus: Campus
    policies: list[Policy]


class Major(BaseModel):
    id: str
    name: str
    campus: Campus
    aliases: list[str]
    status: Literal["verified", "pending_review", "no_course_prerequisites"]
    policy_file: str | None = None


class Admissions:
    def __init__(self, directory: Path):
        self.majors = {
            major.id: major for major in
            (Major(**item) for item in json.loads((directory / "majors.json").read_text()))
        }
        self.policies_by_major: dict[str, list[Policy]] = {}
        self.options_by_course: dict[str, list[tuple[str, Policy, Requirement, Option]]] = defaultdict(list)
        for major in self.majors.values():
            if major.status == "pending_review" or not major.policy_file:
                continue
            path = directory / major.policy_file
            if not path.is_file():
                continue
            data = PolicyFile(**json.loads(path.read_text()))
            if (data.major_id, data.major_campus) != (major.id, major.campus):
                raise ValueError(f"Policy identity does not match {major.id}")
            self.policies_by_major[major.id] = data.policies
            for policy in data.policies:
                for requirement in policy.requirements:
                    for option in requirement.options:
                        for course in option.all_of:
                            self.options_by_course[course.course].append(
                                (major.id, policy, requirement, option)
                            )

    def extract_courses(self, question: str) -> set[str]:
        known_subjects = {code.rsplit(" ", 1)[0] for code in self.options_by_course}
        courses = set()
        for match in COURSE.finditer(question):
            subject = match[1].upper()
            if subject not in known_subjects and " " in subject:
                subject = subject.split()[-1]
            courses.add(normalize_course(f"{subject} {match[2]}"))
        return courses

    def answer(self, question: str, *, today: date | None = None) -> str | None:
        q = question.lower()
        if not re.search(r"prerequisit|入学先修|申请先修|先修要求", q):
            return None
        courses = self.extract_courses(question)
        without_courses = COURSE.sub(" ", q)
        majors = [
            major for major in self.majors.values()
            if any(re.search(rf"(?<![a-z0-9]){re.escape(alias.lower())}(?![a-z0-9])", without_courses)
                   for alias in [major.name, *major.aliases])
        ]
        reverse = bool(courses and re.search(r"which majors|哪些\s*(?:major|专业)", q))
        if not reverse and len(majors) != 1:
            if re.fullmatch(r"info[ -]*200\s+prerequisites?\s*\??", q.strip()):
                return "你想查 INFO 200 本身的选课先修条件，还是它能满足哪个专业的入学要求？"
            return None
        if re.search(r"\b(?:freshman|first.year|direct.to.major)\b|新生直录", q):
            return None
        if courses and not reverse and not re.search(
            r"satisf|count|fulfill|accept|qualif|满足|算作|抵|认可", q
        ):
            return None
        applicant = None
        if re.search(r"already transferred|current uw|在读|已转入", q):
            applicant = "current_uw_student"
        elif re.search(r"\btransfer\b|申请转入|转学申请", q):
            applicant = "transfer_applicant"
        today = today or date.today()
        selected = set(self.majors) if reverse else {majors[0].id}
        active = {
            major_id: [policy for policy in policies if
                       (not policy.effective_from or policy.effective_from <= today)
                       and (not policy.effective_until or today <= policy.effective_until)
                       and (not applicant or applicant in policy.applicant_types)]
            for major_id, policies in self.policies_by_major.items() if major_id in selected
        }
        results: set[str] = set()
        if courses:
            for code in sorted(courses):
                for major_id, policy, requirement, option in self.options_by_course.get(code, []):
                    if policy not in active.get(major_id, []):
                        continue
                    if applicant and requirement.applicant_types and applicant not in requirement.applicant_types:
                        if requirement.id == "info200":
                            results.add(f"- INFO 200：同步申请转入 UW 的申请人可在入读后修读。[官网]({policy.source_url})")
                        continue
                    missing = [item.course for item in option.all_of if item.course not in courses]
                    if missing and any(
                        all(item.course in courses for item in alternative.all_of)
                        for alternative in requirement.options
                    ):
                        continue
                    campus = next(item.campus for item in option.all_of if item.course == code)
                    outcome = f"属于组合选项，还需 {' + '.join(missing)}" if missing else "满足该项的课程选项"
                    scope = ""
                    if requirement.applicant_types and not applicant:
                        scope = "（UW Seattle 在读生申请时须完成。同步申请转入 UW 的申请人可入读后修读。）"
                    results.add(
                        f"- {code}（{campus}）→ {self.majors[major_id].name} · "
                        f"**{requirement.label}**：{outcome}。{scope} [官网]({policy.source_url})"
                    )
            if reverse:
                covered = sum(bool(policies) for policies in active.values())
                matches = "\n".join(sorted(results)) or "已核对的课程选项中没有找到匹配。"
                return f"{matches}\n\n结构化覆盖：本地清单中 {covered}/{len(self.majors)} 个专业有适用政策。清单目前只收录首批专业。"
            if not results:
                return None
        else:
            for major_id, policies in active.items():
                for policy in policies:
                    for requirement in policy.requirements:
                        if applicant and requirement.applicant_types and applicant not in requirement.applicant_types:
                            results.add("- **INFO 200**：同步申请转入 UW 的申请人可入读后修读。")
                            continue
                        options = [
                            " + ".join(f"{item.course}（{item.campus}）" for item in option.all_of)
                            for option in requirement.options
                        ]
                        if requirement.designation:
                            options.append(f"带 {requirement.designation} 标记的课程，排除 {', '.join(requirement.excluded_courses)}")
                        scope = "（UW Seattle 在读生须在申请前完成，同步申请转入 UW 者可入读后修读）" if requirement.applicant_types else ""
                        results.add(f"- **{requirement.label}**{scope}：{'、'.join(options)}")
                    results.add(f"[{self.majors[major_id].name} 官方先修要求]({policy.source_url})")
        return "\n".join(sorted(results)) if results else None


@lru_cache(maxsize=1)
def admissions() -> Admissions:
    return Admissions(DATA_DIR)


def fallback_query(question: str) -> str:
    if re.search(r"prerequisit|入学先修|申请先修|先修要求", question, re.I) and re.search(
        r"(?<![a-z0-9])informatics(?![a-z0-9])|信息学", question, re.I
    ):
        return f"{question} Informatics admission prerequisites"
    return question
