"""Scope rules shared by ingestion and existing-corpus cleanup."""

import re


def keep_section(page_type: str, heading: str) -> bool:
    heading = heading.lower().strip()
    if page_type == "requirements":
        return bool(re.search(r"courses required for admission|double major|restrictions on majoring", heading))
    if page_type in {"non_major_options", "double_degree_policy"}:
        return False
    if page_type == "double_major_policy":
        return heading in {"how to declare a major", "is a double major or double degree right for you?"}
    return not bool(re.search(
        r"^resources:?$|^more information$|why the allen school|"
        r"^allen school undergraduate programs$|similar uw majors|"
        r"multicultural outreach|residency office|international student services|"
        r"financial aid|nondiscrimination|equal opportunity|"
        r"how long does the program take|online or through distance|"
        r"math option change|^news$|^events$",
        heading,
    ))
