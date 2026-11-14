
from typing import Any, Dict, List, Optional


SEVERITY_PRIORITY = {
    5: "Critical",
    4: "High",
    3: "Medium",
    2: "Low",
    1: "Low",
}


def _get_severity(flaw: Dict[str, Any]) -> int:
    """Return severity between 1 and 5."""
    try:
        severity = int(float(flaw.get("severity", 1)))
        return max(1, min(5, severity))
    except (TypeError, ValueError):
        return 1


def _safe_time(value: Any) -> Optional[float]:
    """Convert timestamp to float safely."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_flaw_type(flaw: Dict[str, Any]) -> str:
    """Read flaw name from flaw_detection.py output."""
    return str(
        flaw.get("flaw")
        or flaw.get("flaw_type")
        or flaw.get("type")
        or flaw.get("title")
        or "General observation"
    ).strip()


def _clean_text(value: Any) -> str:
    """Convert evidence into readable text."""
    if value is None:
        return ""

    if isinstance(value, dict):
        return "; ".join(
            f"{key}: {_clean_text(item)}"
            for key, item in value.items()
            if item is not None
        )

    if isinstance(value, (list, tuple)):
        return "; ".join(
            _clean_text(item) for item in value
        )

    return str(value).strip()


def _get_times(flaw: Dict[str, Any]):
    """Read start and end timestamps."""
    start = _safe_time(
        flaw.get("start", flaw.get("start_time"))
    )

    end = _safe_time(
        flaw.get("end", flaw.get("end_time"))
    )

    return start, end


def _format_timestamp(value: Optional[float]) -> str:
    if value is None:
        return "Unknown"

    return f"{value:.2f}s"


def _is_vocal_expression(flaw: Dict[str, Any]) -> bool:
    """Identify pitch and vocal expression flaws."""
    flaw_name = _get_flaw_type(flaw).lower()

    keywords = (
        "pitch",
        "monotone",
        "vocal expression",
        "reduced pitch variation",
        "low pitch variation",
        "flat intonation",
        "intonation",
    )

    return any(
        keyword in flaw_name
        for keyword in keywords
    )


def _is_speaking_rhythm(flaw: Dict[str, Any]) -> bool:
    """Identify speech activity, rhythm and pacing flaws."""
    flaw_name = _get_flaw_type(flaw).lower()

    keywords = (
        "speech activity",
        "speech-activity",
        "speaking rhythm",
        "rhythm",
        "pacing",
        "pause",
        "silence",
        "speaking rate",
        "speech rate",
    )

    return any(
        keyword in flaw_name
        for keyword in keywords
    )


def _get_recommendation(flaw_type: str) -> Dict[str, str]:
    """Generate recommendations according to flaw category."""

    name = flaw_type.lower()

    if any(word in name for word in (
        "pitch",
        "monotone",
        "vocal expression",
        "intonation",
    )):
        return {
            "title": "Vocal Expression",
            "observation": (
                "Your voice shows limited pitch variation in some "
                "parts of the recording. This may make your delivery "
                "sound less expressive."
            ),
            "improvement_tip": (
                "Use natural changes in pitch to emphasize important "
                "words and make your delivery more engaging. Avoid "
                "speaking at the same pitch throughout the sentence."
            ),
            "practice_exercise": (
                "Read a short paragraph aloud three times. First, "
                "read it normally. Next, emphasize important words "
                "using pitch variation. Finally, record yourself and "
                "compare the two versions."
            ),
            "verification_note": (
                "Check whether pitch variation improves across the "
                "identified time windows in your next recording."
            ),
        }

    if any(word in name for word in (
        "speech activity",
        "speech-activity",
        "speaking rhythm",
        "rhythm",
        "pacing",
        "pause",
        "silence",
        "speaking rate",
        "speech rate",
    )):
        return {
            "title": "Speaking Rhythm",
            "observation": (
                "Your speaking activity differs from the reference "
                "pattern. This may indicate differences in pacing, "
                "pauses or overall speech rhythm."
            ),
            "improvement_tip": (
                "Maintain a natural and consistent speaking pace. "
                "Use short pauses between ideas instead of rushing "
                "through sentences or leaving unnecessarily long gaps."
            ),
            "practice_exercise": (
                "Read a 30-second paragraph aloud while recording "
                "yourself. Pause briefly between sentences and maintain "
                "a steady pace throughout the recording."
            ),
            "verification_note": (
                "Compare your speaking activity and pause pattern "
                "with the reference in your next recording."
            ),
        }

    if any(word in name for word in (
        "energy",
        "volume",
        "loudness",
    )):
        return {
            "title": "Voice Energy",
            "observation": (
                "Your vocal energy differs from the expected pattern "
                "in the identified section."
            ),
            "improvement_tip": (
                "Maintain a clear and comfortable volume. Avoid "
                "speaking too softly or changing loudness abruptly."
            ),
            "practice_exercise": (
                "Record a short paragraph while maintaining a steady "
                "and comfortable speaking volume."
            ),
            "verification_note": (
                "Check whether your voice energy becomes more "
                "consistent in the next recording."
            ),
        }

    return {
        "title": "Speech Review",
        "observation": (
            "A difference was detected in your speech. Review the "
            "supporting evidence to understand the identified section."
        ),
        "improvement_tip": (
            "Focus on clear, natural and controlled speech delivery."
        ),
        "practice_exercise": (
            "Record a short paragraph and review the identified "
            "section carefully."
        ),
        "verification_note": (
            "Compare the same section in your next recording."
        ),
    }


def _build_coaching(flaw: Dict[str, Any]) -> Dict[str, Any]:
    """Build a coaching card from one detected flaw."""

    flaw_type = _get_flaw_type(flaw)
    severity = _get_severity(flaw)
    start, end = _get_times(flaw)

    recommendation = _get_recommendation(flaw_type)

    evidence = _clean_text(flaw.get("evidence"))
    explanation = _clean_text(flaw.get("explanation"))

    evidence_parts = []

    if evidence:
        evidence_parts.append(evidence)

    if explanation:
        evidence_parts.append(explanation)

    evidence_text = " | ".join(evidence_parts)

    return {
        "title": recommendation["title"],
        "severity": severity,
        "priority": SEVERITY_PRIORITY.get(severity, "Low"),
        "start": start,
        "end": end,
        "observation": recommendation["observation"],
        "evidence": (
            evidence_text
            or "No additional acoustic evidence available."
        ),
        "improvement_tip": recommendation["improvement_tip"],
        "practice_exercise": recommendation["practice_exercise"],
        "verification_note": recommendation["verification_note"],
    }


def _combine_vocal_expression(
    vocal_flaws: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Combine all pitch detections into one coaching card."""

    coaching_items = [
        _build_coaching(flaw)
        for flaw in vocal_flaws
    ]

    severities = [
        item["severity"]
        for item in coaching_items
    ]

    starts = [
        item["start"]
        for item in coaching_items
        if item["start"] is not None
    ]

    ends = [
        item["end"]
        for item in coaching_items
        if item["end"] is not None
    ]

    evidence_lines = []

    for index, flaw in enumerate(vocal_flaws, start=1):
        start, end = _get_times(flaw)

        timestamp = (
            f"{_format_timestamp(start)} - "
            f"{_format_timestamp(end)}"
        )

        flaw_type = _get_flaw_type(flaw)
        severity = _get_severity(flaw)

        evidence = _clean_text(flaw.get("evidence"))
        explanation = _clean_text(flaw.get("explanation"))

        line = (
            f"{index}. [{timestamp}] {flaw_type} "
            f"(Severity: {severity}/5)"
        )

        if evidence:
            line += f"\n   Acoustic evidence: {evidence}"

        if explanation:
            line += f"\n   Explanation: {explanation}"

        evidence_lines.append(line)

    highest_severity = max(severities) if severities else 1

    return {
        "title": "Vocal Expression",
        "severity": highest_severity,
        "priority": SEVERITY_PRIORITY.get(
            highest_severity, "Low"
        ),
        "start": min(starts) if starts else None,
        "end": max(ends) if ends else None,
        "observation": (
            f"Limited pitch variation was detected across "
            f"{len(vocal_flaws)} time window(s). Review the "
            "individual timestamps below to identify where "
            "vocal expression can be improved."
        ),
        "evidence": "\n\n".join(evidence_lines),
        "improvement_tip": (
            "Use natural pitch variation to emphasize important "
            "words and make your delivery more expressive. Avoid "
            "maintaining the same pitch throughout your speech."
        ),
        "practice_exercise": (
            "Choose a 30-second paragraph and read it aloud three "
            "times. Emphasize keywords differently in each attempt. "
            "Record the attempts and compare your pitch variation."
        ),
        "verification_note": (
            "In your next recording, check whether pitch variation "
            "improves in the individual time windows listed above."
        ),
    }


def _combine_speaking_rhythm(
    rhythm_flaws: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Combine rhythm detections into one coaching card."""

    coaching_items = [
        _build_coaching(flaw)
        for flaw in rhythm_flaws
    ]

    severities = [
        item["severity"]
        for item in coaching_items
    ]

    starts = [
        item["start"]
        for item in coaching_items
        if item["start"] is not None
    ]

    ends = [
        item["end"]
        for item in coaching_items
        if item["end"] is not None
    ]

    evidence_lines = []

    for index, flaw in enumerate(rhythm_flaws, start=1):
        start, end = _get_times(flaw)

        timestamp = (
            f"{_format_timestamp(start)} - "
            f"{_format_timestamp(end)}"
        )

        flaw_type = _get_flaw_type(flaw)
        severity = _get_severity(flaw)

        evidence = _clean_text(flaw.get("evidence"))
        explanation = _clean_text(flaw.get("explanation"))

        line = (
            f"{index}. [{timestamp}] {flaw_type} "
            f"(Severity: {severity}/5)"
        )

        if evidence:
            line += f"\n   Acoustic evidence: {evidence}"

        if explanation:
            line += f"\n   Explanation: {explanation}"

        evidence_lines.append(line)

    highest_severity = max(severities) if severities else 1

    return {
        "title": "Speaking Rhythm",
        "severity": highest_severity,
        "priority": SEVERITY_PRIORITY.get(
            highest_severity, "Low"
        ),
        "start": min(starts) if starts else None,
        "end": max(ends) if ends else None,
        "observation": (
            f"Speech-activity or rhythm differences were detected "
            f"in {len(rhythm_flaws)} detection(s). Review the "
            "individual timestamps and supporting evidence below."
        ),
        "evidence": "\n\n".join(evidence_lines),
        "improvement_tip": (
            "Maintain a natural and consistent speaking pace. "
            "Use short pauses between ideas and avoid rushing "
            "through sentences."
        ),
        "practice_exercise": (
            "Read a 30-second paragraph aloud. Pause briefly "
            "between sentences and maintain a steady pace. "
            "Record yourself and review the timing."
        ),
        "verification_note": (
            "Compare your speaking activity and pause pattern "
            "with the reference in your next recording."
        ),
    }


def generate_speech_coaching(
    flaws: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Generate consolidated speech coaching recommendations.

    Pitch detections are combined into one Vocal Expression card.
    Rhythm detections are combined into one Speaking Rhythm card.
    Other flaw categories remain separate.
    """

    if not flaws:
        return []

    vocal_flaws = []
    rhythm_flaws = []
    other_flaws = []

    for flaw in flaws:
        if not isinstance(flaw, dict):
            continue

        if _is_vocal_expression(flaw):
            vocal_flaws.append(flaw)

        elif _is_speaking_rhythm(flaw):
            rhythm_flaws.append(flaw)

        else:
            other_flaws.append(flaw)

    coaching_cards = []

    if vocal_flaws:
        coaching_cards.append(
            _combine_vocal_expression(vocal_flaws)
        )

    if rhythm_flaws:
        coaching_cards.append(
            _combine_speaking_rhythm(rhythm_flaws)
        )

    for flaw in other_flaws:
        coaching_cards.append(
            _build_coaching(flaw)
        )

    coaching_cards.sort(
        key=lambda item: (
            -item["severity"],
            (
                item["start"]
                if item["start"] is not None
                else float("inf")
            ),
        )
    )

    return coaching_cards
