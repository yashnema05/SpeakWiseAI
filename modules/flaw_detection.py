
import numpy as np


SEVERITY_LABELS = {
    1: "Very Low",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Critical",
}


def _safe_array(features, key):
    """Return a numeric NumPy array, or an empty array if unavailable."""
    try:
        value = np.asarray(features.get(key, []), dtype=float)
        return value[np.isfinite(value)]
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def _make_flaw(name, start, end, severity, evidence, explanation):
    start = max(0.0, float(start))
    end = max(start, float(end))
    severity = int(np.clip(severity, 1, 5))

    return {
        "flaw": name,
        "severity": severity,
        "severity_label": SEVERITY_LABELS[severity],
        "start": round(start, 2),
        "end": round(end, 2),
        "duration": round(end - start, 2),
        "evidence": evidence,
        "explanation": explanation,
        "status": "Detected",
    }


def _get_voiced_pitch(features):
    """Extract pitch values from frames with sufficient energy."""
    pitch = _safe_array(features, "pitch")

    try:
        pitch_times = np.asarray(
            features.get("pitch_times", []), dtype=float
        )
        energy = np.asarray(
            features.get("energy", []), dtype=float
        )
        energy_times = np.asarray(
            features.get("times", []), dtype=float
        )
    except (TypeError, ValueError):
        return np.array([])

    if (
        len(pitch) == 0
        or len(pitch_times) != len(pitch)
        or len(energy) == 0
        or len(energy_times) != len(energy)
    ):
        return np.array([])

    valid_energy = np.isfinite(energy) & np.isfinite(energy_times)

    if not np.any(valid_energy):
        return np.array([])

    interpolated_energy = np.interp(
        pitch_times,
        energy_times[valid_energy],
        energy[valid_energy],
    )

    max_energy = np.max(interpolated_energy)

    if max_energy <= 0:
        return np.array([])

    voiced_mask = (
        np.isfinite(pitch)
        & (pitch >= 65)
        & (pitch <= 400)
        & (interpolated_energy > max_energy * 0.12)
    )

    return pitch[voiced_mask]


# ---------------------------------------------------------
# REFERENCE-FREE INDIVIDUAL ANALYSIS
# ---------------------------------------------------------

def _detect_individual_long_pauses(features):
    """Detect long silent intervals in a single recording."""
    pauses = features.get("pause_intervals", [])
    duration = float(features.get("duration", 0) or 0)

    if duration <= 0 or not isinstance(pauses, (list, tuple)):
        return []

    detected = []

    for pause in pauses:
        if not isinstance(pause, dict):
            continue

        try:
            start = float(pause["start"])
            end = float(pause["end"])
            pause_duration = float(
                pause.get("duration", end - start)
            )
        except (KeyError, TypeError, ValueError):
            continue

        if not np.isfinite(start) or not np.isfinite(end):
            continue

        # Only flag pauses that are long enough to merit review.
        if pause_duration < 1.2:
            continue

        severity = 2
        if pause_duration >= 2.0:
            severity = 3
        if pause_duration >= 3.0:
            severity = 4
        if pause_duration >= 4.5:
            severity = 5

        detected.append(
            _make_flaw(
                name="Possible extended pause",
                start=start,
                end=end,
                severity=severity,
                evidence=(
                    f"Silent interval duration: {pause_duration:.2f}s"
                ),
                explanation=(
                    "A relatively long silent interval was detected. "
                    "Review this segment to check whether the pause "
                    "interrupts the natural flow of your speech."
                ),
            )
        )

    return detected


def _detect_individual_low_pitch_variation(features):
    """
    Flag unusually limited pitch variation as a preliminary indicator.

    This is a reference-free heuristic, not a speaker-independent
    diagnosis of monotone delivery.
    """
    pitch = _get_voiced_pitch(features)
    duration = float(features.get("duration", 0) or 0)

    if duration <= 0 or len(pitch) < 30:
        return []

    pitch_std = float(np.std(pitch))
    pitch_range = float(
        np.percentile(pitch, 90) - np.percentile(pitch, 10)
    )

    # Conservative absolute thresholds. Avoid flagging recordings
    # with insufficient voiced frames or uncertain pitch estimates.
    if pitch_std >= 10 or pitch_range >= 25:
        return []

    severity = 2
    if pitch_std < 5 and pitch_range < 12:
        severity = 3

    return [
        _make_flaw(
            name="Possible limited pitch variation",
            start=0,
            end=duration,
            severity=severity,
            evidence=(
                f"Voiced pitch standard deviation: {pitch_std:.2f} Hz; "
                f"10th–90th percentile pitch range: "
                f"{pitch_range:.2f} Hz; "
                f"valid voiced frames: {len(pitch)}"
            ),
            explanation=(
                "The estimated pitch variation is relatively limited "
                "in this recording. This may make some parts sound "
                "less expressive, but natural voice characteristics "
                "can also produce low pitch variation. Listen before "
                "interpreting this as monotone delivery."
            ),
        )
    ]


def detect_individual_flaws(features):
    """
    Detect preliminary delivery indicators from one recording.

    No reference recording is required. Results are heuristic and
    should be reviewed by listening to the original audio.
    """
    if not isinstance(features, dict):
        raise TypeError("features must be a dictionary.")

    flaws = []

    flaws.extend(_detect_individual_long_pauses(features))
    flaws.extend(_detect_individual_low_pitch_variation(features))

    return sorted(
        flaws,
        key=lambda item: (item["start"], item["end"])
    )


# ---------------------------------------------------------
# REFERENCE VS PARTICIPANT COMPARISON
# ---------------------------------------------------------

def _detect_extra_pauses(reference_features, participant_features):
    """Find unusually long pauses in the participant recording."""
    reference_pauses = reference_features.get("pause_intervals", [])
    participant_pauses = participant_features.get("pause_intervals", [])

    ref_duration = float(reference_features.get("duration", 0) or 0)
    participant_duration = float(
        participant_features.get("duration", 0) or 0
    )

    if participant_duration <= 0:
        return []

    detected = []

    for pause in participant_pauses:
        try:
            start = float(pause["start"])
            end = float(pause["end"])
            duration = float(pause.get("duration", end - start))
        except (KeyError, TypeError, ValueError):
            continue

        if duration < 0.8:
            continue

        nearest_reference_duration = 0.0
        nearest_distance = float("inf")

        if ref_duration > 0:
            participant_position = (
                (start + end) / 2
            ) / participant_duration

            for ref_pause in reference_pauses:
                try:
                    ref_start = float(ref_pause["start"])
                    ref_end = float(ref_pause["end"])
                    ref_pause_duration = float(
                        ref_pause.get("duration", ref_end - ref_start)
                    )
                except (KeyError, TypeError, ValueError):
                    continue

                ref_position = (
                    (ref_start + ref_end) / 2
                ) / ref_duration

                distance = abs(participant_position - ref_position)

                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_reference_duration = ref_pause_duration

        if nearest_distance <= 0.12:
            extra_duration = duration - nearest_reference_duration
            is_extra = extra_duration >= 0.4
        else:
            extra_duration = duration
            is_extra = duration >= 0.8

        if not is_extra:
            continue

        severity = 2
        if extra_duration >= 1.0:
            severity = 3
        if extra_duration >= 2.0:
            severity = 4
        if extra_duration >= 3.0:
            severity = 5

        detected.append(
            _make_flaw(
                name="Possible extended pause",
                start=start,
                end=end,
                severity=severity,
                evidence=(
                    f"Participant pause: {duration:.2f}s; "
                    f"reference pause: "
                    f"{nearest_reference_duration:.2f}s"
                ),
                explanation=(
                    "The participant has a noticeably longer silent "
                    "interval than the corresponding reference region. "
                    "This may interrupt speech flow."
                ),
            )
        )

    return detected


def _detect_reduced_pitch_variation(
    reference_features, participant_features
):
    """Conservatively detect unusually low pitch variation."""
    ref_pitch = _get_voiced_pitch(reference_features)
    participant_pitch = _get_voiced_pitch(participant_features)

    if len(ref_pitch) < 20 or len(participant_pitch) < 20:
        return []

    ref_std = float(np.std(ref_pitch))
    participant_std = float(np.std(participant_pitch))

    if ref_std < 12:
        return []

    ratio = participant_std / ref_std

    if ratio >= 0.55:
        return []

    participant_duration = float(
        participant_features.get("duration", 0) or 0
    )

    if participant_duration <= 0:
        return []

    severity = 3
    if ratio < 0.25:
        severity = 4

    return [
        _make_flaw(
            name="Possible reduced pitch variation",
            start=0,
            end=participant_duration,
            severity=severity,
            evidence=(
                f"Reference pitch standard deviation: {ref_std:.2f} Hz; "
                f"participant: {participant_std:.2f} Hz; "
                f"variation ratio: {ratio:.2f}"
            ),
            explanation=(
                "The participant's voiced pitch varies considerably "
                "less than the reference. This may indicate a more "
                "monotone delivery."
            ),
        )
    ]


def _detect_speech_activity_difference(
    reference_features, participant_features
):
    """Detect substantial differences in one-second speech activity."""
    ref_activity = _safe_array(reference_features, "speech_activity")
    participant_activity = _safe_array(
        participant_features, "speech_activity"
    )

    if len(ref_activity) < 3 or len(participant_activity) < 3:
        return []

    common_length = min(len(ref_activity), len(participant_activity))

    ref_values = ref_activity[:common_length]
    participant_values = participant_activity[:common_length]

    differences = np.abs(ref_values - participant_values)
    mean_difference = float(np.mean(differences))

    if mean_difference < 0.25:
        return []

    participant_duration = float(
        participant_features.get("duration", 0) or 0
    )

    if participant_duration <= 0:
        return []

    severity = 2
    if mean_difference >= 0.40:
        severity = 3

    return [
        _make_flaw(
            name="Possible speech-activity difference",
            start=0,
            end=participant_duration,
            severity=severity,
            evidence=(
                f"Mean speech-activity difference: "
                f"{mean_difference:.2f}"
            ),
            explanation=(
                "Speech activity differs noticeably from the reference. "
                "This may indicate differences in speaking rhythm "
                "or delivery."
            ),
        )
    ]


def detect_flaws(reference_features, participant_features):
    """
    Compare reference and participant acoustic features.

    Returns a list of detected flaw dictionaries.
    """
    if not isinstance(reference_features, dict):
        raise TypeError("reference_features must be a dictionary.")

    if not isinstance(participant_features, dict):
        raise TypeError("participant_features must be a dictionary.")

    flaws = []

    pause_flaws = _detect_extra_pauses(
        reference_features, participant_features
    )
    flaws.extend(pause_flaws)

    pitch_flaws = _detect_reduced_pitch_variation(
        reference_features, participant_features
    )
    flaws.extend(pitch_flaws)

    if not pause_flaws:
        activity_flaws = _detect_speech_activity_difference(
            reference_features, participant_features
        )
        flaws.extend(activity_flaws)

    return sorted(
        flaws,
        key=lambda item: (item["start"], item["end"])
    )
