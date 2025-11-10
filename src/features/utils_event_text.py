def score_state(x: float) -> str:
    if x is None:
        return "unknown"
    try:
        x = float(x)
    except Exception:
        return "unknown"
    if x < 40: return "critical stress"
    if x < 60: return "moderate stress"
    return "normal / healthy"
