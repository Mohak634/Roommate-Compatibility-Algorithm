"""Short profile bios built from survey answers."""

BEDTIME = {"Early (Before 10 PM)": "Early riser", "Moderate (10PM - 12AM)": "In bed by midnight",
           "Late (After 12AM)": "Night owl"}
SOCIAL = {"Introverted (Prefer minimal social interaction)": "enjoys quiet evenings",
          "Ambiverted (Balanced)": "up for plans but likes downtime",
          "Extroverted (Very social and outgoing)": "always up for something"}
CLEAN = {"Daily": "keeps things tidy", "Weekly": "does a weekly clean-up",
         "Occasionally": "relaxed about mess", "Rarely": "very relaxed about mess"}


def bio_from_answers(a: dict) -> str | None:
    try:
        hobbies = list(a.get("Hobbies") or [])
        parts = [f"{a['Year of Study']} {a['Course of Study']} student.",
                 f"{BEDTIME[a['Bedtime']]}, {SOCIAL[a['Social Preference']]} and {CLEAN[a['Cleaning Frequency']]}."]
        if len(hobbies) >= 2:
            parts.append(f"Into {hobbies[0].lower()} and {hobbies[1].lower()}.")
        elif hobbies:
            parts.append(f"Into {hobbies[0].lower()}.")
        return " ".join(parts)
    except (KeyError, TypeError):
        return None
