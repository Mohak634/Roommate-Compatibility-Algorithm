"""Survey question schema served to the Flutter app.

Options come straight from main_script's ORDINAL_MAPPINGS / CATEGORY_OPTIONS,
so the form vocabulary can never drift from what encode_data understands.
Question labels mirror the original Google Form wording (clean_data).

Question types: scale (int slider), choice (single select), multi (multi
select, maxSelections), number (free int), text (free string).
"""

from app.core.main_script import CATEGORY_OPTIONS, HOBBY_OPTIONS, ORDINAL_MAPPINGS


def _ordinal_options(key):
    """Options for an ordinal question, in encoded order."""
    return [k for k, _ in sorted(ORDINAL_MAPPINGS[key].items(), key=lambda kv: kv[1])]


def get_survey_schema():
    hobbies = [h for h in HOBBY_OPTIONS if h != "Empty"]
    return {
        "sections": [
            "Basics",
            "Sleep",
            "Cleanliness",
            "Study",
            "Social",
            "Guests & Space",
            "About You",
        ],
        "questions": [
            # --- Basics ---
            {"key": "Full Name", "section": "Basics", "type": "text",
             "label": "Full name"},
            {"key": "Age", "section": "Basics", "type": "number",
             "label": "Age", "min": 16, "max": 60},
            {"key": "Gender", "section": "Basics", "type": "choice",
             "label": "Gender", "options": _ordinal_options("Gender")},
            {"key": "Course of Study", "section": "Basics", "type": "choice",
             "label": "Course of study",
             "options": CATEGORY_OPTIONS["Course of Study"]},
            {"key": "Year of Study", "section": "Basics", "type": "choice",
             "label": "Year of study", "options": _ordinal_options("Year of Study")},

            # --- Sleep ---
            {"key": "Bedtime", "section": "Sleep", "type": "choice",
             "label": "What is your typical bedtime?",
             "options": _ordinal_options("Bedtime")},
            {"key": "Noise Tolerance Sleep", "section": "Sleep", "type": "scale",
             "label": "How much does noise during your sleep bother you?",
             "help": "1 = doesn't bother me, 10 = bothers me a lot",
             "min": 1, "max": 10},

            # --- Cleanliness ---
            {"key": "Cleanliness Scale", "section": "Cleanliness", "type": "scale",
             "label": "How important is cleanliness to you?",
             "help": "1 = not important, 10 = extremely important",
             "min": 1, "max": 10},
            {"key": "Cleaning Frequency", "section": "Cleanliness", "type": "choice",
             "label": "How often do you clean your personal space?",
             "options": _ordinal_options("Cleaning Frequency")},

            # --- Study ---
            {"key": "Study Frequency", "section": "Study", "type": "choice",
             "label": "How often do you study?",
             "options": _ordinal_options("Study Frequency")},
            {"key": "Study Location", "section": "Study", "type": "choice",
             "label": "Where do you prefer to study?",
             "options": CATEGORY_OPTIONS["Study Location"]},
            {"key": "Noise Tolerance Study", "section": "Study", "type": "scale",
             "label": "How tolerant are you of noise or activity while studying?",
             "help": "1 = need silence, 10 = noise is fine",
             "min": 1, "max": 10},

            # --- Social ---
            {"key": "Social Preference", "section": "Social", "type": "choice",
             "label": "How social are you?",
             "options": _ordinal_options("Social Preference")},
            {"key": "Party Frequency", "section": "Social", "type": "choice",
             "label": "How often do you attend or host parties?",
             "options": _ordinal_options("Party Frequency")},
            {"key": "Mutual Friends", "section": "Social", "type": "choice",
             "label": "How many mutual friends do you and your roommate share?",
             "options": _ordinal_options("Mutual Friends")},

            # --- Guests & Space ---
            {"key": "Roommate Host", "section": "Guests & Space", "type": "choice",
             "label": "Are you comfortable with your roommate hosting friends or gatherings?",
             "options": CATEGORY_OPTIONS["Roommate Host"]},
            {"key": "Personal Space", "section": "Guests & Space", "type": "choice",
             "label": "How much personal space do you need?",
             "options": _ordinal_options("Personal Space")},

            # --- About You ---
            {"key": "Hobbies", "section": "About You", "type": "multi",
             "label": "What are your top 3 hobbies?",
             "options": hobbies, "maxSelections": 3},
            {"key": "Roommate Compatibility Rating", "section": "About You",
             "type": "scale",
             "label": "How would you rate your overall compatibility as a roommate?",
             "help": "Honest self-rating, 1-10",
             "min": 1, "max": 10},
        ],
    }
