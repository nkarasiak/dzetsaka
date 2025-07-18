# dzetsaka Anniversary Feedback Collection

## Overview
This document outlines the multiple feedback collection methods implemented for dzetsaka's 10th anniversary preparation.

## Feedback Methods Available

### 1. 📊 Quick Poll (Structured Feedback)
**Purpose:** Collect quantitative data about user preferences and priorities

**Current Implementation:** GitHub Discussions with poll category
**URL:** `https://github.com/nkarasiak/dzetsaka/discussions/new?category=polls`

**Alternative Poll Services You Can Use:**

#### Google Forms
- **Pros:** Free, easy to use, great analytics, works offline
- **Setup:** Create form at forms.google.com
- **Integration:** Replace URL in `_open_quick_poll()` method
- **Example URL:** `https://forms.gle/YOUR_FORM_ID_HERE`

#### Microsoft Forms
- **Pros:** Professional look, Office 365 integration, real-time responses
- **Setup:** Create form at forms.office.com
- **Integration:** Replace URL in `_open_quick_poll()` method
- **Example URL:** `https://forms.office.com/YOUR_FORM_ID`

#### SurveyMonkey
- **Pros:** Advanced survey logic, professional features, great analytics
- **Setup:** Create survey at surveymonkey.com
- **Integration:** Replace URL in `_open_quick_poll()` method
- **Example URL:** `https://www.surveymonkey.com/r/YOUR_SURVEY_ID`

#### Typeform
- **Pros:** Beautiful interactive forms, conversational style
- **Setup:** Create form at typeform.com
- **Integration:** Replace URL in `_open_quick_poll()` method
- **Example URL:** `https://YOUR_ACCOUNT.typeform.com/to/YOUR_FORM_ID`

### 2. 💬 GitHub Discussions (Community Forum)
**Purpose:** Open discussions, brainstorming, community interaction
**URL:** `https://github.com/nkarasiak/dzetsaka/discussions`

**Benefits:**
- Threaded conversations
- Community voting on ideas
- Long-term searchable archive
- Integration with GitHub ecosystem

### 3. 🐛 GitHub Issues (Structured Requests)
**Purpose:** Specific feature requests, bug reports, technical suggestions
**URL:** `https://github.com/nkarasiak/dzetsaka/issues`

**Benefits:**
- Trackable and manageable
- Labels and milestones for organization
- Developer-friendly format
- Version control integration

## Recommended Poll Questions

Here are some suggested questions for your structured poll:

### Priority Features
- Which type of new classifier would you like to see most?
- What's your biggest pain point with current dzetsaka workflow?
- Which UI improvement would help you most?

### Usage Patterns
- How often do you use dzetsaka?
- What's your primary use case?
- Which algorithms do you use most?

### Technical Preferences
- Do you prefer GUI or command-line workflows?
- How important is processing speed vs accuracy?
- Would you use cloud/remote processing features?

### Future Features
- Interest in automated parameter tuning?
- Need for batch processing improvements?
- Desire for integration with other tools?

## Implementation Instructions

### To Set Up Google Forms Poll:

1. Go to forms.google.com
2. Create a new form with anniversary questions
3. Get the shareable link (forms.gle/YOUR_ID)
4. Update the URL in `anniversary_widget.py`:

```python
def _open_quick_poll(self):
    webbrowser.open("https://forms.gle/YOUR_ACTUAL_FORM_ID")
```

### To Set Up Custom Survey Platform:

1. Choose your preferred platform from the list above
2. Create your survey with relevant questions
3. Get the public link to your survey
4. Update the `_open_quick_poll()` method with your URL

## Analytics and Data Collection

### What You Can Track:
- Response rates by feedback method
- Most requested features
- User demographics (if collected)
- Geographic distribution of users
- Preferred interaction methods

### Privacy Considerations:
- Keep surveys anonymous unless users opt-in
- Follow GDPR guidelines for EU users
- Clearly state data usage policies
- Provide opt-out mechanisms

## Integration Timeline

**Phase 1 (Current):** Basic implementation with GitHub integration
**Phase 2 (Recommended):** Set up dedicated survey platform
**Phase 3 (Advanced):** Analyze data and prioritize features for anniversary release

## Technical Notes

The popup system will:
- Show once per day during the collection period (July 17, 2025 - May 17, 2026)
- Remember user preferences (don't show again)
- Track which feedback method users prefer
- Provide easy access to all feedback channels

This multi-channel approach ensures you can gather both quantitative data (polls) and qualitative feedback (discussions/issues) to make informed decisions for the anniversary release.