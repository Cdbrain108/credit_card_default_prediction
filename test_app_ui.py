"""Headless smoke test for the two-tab Streamlit app."""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=300)
at.run()

# Navigate to inference through the app's left sidebar.
navigation = [b for b in at.button if "Make a prediction" in b.label]
if navigation:
    navigation[0].click().run()

print("PAGE load      :", "OK" if not at.exception else f"EXCEPTION -> {at.exception}")
print("buttons        :", [b.label for b in at.button][:8])
print("dataframes     :", len(at.dataframe))
print("selectboxes    :", [s.label for s in at.selectbox][:6])

# Apply the high-risk preset (this also clears any previous prediction).
preset = [b for b in at.button if "High-risk" in b.label]
if preset:
    preset[0].click().run()
    print("PRESET         :", "OK" if not at.exception else f"EXCEPTION -> {at.exception}")

# Submit the form.
submit = [b for b in at.button if "Predict default risk" in b.label]
if submit:
    submit[0].click().run()
    print("FORM SUBMIT    :", "OK" if not at.exception else f"EXCEPTION -> {at.exception}")
else:
    print("FORM SUBMIT    : button not found")

markdown_text = " ".join(m.value for m in at.markdown)
checks = {
    "verdict": "probability of default" in markdown_text.lower(),
    "models table": len(at.dataframe) >= 2,
    "accuracy": "Test accuracy" in markdown_text or "test accuracy" in markdown_text,
}
print("checks         :", checks)
print("exceptions     :", at.exception)
print("DONE")
