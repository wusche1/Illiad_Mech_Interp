import re
import subprocess
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

ROOT = Path(__file__).resolve().parents[2]
QUIZ = ROOT / "quiz" / "quiz.md"
FORM_ID = ROOT / "quiz" / "form_id.txt"
SHA = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
RAW = f"https://raw.githubusercontent.com/wusche1/Illiad_Mech_Interp/{SHA}/"
TOKEN = ROOT.parent / "steganography-benchmark-paper" / "scripts" / ".forms_token.json"
SCOPES = ["https://www.googleapis.com/auth/forms.body", "https://www.googleapis.com/auth/forms.responses.readonly"]


def parse():
    text = QUIZ.read_text()
    title = re.search(r"^# (.+)$", text, re.M).group(1)
    head, *questions = re.split(r"^## ", text.split("\n", 1)[1], flags=re.M)
    parsed = []
    for q in questions:
        lines = q.strip().splitlines()
        options = [(l[6:], l[3] == "x") for l in lines[1:] if l.startswith("- [")]
        parsed.append((lines[0], options))
    return title, head.strip(), parsed


def option(text):
    m = re.fullmatch(r"!\[(.+)\]\((.+)\)", text)
    return {"value": m.group(1), "image": {"sourceUri": RAW + m.group(2)}} if m else {"value": text}


def question_item(title, options):
    if not options:
        return {"title": title, "questionItem": {"question": {"textQuestion": {"paragraph": True}}}}
    correct = [{"value": option(o)["value"]} for o, ok in options if ok]
    kind = "CHECKBOX" if len(correct) > 1 else "RADIO"
    return {"title": title, "questionItem": {"question": {
        "grading": {"pointValue": len(correct), "correctAnswers": {"answers": correct}},
        "choiceQuestion": {"type": kind, "options": [option(o) for o, _ in options]},
    }}}


def push():
    c = Credentials.from_authorized_user_file(TOKEN, SCOPES)
    if not c.valid:
        c.refresh(Request())
    svc = build("forms", "v1", credentials=c)
    title, description, questions = parse()
    form_id = FORM_ID.read_text().strip()
    items = svc.forms().get(formId=form_id).execute().get("items", [])
    requests = [{"deleteItem": {"location": {"index": 0}}} for _ in items]
    requests.append({"updateFormInfo": {"info": {"title": title, "description": description}, "updateMask": "title,description"}})
    requests += [{"createItem": {"item": question_item(t, o), "location": {"index": i}}} for i, (t, o) in enumerate(questions)]
    svc.forms().batchUpdate(formId=form_id, body={"requests": requests}).execute()
    form = svc.forms().get(formId=form_id).execute()
    print("edit:   ", f"https://docs.google.com/forms/d/{form_id}/edit")
    print("respond:", form["responderUri"])


if __name__ == "__main__":
    push()
