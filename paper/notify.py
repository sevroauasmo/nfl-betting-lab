"""Post live-trader alerts as GitHub issues. One issue per trigger (later orders and the settled result are added as
comments on it); errors share one issue. The owner is @-mentioned so GitHub notifies them by email / mobile app.
Also pushes each alert to the ntfy phone app (ntfy.sh) when the NTFY_TOPIC secret is set -- GitHub's own notifications
for bot-posted issues are unreliable. Pick a long random topic name: anyone who knows it can read the alerts.
Reads JSON lines {key, title, body} from $ALERTS_FILE. Needs GITHUB_TOKEN (issues: write) and GITHUB_REPOSITORY."""
import json
import os
import sys

import requests

API = f"https://api.github.com/repos/{os.environ.get('GITHUB_REPOSITORY', '')}"
H = {"Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN', '')}", "Accept": "application/vnd.github+json"}
OWNER = os.environ.get("GITHUB_REPOSITORY_OWNER", "")


NTFY = os.environ.get("NTFY_TOPIC", "")
SEEN = {}                                                         # key -> issue number created earlier in this run


def find(key):
    """Issue whose body carries this alert key (hidden marker), newest first."""
    if key in SEEN:
        return SEEN[key]
    r = requests.get(f"{API}/issues", headers=H, params={"state": "all", "per_page": 100}, timeout=30)
    r.raise_for_status()
    return next((i["number"] for i in r.json() if f"<!-- key:{key} -->" in (i.get("body") or "")), None)


def push(title, body, num):
    """Phone push via ntfy; never fails the run."""
    if not NTFY:
        return
    try:
        link = f"https://github.com/{os.environ.get('GITHUB_REPOSITORY', '')}/issues/{num}" if num else None
        r = requests.post("https://ntfy.sh/", timeout=20, json={"topic": NTFY, "title": title[:200], "message": body[:3500],
                                                               "markdown": True, **({"click": link} if link else {})})
        print("ntfy", r.status_code)
    except Exception as e:                                        # noqa: BLE001
        print("ntfy failed", e, file=sys.stderr)


def main():
    path = os.environ.get("ALERTS_FILE")
    if not path or not os.path.exists(path):
        return
    for line in open(path):
        a = json.loads(line)
        num = find(a["key"])
        if num:
            r = requests.post(f"{API}/issues/{num}/comments", headers=H, json={"body": f"@{OWNER} {a['body']}"}, timeout=30)
        else:
            title = a["title"] or f"[trade] {a['key']}"
            r = requests.post(f"{API}/issues", headers=H, timeout=30,
                              json={"title": title, "body": f"@{OWNER}\n\n{a['body']}\n\n<!-- key:{a['key']} -->",
                                    "assignees": [OWNER]})              # assignment = a second native notification reason
        if not num and r.status_code < 300:
            SEEN[a["key"]] = r.json()["number"]
        print(a["key"], r.status_code)
        push(a["title"] or f"NFL bot: {a['key']}", a["body"], num or SEEN.get(a["key"]))
        if r.status_code >= 300:
            print(r.text[:300], file=sys.stderr)


if __name__ == "__main__":
    main()
