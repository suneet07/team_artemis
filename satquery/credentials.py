"""Load credentials from a dotenv file that lives *outside* the repository.

The standing rule is that credentials are never read from a file in this repo.
That rule is not paranoia about ``.gitignore`` -- ``.gitignore`` works fine. It
is about everything else that copies a working tree:

* ``scripts/modal_phase0.py`` mounts the whole repo with ``add_local_dir``. A
  repo-local ``.env`` would be uploaded into every Modal container, sit in the
  image layer, and reach any workspace member who can read the app.
* zipping the folder for a submission, a backup, or a teammate takes ignored
  files with it. ``.gitignore`` only governs git.

So the file lives at ``~/.satquery/credentials.env`` by default, or wherever
``SATQUERY_ENV_FILE`` points. Either way it is outside the tree that gets
copied, and :func:`load_credentials` refuses to read one from inside the repo.

The shell always wins. A variable already set in the environment is never
overwritten by the file, so a one-off ``$env:CDSE_PASSWORD`` still overrides
whatever is on disk without editing anything.

Format is the usual one, parsed here rather than pulling in ``python-dotenv``::

    # comments and blank lines are ignored
    CDSE_USERNAME=someone@example.org
    CDSE_PASSWORD="quotes are optional and stripped"

No interpolation, no ``export`` prefix handling, no multi-line values. A
credential file that needs those features is doing something this project does
not do.
"""

import os
from pathlib import Path

__all__ = ["DEFAULT_ENV_FILE", "credential_source", "load_credentials", "parse_env_file"]

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Where the credential file lives unless ``SATQUERY_ENV_FILE`` says otherwise.
DEFAULT_ENV_FILE = Path.home() / ".satquery" / "credentials.env"

#: The names this project reads. Listed so that :func:`credential_source` can
#: report where each one came from without the caller naming them again.
CREDENTIAL_KEYS = (
    "CDSE_USERNAME",
    "CDSE_PASSWORD",
    "CDSE_S3_ACCESS_KEY",
    "CDSE_S3_SECRET_KEY",
    "HF_TOKEN",
    "WANDB_API_KEY",
)


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a dotenv file into a dict. Malformed lines raise rather than skip.

    A silently ignored line is how a credential file ends up half-loaded and
    the failure surfaces later as an authentication error against the wrong
    account.
    """
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(
                f"{path}:{number}: expected NAME=value, got {line!r}. Fix the line "
                "rather than leaving it -- a skipped line loads half a credential set."
            )
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name] = value
    return values


def load_credentials(path: str | Path | None = None, *, override: bool = False) -> list[str]:
    """Populate ``os.environ`` from the credential file. Returns names loaded.

    Missing file is not an error -- the shell alone is a perfectly good source,
    and CI has no file. Returns an empty list in that case.
    """
    target = Path(path or os.environ.get("SATQUERY_ENV_FILE") or DEFAULT_ENV_FILE).expanduser()

    if target.exists():
        resolved = target.resolve()
        if resolved.is_relative_to(_REPO_ROOT):
            raise ValueError(
                f"{resolved} is inside the repository. Credentials are not read from "
                "files in this repo: the whole tree is uploaded to Modal by "
                "scripts/modal_phase0.py and travels with any zip or backup of the "
                f"folder. Move it to {DEFAULT_ENV_FILE} or point SATQUERY_ENV_FILE "
                "somewhere outside the tree."
            )
    if not target.exists():
        return []

    loaded = []
    for name, value in parse_env_file(target).items():
        if override or not os.environ.get(name):
            os.environ[name] = value
            loaded.append(name)
    return loaded


def credential_source() -> dict[str, str]:
    """Where each known credential came from. Never returns a value.

    Used by the fetch scripts to print something checkable without putting a
    secret on a terminal or in a log.

    Loads the file first. Reporting on ``os.environ`` alone made this print
    "missing" for names that were sitting in the file unread, which is worse
    than not reporting at all -- it sends you to debug a credential that works.
    """
    load_credentials()
    file_path = Path(
        os.environ.get("SATQUERY_ENV_FILE") or DEFAULT_ENV_FILE
    ).expanduser()
    from_file = set()
    if file_path.exists():
        try:
            from_file = set(parse_env_file(file_path))
        except ValueError:
            from_file = set()

    report = {}
    for name in CREDENTIAL_KEYS:
        if not os.environ.get(name):
            report[name] = "missing"
        elif name in from_file:
            report[name] = f"file ({file_path})"
        else:
            report[name] = "shell"
    return report
