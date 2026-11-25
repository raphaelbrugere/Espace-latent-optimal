"""Noms, dossiers de resultats, publication sur le Hub Hugging Face et telechargement
(docs/HUB.md).

Un dossier par backbone, qui est aussi son depot (Espace-latent-optimal/<run>) :

    results/<donnees>-<archi>-<methode>-<AAAAMMJJ_HHMMSS>/     <run>
    |-- README.md                         fiche du depot, ecrite par le premier `push`
    |-- backbone/                         lo pretrain : backbone.pt, README.md, mlflow.db...
    `-- downstream/<tache>-<neck>-<tete>-<AAAAMMJJ_HHMMSS>/
        |                                 lo train : neck.pt (sauf neck repris), head.pt,
        |                                 README.md, results.json, mlflow.db...
        `-- eval/<split>-<AAAAMMJJ_HHMMSS>/   lo eval : results.json, mlflow.db...

Un run = un dossier, ecrit par lui seul, puis plus jamais modifie : `push` n'envoie que
les fichiers absents du depot, et plusieurs personnes peuvent ajouter des entrainements et
des evaluations au meme backbone sans jamais ecraser ceux des autres.

Chaque partie de nom est faite de lettres, chiffres, " _ " et " . ", jamais de " - "
(normalize) : " - " ne separe que les parties. Le depot est la copie du dossier : `fetch`
le recree a l'identique, au meme endroit. `locate` accepte indifferemment un chemin ou un nom.
"""

import os
import re
from collections.abc import Callable, Collection
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

from latentopt.data.paths import PROJECT_ROOT

ORG = "Espace-latent-optimal"
STAMP = "%Y%m%d_%H%M%S"
MAX_REPO_NAME = 96  # limite du Hub pour le nom (l'organisation a la sienne)
NONE = "none"  # partie inconnue : backbone externe ou aleatoire sans backbone_data/method

BACKBONE, DOWNSTREAM, EVAL = "backbone", "downstream", "eval"
CARD = "README.md"
# fichiers qui font d'un dossier une partie terminee (results.json : ecrit en dernier)
BACKBONE_FILES = ("backbone.pt", "config.yaml", "results.json")
DOWNSTREAM_FILES = ("head.pt", "config.yaml", "results.json")
EVAL_FILES = ("results.json",)
# jamais publies : reprise sur le DCE (checkpoints), artefacts MLflow, Slurm, cache du Hub.
# mlflow.db est publie : `lo mlflow sync` le relit sur le Hub pour la base d'equipe.
EXCLUDED_DIRS = {"checkpoints", "mlartifacts", ".submitit", ".cache"}

_PART = re.compile(r"[A-Za-z0-9_.]+")
_DATE = re.compile(r"\d{8}_\d{6}(_\d+)?")  # _<n> : job n > 0 d'un balayage


class Name(NamedTuple):
    """Parties du nom d'un backbone, donc de son dossier et de son depot."""

    data: str
    arch: str
    method: str
    date: str


class DownstreamName(NamedTuple):
    """Parties du nom d'un dossier downstream/<tache>-<neck>-<tete>-<date>."""

    task: str
    neck: str
    head: str
    date: str


class EvalName(NamedTuple):
    """Parties du nom d'un dossier eval/<split>-<date>."""

    split: str
    date: str


# --- noms -------------------------------------------------------------------------------


def normalize(part: str) -> str:
    """Une partie de nom : " - " et espaces deviennent " _ " ; autres caracteres refuses."""
    out = re.sub(r"[-\s]+", "_", str(part).strip())
    if not _PART.fullmatch(out) or ".." in out:
        raise ValueError(f"partie de nom invalide : {part!r} (lettres, chiffres, _ et .)")
    return out


@cache
def _process_stamp() -> str:
    """Date du lancement, figee pour le processus : chaque resolution du dossier de run
    renvoie le meme nom."""
    return datetime.now().strftime(STAMP)


def stamp(num: int | str | None = None) -> str:
    """Date du processus, suffixee par le numero de job d'un balayage s'il est > 0."""
    num = int(num) if num not in (None, "") else 0
    return _process_stamp() + (f"_{num}" if num else "")


def _given(value: Any) -> bool:
    return value not in (None, "", "None", "null")


def _split(name: str, n: int, form: str) -> list[str]:
    parts = name.split("-")
    if len(parts) != n:
        raise ValueError(f"{name!r} : attendu {form}")
    for p in parts:
        normalize(p)  # leve si la partie contient un caractere interdit
    if not _DATE.fullmatch(parts[-1]):
        raise ValueError(f"{name!r} : date {parts[-1]!r} au format AAAAMMJJ_HHMMSS attendue")
    return parts


def run_name(data: str, arch: str, method: str, num: int | str | None = None) -> str:
    """Nom d'un backbone : <donnees>-<archi>-<methode>-<date>."""
    parts = [p if _given(p) else NONE for p in (data, arch, method)]
    return "-".join(normalize(p) for p in parts) + f"-{stamp(num)}"


def parse(name: str) -> Name:
    """Decoupe un nom de backbone ; ValueError si le nombre de parties ou la date est invalide."""
    return Name(*_split(name, 4, "<donnees>-<archi>-<methode>-<date>"))


def parse_or_none(name: str) -> Name | None:
    """`parse`, mais None au lieu de ValueError (test " est-ce un nom de backbone ? ")."""
    try:
        return parse(name)
    except ValueError:
        return None


def downstream_name(task: str, neck: str, head: str, date: str) -> str:
    """Dossier d'un entrainement sous downstream/ : <tache>-<neck>-<tete>-<date>."""
    return f"{normalize(task)}-{normalize(neck)}-{normalize(head)}-{date}"


def parse_downstream(name: str) -> DownstreamName:
    return DownstreamName(*_split(name, 4, "<tache>-<neck>-<tete>-<date>"))


def eval_name(split: str, date: str) -> str:
    """Dossier d'une evaluation sous downstream/<...>/eval/ : <split>-<date>."""
    return f"{normalize(split)}-{date}"


def parse_eval(name: str) -> EvalName:
    """Inverse de eval_name ; utilisee par les tests et les outils, jamais pendant un run."""
    return EvalName(*_split(name, 2, "<split>-<date>"))


def repo_id(name: str, org: str = ORG) -> str:
    """`<org>/<nom>` ; ValueError si le nom est invalide ou depasse MAX_REPO_NAME."""
    parse(name)
    if len(name) > MAX_REPO_NAME:
        raise ValueError(f"{name} : {len(name)} caracteres, le Hub en accepte {MAX_REPO_NAME}")
    return f"{org}/{name}"


# --- dossiers ---------------------------------------------------------------------------


def run_root(path: str | Path) -> Path:
    """Dossier du backbone qui contient `path` (lui-meme, ou un parent : backbone/,
    downstream/<...>, downstream/<...>/eval/<...>) ; ValueError s'il n'est sous aucun dossier de
    backbone. Un dossier de downstream/ a aussi 4 parties : jamais pris pour un backbone."""
    path = Path(path)
    for p in (path, *path.parents):
        if p.parent.name != DOWNSTREAM and parse_or_none(p.name):
            return p
    raise ValueError(f"{path} n'est pas dans un dossier <donnees>-<archi>-<methode>-<date>")


def backbone_run(
    choice: str, run: str = "", forced: str = "", data: str = "", method: str = ""
) -> str:
    """Nom du backbone d'un run de `lo train` : `forced` (nom impose par l'utilisateur) s'il est
    donne, celui du run de pre-entrainement (backbone=from_run), sinon un nouveau nom date a
    partir de `backbone_data` et `backbone_method` (backbone externe ou aleatoire)."""
    if _given(forced):
        parse(forced)
        return forced
    if _given(run):
        return run_root(locate(run, pull=False)).name
    return run_name(data, choice, method)


def default_pair(task: str) -> tuple[str, str]:
    """Neck et tete de la baseline de la tache (import differe : evaluation.tasks est lourd)."""
    from latentopt.evaluation.tasks import get_task

    t = get_task(task)
    return t.default_neck, t.default_head


def train_dir(
    choice: str,
    run: str,
    forced: str,
    data: str,
    method: str,
    task: str,
    neck: str | None = None,
    head: str | None = None,
    neck_from: str | None = None,
    num: int | str | None = None,
) -> str:
    """Dossier d'un run de `lo train` sous results/ : <run>/downstream/<tache>-<neck>-<tete>-<date>
    (resolver de configs/train.yaml, qui passe les arguments dans cet ordre). <run> : voir
    `backbone_run` (choice, run, forced, data, method). Neck ou tete non choisis : ceux
    de la baseline de la tache ; avec `neck_from`, le neck est celui du dossier repris."""
    if _given(neck_from):
        neck = parse_downstream(Path(str(neck_from)).name).neck
    default_neck, default_head = default_pair(task)
    neck = neck if _given(neck) else default_neck
    head = head if _given(head) else default_head
    name = backbone_run(choice, run, forced, data, method)
    return f"{name}/{DOWNSTREAM}/{downstream_name(task, neck, head, stamp(num))}"


def results_root() -> Path:
    """results_root des configs env : OUTPUT_ROOT, sinon <projet>/results."""
    return Path(os.environ.get("OUTPUT_ROOT") or PROJECT_ROOT / "results")


def local_dir(name: str) -> Path:
    """Dossier local d'un backbone : results/<nom>."""
    parse(name)
    return results_root() / name


def locate(run: str | Path, pull: bool = True, org: str = ORG) -> Path:
    """Dossier d'un backbone, ou d'une de ses parties, a partir d'un chemin ou d'un nom.

    Chemin existant : tel quel (absolu, ou relatif a la racine du projet). Sinon, le premier
    element du chemin qui est un nom de backbone designe son dossier local, suivi du reste
    du chemin : `<run>`, `<run>/downstream/pets_seg-fpn-conv_seg-20261117_090000`, ou le
    chemin d'une autre machine (`/usr/users/.../results/<run>/...`). Absent en local : telecharge.
    """
    path = Path(run)
    path = path if path.is_absolute() else PROJECT_ROOT / path
    if path.exists():
        return path
    parts = Path(run).parts
    for i, part in enumerate(parts):
        if parse_or_none(part) is None:
            continue
        target = local_dir(part).joinpath(*parts[i + 1 :])
        if pull and not target.exists():
            fetch(part, org)
            if not target.exists():
                raise FileNotFoundError(f"{target} absent, y compris dans {repo_id(part, org)}")
        return target
    return path


def fetch(name: str, org: str = ORG, only: list[str] | None = None) -> Path:
    """Telecharge le depot `name` dans son dossier local (results/<nom>/), a l'identique du
    dossier publie ; `only` : motifs des fichiers a prendre (ex. ["downstream/pets_seg-*"]). Les
    fichiers deja presents et a jour sont conserves."""
    from huggingface_hub import snapshot_download

    dest = local_dir(name)
    snapshot_download(repo_id(name, org), local_dir=dest, allow_patterns=only)
    return dest


# --- parties d'un backbone --------------------------------------------------------------


def _done(folder: Path, required: tuple[str, ...]) -> bool:
    return all((folder / f).exists() for f in required)


def parts(run_dir: Path) -> list[Path]:
    """Parties terminees d'un backbone : backbone/, puis chaque downstream/<...> et ses
    evaluations eval/<...>, par nom. Un run en cours (sans son results.json) n'en fait pas
    encore partie."""
    found = [run_dir / BACKBONE] if _done(run_dir / BACKBONE, BACKBONE_FILES) else []
    if (run_dir / DOWNSTREAM).is_dir():
        for d in sorted((run_dir / DOWNSTREAM).iterdir()):
            if _done(d, DOWNSTREAM_FILES):
                found.append(d)
                if (d / EVAL).is_dir():
                    found += [e for e in sorted((d / EVAL).iterdir()) if _done(e, EVAL_FILES)]
    return found


def _owner(rel: Path) -> tuple[str, ...]:
    """Partie a laquelle appartient un fichier (chemin relatif au dossier du backbone) :
    backbone, downstream/<...>, ou downstream/<...>/eval/<...> pour ce qui est dans eval/."""
    p = rel.parts
    if p[0] == DOWNSTREAM:
        return p[:4] if len(p) > 4 and p[2] == EVAL else p[:2]
    return p[:1]


def files(folder: Path, keep: Callable[[Path], bool] = lambda rel: True) -> list[str]:
    """Fichiers publies d'un dossier (chemins relatifs, separateurs /, tries comme des
    chaines : meme ordre sous Windows et Linux), hors exclusions."""
    out = []
    for p in folder.rglob("*"):
        rel = p.relative_to(folder)
        excluded = EXCLUDED_DIRS & set(rel.parts)
        if p.is_file() and not excluded and rel.as_posix() != CARD and keep(rel):
            out.append(rel.as_posix())
    return sorted(out)


# --- fiches (README.md) -----------------------------------------------------------------
# Une fiche par dossier qui contient des poids, ecrite une fois a la fin de son run, et une
# fiche de depot fixe : aucune ne liste ce que d'autres ajoutent ensuite, donc aucune n'est
# jamais reecrite. La vue d'ensemble des runs est la base MLflow d'equipe (`lo mlflow sync`).


def _commit(res: dict) -> str:
    commit = res.get("git_commit")
    dirty = " (non commite)" if res.get("git_dirty") else ""
    return (f"`{commit[:8]}`" if commit else "") + dirty


def _num(n: Any) -> str:
    return f"{n:,}" if isinstance(n, int) else ""


def _rows(rows: list[tuple[str, Any]]) -> list[str]:
    out = ["| | |", "|---|---|"]
    return out + [f"| {k} | {'' if v is None else v} |" for k, v in rows]


def repo_card(name: str) -> str:
    """Fiche du depot (README.md a la racine) : ne depend que du nom du backbone."""
    parsed = parse(name)
    tags = [t for t in ("latentopt", parsed.method, parsed.data) if t != NONE]
    out = ["---", "library_name: pytorch", "tags:", *[f"- {t}" for t in tags], "---", ""]
    out += [
        f"# {name}",
        "",
        "Backbone du projet Espace latent optimal : copie du dossier "
        f"`results/{name}/`. Chaque dossier qui contient des poids a sa propre fiche :",
        "",
        "- `backbone/` : le backbone pre-entraine (`lo pretrain`) ;",
        "- `downstream/<tache>-<neck>-<tete>-<date>/` : un neck et une tete entraines sur le "
        "backbone gele (`lo train`), et leurs evaluations dans `eval/<split>-<date>/` (`lo eval`).",
        "",
        "```bash",
        f"uv run lo train backbone=from_run backbone.run={name} task=<tache>",
        f"uv run lo train backbone=from_run backbone.run={name} task=<tache> "
        "neck_from=<tache>-<neck>-<tete>-<date>",
        f"uv run lo eval downstream='[{name}/{DOWNSTREAM}/<tache>-<neck>-<tete>-<date>,...]'",
        "```",
        "",
        f"Absent en local, le depot est telecharge dans `results/{name}/`. Les poids sont verifies "
        "contre leurs empreintes au chargement (`backbone_sha256`, `neck_sha256`).",
        "",
    ]
    return "\n".join(out)


def backbone_card(name: str, res: dict) -> str:
    """Fiche de backbone/ : ce qui le decrit dans results.json."""
    parsed = parse(name)
    rows = [
        ("Architecture", f"`{parsed.arch}`"),
        ("Methode", parsed.method),
        ("Donnees", parsed.data),
        ("Epochs / pas", f"{res.get('epochs')} / {res.get('steps')}"),
        ("Parametres", _num(res.get("backbone_params"))),
        ("Heures GPU", res.get("gpu_hours")),
        ("Empreinte", f"`{str(res.get('backbone_sha256'))[:12]}`"),
        ("Commit", _commit(res)),
    ]
    usage = f"uv run lo train backbone=from_run backbone.run={name} task=<tache>"
    return "\n".join([f"# {name} - backbone", "", *_rows(rows), "", "```bash", usage, "```", ""])


def downstream_card(name: str, folder: str, res: dict) -> str:
    """Fiche de downstream/<folder>/ : tache, score val, neck (appris ou repris), parametres."""
    parsed = parse_downstream(folder)
    metric = res.get("metric")
    val = res.get("val", {}).get(metric)
    params = res.get("params", {})
    neck = f"`{parsed.neck}`"
    if res.get("neck_frozen"):
        neck += f", repris gele de `{res.get('neck')}`"
    rows = [
        ("Tache", parsed.task),
        ("Neck", neck),
        ("Tete", f"`{parsed.head}`"),
        (f"Val {metric} (entrainement)", f"{val:.4f}" if isinstance(val, float) else ""),
        ("Parametres neck / tete", f"{_num(params.get('neck'))} / {_num(params.get('head'))}"),
        ("Graine, epochs", f"{res.get('seed')}, {res.get('epochs')}"),
        ("Commit", _commit(res)),
    ]
    out = [f"# {name} - {folder}", "", *_rows(rows), ""]
    out += [
        f"Evaluations : `{EVAL}/<split>-<date>/results.json`.",
        "",
        "```bash",
        f"uv run lo eval downstream='[{name}/{DOWNSTREAM}/{folder}]'",
        f"uv run lo train backbone=from_run backbone.run={name} task=<tache> neck_from={folder}",
        "```",
        "",
    ]
    return "\n".join(out)


def write_card(part: Path, res: dict) -> None:
    """README.md d'un dossier backbone/ ou downstream/<...>, a la fin de son run. Hors de
    l'arborescence (hydra.run.dir surcharge) : pas de fiche."""
    try:
        name = run_root(part).name
    except ValueError:
        return
    if part.name == BACKBONE:
        text = backbone_card(name, res)
    elif part.parent.name == DOWNSTREAM:
        text = downstream_card(name, part.name, res)
    else:
        return
    (part / CARD).write_text(text, encoding="utf-8")


# --- publication ------------------------------------------------------------------------


class Upload(NamedTuple):
    """Un depot Hub a publier : son identifiant, le dossier source et les fichiers a envoyer."""

    repo_id: str
    folder: Path  # dossier local du backbone
    files: list[str]  # chemins relatifs a `folder`, absents du depot (fiche du depot comprise)
    present: list[str]  # fichiers termines deja publies : jamais renvoyes


def plan(run: str | Path, published: Collection[str] = (), org: str = ORG) -> Upload:
    """Ce que publie `push` : les fichiers des parties terminees du dossier de backbone qui
    contient `run`, sauf ceux deja dans le depot (`published`). Un dossier publie n'est
    jamais modifie, donc jamais renvoye : un push ne peut pas ecraser le travail d'un autre."""
    run_dir = run_root(Path(run).resolve())
    done = parts(run_dir)
    if not done:
        raise FileNotFoundError(f"{run_dir} : ni backbone, ni entrainement, ni evaluation termines")
    keep = {p.relative_to(run_dir).parts for p in done}
    local = files(run_dir, lambda rel: _owner(rel) in keep) + [CARD]
    published = set(published)
    return Upload(
        repo_id(run_dir.name, org),
        run_dir,
        [f for f in local if f not in published],
        [f for f in local if f in published],
    )


def remote(repo: str, api: Any) -> tuple[str | None, set[str]]:
    """Commit courant et fichiers du depot ; (None, vide) s'il n'existe pas encore."""
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        info = api.model_info(repo)
    except RepositoryNotFoundError:
        return None, set()
    return info.sha, {s.rfilename for s in info.siblings or []}


def push(upload: Upload, parent: str | None = None, private: bool = True, api: Any = None) -> None:
    """Cree le depot s'il manque, ecrit sa fiche si elle n'y est pas encore, puis publie les
    nouveaux fichiers en un commit. `parent` : commit du depot vu au moment du `plan` ; si
    quelqu'un a publie entre-temps, le Hub refuse le commit (relancer le push), comme git."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    api.create_repo(upload.repo_id, private=private, exist_ok=True)
    if CARD in upload.files:
        (upload.folder / CARD).write_text(repo_card(upload.folder.name), encoding="utf-8")
    api.upload_folder(
        repo_id=upload.repo_id,
        folder_path=upload.folder,
        allow_patterns=upload.files,
        commit_message=f"publication de {upload.folder.name}",
        parent_commit=parent,
    )
