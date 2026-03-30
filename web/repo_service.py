"""
Clone a GitHub repo and run repomap + churn analysis.
All heavy lifting happens here so the FastAPI layer stays thin.
"""

import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# Make sure parent dir is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from repomap import RepoMap, parse_dir


# ---------------------------------------------------------------------------
# Git clone
# ---------------------------------------------------------------------------

def clone_repo(github_url: str, churn_months: int = 6) -> str:
    """Clone a GitHub repo with enough history for churn analysis.

    Tries --shallow-since first (lighter). Falls back to --depth 200.
    After cloning, unshallows the grafted commits so pydriller's
    diff-tree calls don't fail on truncated parent references.
    """
    since_date = (datetime.now() - timedelta(days=churn_months * 30)).strftime(
        "%Y-%m-%d"
    )
    tmp = tempfile.mkdtemp(prefix="grasprepo_")

    strategies = [
        ["git", "clone", "--shallow-since", since_date, github_url, tmp],
        ["git", "clone", "--depth", "200", github_url, tmp],
    ]

    last_err = ""
    for cmd in strategies:
        # Clean dir for retry
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            return tmp
        except subprocess.TimeoutExpired:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RuntimeError("Clone timed out — the repository may be too large.")
        except subprocess.CalledProcessError as e:
            last_err = e.stderr
            continue

    shutil.rmtree(tmp, ignore_errors=True)
    raise RuntimeError(
        f"Failed to clone repository. Make sure it exists and is public.\n{last_err}"
    )


# ---------------------------------------------------------------------------
# Repomap analysis  (mirrors web/app.py get_repomap_data but decoupled)
# ---------------------------------------------------------------------------

def _build_ast_tree(file_symbols: dict) -> list:
    """Build a nested directory tree with symbols for the AST view.

    Returns a list of tree nodes like:
    [{"name": "src", "type": "dir", "children": [
        {"name": "app.py", "type": "file", "symbols": [...], "children": []},
    ]}]
    """
    root = {}

    for rel_path, symbols in sorted(file_symbols.items()):
        parts = rel_path.replace("\\", "/").split("/")
        node = root
        # Build directory nodes
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]
        # Leaf = file with its symbols
        filename = parts[-1]
        node[filename] = {"__symbols__": symbols}

    def to_list(tree, depth=0) -> list:
        result = []
        dirs = []
        files = []
        for name, val in sorted(tree.items()):
            if name == "__symbols__":
                continue
            if isinstance(val, dict) and "__symbols__" in val:
                files.append({"name": name, "type": "file", "symbols": val["__symbols__"]})
            else:
                children = to_list(val, depth + 1)
                # Count total symbols under this dir
                sym_count = sum(
                    len(c.get("symbols", []))
                    for c in _flatten(children)
                )
                dirs.append({"name": name, "type": "dir", "children": children, "symbol_count": sym_count})
        return dirs + files

    return to_list(root)


def _flatten(nodes: list) -> list:
    """Flatten nested tree into a list of file nodes."""
    out = []
    for n in nodes:
        if n["type"] == "file":
            out.append(n)
        elif "children" in n:
            out.extend(_flatten(n["children"]))
    return out


def run_repomap(repo_path: str):
    """Return structured repomap data for the given local repo."""
    import networkx as nx

    unpacked = parse_dir(repo_path)
    if not unpacked:
        return {"total_files": 0, "analyzed_files": 0, "ranked_files": [], "graph": {"nodes": [], "edges": []}}

    rm = RepoMap()
    repo_abs = os.path.abspath(repo_path)

    defines = defaultdict(set)
    references = defaultdict(list)
    definitions = defaultdict(set)

    # Collect per-file symbol definitions for the AST tree view
    file_symbols = defaultdict(list)

    for fname in sorted(unpacked):
        # Use path relative to the repo root, not cwd
        rel_fname = os.path.relpath(fname, repo_abs)
        tags = list(rm.get_tags(fname, rel_fname))
        if tags is None:
            continue
        for tag in tags:
            if tag.kind == "def":
                defines[tag.name].add(rel_fname)
                definitions[(rel_fname, tag.name)].add(tag)
                file_symbols[rel_fname].append({
                    "name": tag.name,
                    "line": tag.line,
                })
            elif tag.kind == "ref":
                references[tag.name].append(rel_fname)

    if not references:
        references = {k: list(v) for k, v in defines.items()}

    idents = set(defines.keys()).intersection(set(references.keys()))

    G = nx.MultiDiGraph()
    for ident in idents:
        definers = defines[ident]
        mul = 0.1 if ident.startswith("_") else 1
        for referencer, num_refs in Counter(references[ident]).items():
            for definer in definers:
                G.add_edge(
                    referencer, definer,
                    weight=mul * math.sqrt(num_refs), ident=ident,
                )

    try:
        ranked = nx.pagerank(G, weight="weight")
    except ZeroDivisionError:
        ranked = {}

    top_rank = sorted(
        [(rank, node) for node, rank in ranked.items()], reverse=True
    )
    centrality = nx.degree_centrality(G) if G.nodes() else {}

    ranked_files = []
    for i, (rank, node) in enumerate(top_rank):
        ranked_files.append({
            "rank_position": i + 1,
            "file": node,
            "pagerank_score": round(rank, 6),
            "centrality": round(centrality.get(node, 0), 6),
        })

    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            "source": u, "target": v,
            "weight": round(data["weight"], 4),
            "ident": data["ident"],
        })

    nodes = []
    for node in G.nodes():
        nodes.append({
            "id": node,
            "pagerank": round(ranked.get(node, 0), 6),
            "centrality": round(centrality.get(node, 0), 6),
        })

    # Build hierarchical AST tree: dir → file → symbols
    ast_tree = _build_ast_tree(file_symbols)

    return {
        "total_files": len(unpacked),
        "analyzed_files": len(G.nodes()),
        "ranked_files": ranked_files,
        "graph": {"nodes": nodes, "edges": edges},
        "ast_tree": ast_tree,
    }


# ---------------------------------------------------------------------------
# Churn analysis
# ---------------------------------------------------------------------------

def run_churn(repo_path: str, churn_months: int = 6):
    """Compute churn metrics using git log --numstat.

    This approach works reliably on shallow clones (unlike pydriller's
    diff-tree which fails on grafted boundary commits).
    """
    totime = datetime.now()
    month = totime.month - churn_months
    year = totime.year
    if month <= 0:
        month += 12
        year -= 1
    fromtime = totime.replace(year=year, month=month)

    since_str = fromtime.strftime("%Y-%m-%d")
    until_str = totime.strftime("%Y-%m-%d")

    # git log with --numstat gives: added\tremoved\tfilename per commit
    # Using %H%n%ae as separator lets us also count contributors
    try:
        result = subprocess.run(
            [
                "git", "-C", repo_path, "log",
                f"--since={since_str}", f"--until={until_str}",
                "--pretty=format:COMMIT:%H:%ae", "--numstat",
            ],
            capture_output=True, text=True, timeout=120,
        )
        raw = result.stdout
    except Exception as exc:
        return {
            "period": f"{since_str} to {until_str}",
            "total_files_with_commits": 0,
            "top_files": [],
            "warning": f"Churn analysis failed: {exc}",
        }

    if not raw.strip():
        return {
            "period": f"{since_str} to {until_str}",
            "total_files_with_commits": 0,
            "top_files": [],
        }

    # Parse git log output
    from collections import defaultdict
    file_commits = defaultdict(int)
    file_added = defaultdict(int)
    file_removed = defaultdict(int)
    file_max_added = defaultdict(int)
    file_max_removed = defaultdict(int)
    file_contributors = defaultdict(set)

    # Track per-commit stats for max/avg calculations
    file_commit_added = defaultdict(list)
    file_commit_removed = defaultdict(list)

    current_author = ""
    commit_file_added = defaultdict(int)
    commit_file_removed = defaultdict(int)

    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("COMMIT:"):
            # Flush previous commit's per-file stats
            for fname, added in commit_file_added.items():
                file_commit_added[fname].append(added)
            for fname, removed in commit_file_removed.items():
                file_commit_removed[fname].append(removed)
            commit_file_added.clear()
            commit_file_removed.clear()

            parts = line.split(":", 3)
            current_author = parts[2] if len(parts) > 2 else ""
            continue

        parts = line.split("\t")
        if len(parts) != 3:
            continue

        added_str, removed_str, fname = parts
        # Binary files show "-" for added/removed
        if added_str == "-" or removed_str == "-":
            continue

        try:
            added = int(added_str)
            removed = int(removed_str)
        except ValueError:
            continue

        file_commits[fname] += 1
        file_added[fname] += added
        file_removed[fname] += removed
        file_max_added[fname] = max(file_max_added[fname], added)
        file_max_removed[fname] = max(file_max_removed[fname], removed)
        commit_file_added[fname] += added
        commit_file_removed[fname] += removed
        if current_author:
            file_contributors[fname].add(current_author)

    # Flush last commit
    for fname, added in commit_file_added.items():
        file_commit_added[fname].append(added)
    for fname, removed in commit_file_removed.items():
        file_commit_removed[fname].append(removed)

    sortedby_commit = sorted(file_commits.items(), key=lambda x: x[1], reverse=True)

    results = []
    for fname, commits in sortedby_commit[:30]:
        total_churn = file_added[fname] + file_removed[fname]
        avg_added = round(file_added[fname] / commits, 2) if commits else 0
        avg_removed = round(file_removed[fname] / commits, 2) if commits else 0
        results.append({
            "filename": fname,
            "commits": commits,
            "total_contributor": len(file_contributors[fname]),
            "minor_contributor": sum(1 for a in file_contributors[fname]
                                     if sum(1 for c_f, _ in sortedby_commit if a in file_contributors[c_f]) <= 2),
            "contributor_exp": 0,
            "total_churn": total_churn,
            "max_churn": file_max_added[fname] + file_max_removed[fname],
            "avg_churn": round(total_churn / commits, 2) if commits else 0,
            "lines_added": file_added[fname],
            "max_lines_added": file_max_added[fname],
            "avg_lines_added": avg_added,
            "removed_lines": file_removed[fname],
            "max_lines_removed": file_max_removed[fname],
            "avg_lines_removed": avg_removed,
        })

    return {
        "period": f"{since_str} to {until_str}",
        "total_files_with_commits": len(sortedby_commit),
        "top_files": results,
    }


# ---------------------------------------------------------------------------
# Hotspot detection  (high rank + high churn = risk)
# ---------------------------------------------------------------------------

def compute_hotspots(repomap_data: dict, churn_data: dict) -> list[dict]:
    """Cross-reference ranked files and churned files to find hotspots."""
    ranked = {f["file"]: f for f in repomap_data.get("ranked_files", [])}
    churned = {f["filename"]: f for f in churn_data.get("top_files", [])}

    if not ranked or not churned:
        return []

    max_pr = max((f["pagerank_score"] for f in ranked.values()), default=1) or 1
    max_commits = max((f["commits"] for f in churned.values()), default=1) or 1

    hotspots = []
    for fname, rdata in ranked.items():
        if not fname:
            continue
        # Match by suffix since repomap uses relative paths
        cdata = None
        for cname, cd in churned.items():
            if not cname:
                continue
            if fname.endswith(cname) or cname.endswith(fname) or fname == cname:
                cdata = cd
                break
        if not cdata:
            continue

        norm_pr = rdata["pagerank_score"] / max_pr
        norm_commits = cdata["commits"] / max_commits
        score = round(norm_pr * norm_commits, 4)

        hotspots.append({
            "file": fname,
            "pagerank_score": rdata["pagerank_score"],
            "centrality": rdata["centrality"],
            "commits": cdata["commits"],
            "total_churn": cdata["total_churn"],
            "hotspot_score": score,
        })

    hotspots.sort(key=lambda h: h["hotspot_score"], reverse=True)
    return hotspots


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def analyze_repo(github_url: str, churn_months: int = 6) -> dict:
    """Clone, analyze, cleanup. Returns combined results dict."""
    repo_name = github_url.rstrip("/").split("/")[-1]
    tmp_dir = clone_repo(github_url, churn_months)

    try:
        repomap_data = run_repomap(tmp_dir)
        churn_data = run_churn(tmp_dir, churn_months)
        hotspots = compute_hotspots(repomap_data, churn_data)

        return {
            "repo_name": repo_name,
            "repo_url": github_url,
            "file_count": repomap_data["total_files"],
            "repomap": repomap_data,
            "churn": churn_data,
            "hotspots": hotspots,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
