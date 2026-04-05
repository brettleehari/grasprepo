"""
grasprepo web dashboard.
Single GitHub URL input → auto clone, analyze, display combined results.
Internal analytics tracked in SQLite.
"""

import asyncio
import csv
import hashlib
import io
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure the project root is on sys.path so repomap.py can find utils.py
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from web.analytics import (
    init_db,
    record_analysis,
    record_lead,
    get_total_count,
    get_recent,
    get_popular_repos,
    get_unique_repos_count,
    get_leads,
    get_leads_count,
    get_all_analyses,
    get_all_leads,
)
from web.models import AnalyzeRequest, DownloadRequest
from web.repo_service import analyze_repo


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Result cache (saves compute for example repos and repeat analyses)
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent / "data" / "cache"

def _cache_key(repo_url: str) -> str:
    return hashlib.sha256(repo_url.strip().rstrip("/").lower().encode()).hexdigest()[:16]

def cache_get(repo_url: str) -> dict | None:
    path = CACHE_DIR / f"{_cache_key(repo_url)}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None

def cache_put(repo_url: str, data: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(repo_url)}.json"
    try:
        path.write_text(json.dumps(data))
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    os.makedirs(CACHE_DIR, exist_ok=True)
    yield


app = FastAPI(title="grasprepo", version="0.2.0", lifespan=lifespan)

templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=str(templates_dir))
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/internal/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    return templates.TemplateResponse(request, "admin.html")


# ---------------------------------------------------------------------------
# API: Analyze
# ---------------------------------------------------------------------------

@app.post("/api/analyze")
async def api_analyze(body: AnalyzeRequest, request: Request):
    start = time.time()

    # Check cache first
    cached = cache_get(body.repo_url)
    if cached:
        cached["duration_seconds"] = 0
        cached["cached"] = True
        # Still record the analytics hit
        try:
            repomap = cached.get("repomap", {})
            ranked = repomap.get("ranked_files", [])
            churned = cached.get("churn", {}).get("top_files", [])
            hotspots = cached.get("hotspots", [])
            record_analysis(
                repo_url=body.repo_url,
                repo_name=cached.get("repo_name", ""),
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                file_count=cached.get("file_count", 0),
                ranked_files_count=len(ranked),
                churn_files_count=len(churned),
                hotspot_count=len(hotspots),
                top_ranked_file=ranked[0]["file"] if ranked else None,
                top_churned_file=churned[0]["filename"] if churned else None,
                top_hotspot_file=hotspots[0]["file"] if hotspots else None,
                duration_seconds=0,
            )
        except Exception:
            pass
        return JSONResponse(cached)

    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, analyze_repo, body.repo_url)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    duration = round(time.time() - start, 2)

    # Record to analytics (fire and forget)
    try:
        repomap = data.get("repomap", {})
        ranked = repomap.get("ranked_files", [])
        churned = data.get("churn", {}).get("top_files", [])
        hotspots = data.get("hotspots", [])
        graph = repomap.get("graph", {})

        # Count total AST symbols from the tree
        def count_symbols(tree):
            total = 0
            for n in (tree or []):
                if n.get("type") == "file":
                    total += len(n.get("symbols", []))
                elif n.get("children"):
                    total += count_symbols(n["children"])
            return total

        # Count unique contributors across all churn files
        all_contributors = set()
        for f in churned:
            all_contributors.update(range(f.get("total_contributor", 0)))

        record_analysis(
            repo_url=body.repo_url,
            repo_name=data.get("repo_name", ""),
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            file_count=data.get("file_count", 0),
            ranked_files_count=len(ranked),
            churn_files_count=len(churned),
            hotspot_count=len(hotspots),
            top_ranked_file=ranked[0]["file"] if ranked else None,
            top_churned_file=churned[0]["filename"] if churned else None,
            top_hotspot_file=hotspots[0]["file"] if hotspots else None,
            graph_nodes=len(graph.get("nodes", [])),
            graph_edges=len(graph.get("edges", [])),
            ast_symbol_count=count_symbols(repomap.get("ast_tree")),
            churn_total_contributors=max((f.get("total_contributor", 0) for f in churned), default=0),
            duration_seconds=duration,
        )
    except Exception:
        pass  # analytics should never break user flow

    data["duration_seconds"] = duration

    # Cache the result for future requests
    cache_put(body.repo_url, data)

    return JSONResponse(data)


# ---------------------------------------------------------------------------
# API: Download report (gated behind email / LinkedIn)
# ---------------------------------------------------------------------------

@app.post("/api/download")
async def api_download(body: DownloadRequest, request: Request):
    """Generate a CSV report. Requires email or LinkedIn to capture the lead."""
    # Record the lead
    try:
        record_lead(
            email=body.email,
            linkedin=body.linkedin,
            repo_url=body.repo_url,
            repo_name=body.repo_name,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
    except Exception:
        pass

    # Build CSV from the analysis data sent by the client
    output = io.StringIO()
    writer = csv.writer(output)

    # Sheet 1: Ranked files
    writer.writerow(["=== RANKED FILES ==="])
    writer.writerow(["Rank", "File", "PageRank Score", "Centrality"])
    for f in body.ranked_files:
        writer.writerow([f["rank_position"], f["file"], f["pagerank_score"], f["centrality"]])

    writer.writerow([])

    # Sheet 2: Churn metrics
    writer.writerow(["=== CHURN METRICS ==="])
    writer.writerow(["#", "File", "Commits", "Total Churn", "Lines Added", "Lines Removed", "Contributors"])
    for i, f in enumerate(body.churn_files, 1):
        writer.writerow([
            i, f["filename"], f["commits"], f["total_churn"],
            f["lines_added"], f["removed_lines"], f["total_contributor"],
        ])

    writer.writerow([])

    # Sheet 3: Hotspots
    writer.writerow(["=== HOTSPOTS ==="])
    writer.writerow(["File", "Hotspot Score", "PageRank", "Commits", "Total Churn", "Centrality"])
    for h in body.hotspots:
        writer.writerow([
            h["file"], h["hotspot_score"], h["pagerank_score"],
            h["commits"], h["total_churn"], h["centrality"],
        ])

    csv_content = output.getvalue()
    filename = f"grasprepo-{body.repo_name}-report.csv"

    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# API: Internal analytics
# ---------------------------------------------------------------------------

@app.get("/api/internal/analytics")
async def api_analytics():
    return JSONResponse({
        "total_analyses": get_total_count(),
        "unique_repos": get_unique_repos_count(),
        "total_leads": get_leads_count(),
        "recent": get_recent(50),
        "popular": get_popular_repos(10),
        "leads": get_leads(50),
    })


@app.get("/api/internal/export/analyses")
async def export_analyses():
    """Export all analyses as CSV — internal only."""
    columns, rows = get_all_analyses()
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="grasprepo-analyses.csv"'},
    )


@app.get("/api/internal/export/leads")
async def export_leads():
    """Export all leads as CSV — internal only."""
    columns, rows = get_all_leads()
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="grasprepo-leads.csv"'},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
