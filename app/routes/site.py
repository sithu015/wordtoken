"""Public documentation and operations wiki routes."""

from __future__ import annotations

from html import escape
from textwrap import dedent

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse


router = APIRouter(include_in_schema=False)


def _render_layout(*, title: str, summary: str, body: str, active_path: str) -> str:
    nav_items = [
        ("/", "Overview"),
        ("/wiki", "Wiki"),
        ("/docs", "OpenAPI"),
        ("/redoc", "ReDoc"),
        ("/health", "Health"),
    ]
    nav_html = "".join(
        (
            '<a class="nav-link{active}" href="{href}">{label}</a>'.format(
                active=" is-active" if href == active_path else "",
                href=href,
                label=escape(label),
            )
        )
        for href, label in nav_items
    )

    return dedent(
        f"""\
        <!DOCTYPE html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>{escape(title)}</title>
            <meta name="description" content="{escape(summary)}" />
            <style>
              :root {{
                color-scheme: light;
                --bg: #f3efe5;
                --surface: rgba(255, 252, 245, 0.84);
                --surface-strong: rgba(255, 248, 233, 0.92);
                --ink: #14281f;
                --muted: #4c6358;
                --line: rgba(20, 40, 31, 0.12);
                --accent: #0f6a52;
                --accent-strong: #c95a28;
                --code-bg: #1c241f;
                --code-ink: #f5f2e8;
                --shadow: 0 24px 60px rgba(20, 40, 31, 0.10);
              }}

              * {{
                box-sizing: border-box;
              }}

              body {{
                margin: 0;
                font-family: "Avenir Next", "Segoe UI", "Noto Sans Myanmar", sans-serif;
                color: var(--ink);
                background:
                  radial-gradient(circle at top left, rgba(201, 90, 40, 0.18), transparent 30%),
                  radial-gradient(circle at top right, rgba(15, 106, 82, 0.18), transparent 26%),
                  linear-gradient(180deg, #f7f3ea 0%, var(--bg) 100%);
              }}

              a {{
                color: inherit;
              }}

              .shell {{
                max-width: 1160px;
                margin: 0 auto;
                padding: 28px 20px 64px;
              }}

              .nav {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 16px;
                flex-wrap: wrap;
                margin-bottom: 24px;
              }}

              .brand {{
                font-family: "Iowan Old Style", "Palatino Linotype", serif;
                font-size: 1.25rem;
                font-weight: 700;
                letter-spacing: 0.02em;
                text-decoration: none;
              }}

              .nav-links {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
              }}

              .nav-link {{
                text-decoration: none;
                padding: 10px 14px;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.5);
                border: 1px solid var(--line);
                color: var(--muted);
              }}

              .nav-link.is-active {{
                color: white;
                background: var(--accent);
                border-color: var(--accent);
              }}

              .hero {{
                display: grid;
                grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.9fr);
                gap: 22px;
                align-items: stretch;
                margin-bottom: 24px;
              }}

              .panel {{
                background: var(--surface);
                backdrop-filter: blur(16px);
                border: 1px solid var(--line);
                border-radius: 28px;
                box-shadow: var(--shadow);
              }}

              .hero-copy {{
                padding: 30px;
              }}

              .kicker {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 16px;
                padding: 8px 12px;
                border-radius: 999px;
                color: var(--accent);
                background: rgba(15, 106, 82, 0.08);
                font-size: 0.95rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
              }}

              h1, h2, h3 {{
                margin: 0 0 12px;
                line-height: 1.05;
              }}

              h1 {{
                font-family: "Iowan Old Style", "Palatino Linotype", serif;
                font-size: clamp(2.3rem, 4vw, 4.6rem);
                letter-spacing: -0.03em;
              }}

              h2 {{
                font-family: "Iowan Old Style", "Palatino Linotype", serif;
                font-size: clamp(1.7rem, 2.4vw, 2.4rem);
              }}

              h3 {{
                font-size: 1.05rem;
              }}

              p {{
                margin: 0 0 14px;
                line-height: 1.72;
                color: var(--muted);
              }}

              .hero-actions {{
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                margin-top: 22px;
              }}

              .button {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
                min-height: 44px;
                padding: 0 16px;
                border-radius: 14px;
                border: 1px solid var(--ink);
                text-decoration: none;
                font-weight: 700;
              }}

              .button-primary {{
                background: var(--ink);
                color: white;
                border-color: var(--ink);
              }}

              .button-secondary {{
                background: transparent;
                color: var(--ink);
                border-color: var(--line);
              }}

              .status-panel {{
                padding: 24px;
                display: grid;
                gap: 14px;
                background: linear-gradient(180deg, rgba(15, 106, 82, 0.93), rgba(8, 71, 55, 0.98));
                color: white;
              }}

              .status-grid {{
                display: grid;
                gap: 12px;
              }}

              .status-card {{
                border-radius: 18px;
                padding: 14px 16px;
                background: rgba(255, 255, 255, 0.12);
                border: 1px solid rgba(255, 255, 255, 0.14);
              }}

              .status-label {{
                font-size: 0.86rem;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                opacity: 0.78;
              }}

              .status-value {{
                display: block;
                margin-top: 6px;
                font-size: 1.2rem;
                font-weight: 800;
              }}

              .section-grid {{
                display: grid;
                grid-template-columns: repeat(12, minmax(0, 1fr));
                gap: 20px;
              }}

              .section {{
                grid-column: span 12;
                padding: 26px;
              }}

              .section.split {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 18px;
              }}

              .card-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 16px;
              }}

              .card {{
                padding: 18px;
                border-radius: 22px;
                background: rgba(255, 255, 255, 0.68);
                border: 1px solid var(--line);
              }}

              .pill {{
                display: inline-block;
                padding: 6px 10px;
                border-radius: 999px;
                font-size: 0.82rem;
                font-weight: 700;
                background: rgba(201, 90, 40, 0.1);
                color: var(--accent-strong);
                margin-bottom: 10px;
              }}

              .code {{
                margin: 0;
                padding: 16px;
                overflow-x: auto;
                border-radius: 18px;
                background: var(--code-bg);
                color: var(--code-ink);
                font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
                font-size: 0.92rem;
                line-height: 1.6;
              }}

              .list {{
                margin: 0;
                padding-left: 18px;
                color: var(--muted);
                line-height: 1.7;
              }}

              .list li + li {{
                margin-top: 8px;
              }}

              .table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 8px;
              }}

              .table th,
              .table td {{
                text-align: left;
                padding: 12px 10px;
                border-bottom: 1px solid var(--line);
                vertical-align: top;
              }}

              .muted {{
                color: var(--muted);
              }}

              @media (max-width: 900px) {{
                .hero,
                .section.split {{
                  grid-template-columns: 1fr;
                }}

                .hero-copy,
                .status-panel,
                .section {{
                  padding: 22px;
                }}
              }}
            </style>
          </head>
          <body>
            <div class="shell">
              <header class="nav">
                <a class="brand" href="/">Wordtoken</a>
                <nav class="nav-links">
                  {nav_html}
                </nav>
              </header>
              {body}
            </div>
          </body>
        </html>
        """
    )


def _status_panel(request: Request) -> str:
    settings = request.app.state.settings
    model = request.app.state.model
    health_state = "Ready" if model.model_loaded else "Degraded"
    auth_state = "enabled" if settings.api_keys else "disabled"
    return dedent(
        f"""\
        <aside class="panel status-panel">
          <div>
            <div class="kicker" style="background: rgba(255,255,255,0.1); color: white;">Live Runtime</div>
            <h2 style="color: white;">{escape(health_state)} on production</h2>
            <p style="color: rgba(255,255,255,0.78);">
              This page reflects the application state reported by the running
              FastAPI process.
            </p>
          </div>
          <div class="status-grid">
            <div class="status-card">
              <span class="status-label">Backend</span>
              <span class="status-value">{escape(model.backend)}</span>
            </div>
            <div class="status-card">
              <span class="status-label">Model</span>
              <span class="status-value">{escape(settings.model_name)}</span>
            </div>
            <div class="status-card">
              <span class="status-label">Device</span>
              <span class="status-value">{escape(settings.device)}</span>
            </div>
            <div class="status-card">
              <span class="status-label">Fallback</span>
              <span class="status-value">{'enabled' if model.fallback_enabled else 'disabled'}</span>
            </div>
            <div class="status-card">
              <span class="status-label">API key auth</span>
              <span class="status-value">{escape(auth_state)}</span>
            </div>
          </div>
        </aside>
        """
    )


@router.get("/", response_class=HTMLResponse)
async def overview(request: Request) -> HTMLResponse:
    """Serve the public documentation homepage."""
    body = dedent(
        f"""\
        <section class="hero">
          <div class="panel hero-copy">
            <div class="kicker">Myanmar NLP API</div>
            <h1>Production docs for segmentation and POS tagging.</h1>
            <p>
              Wordtoken exposes a focused API for Myanmar word segmentation,
              POS tagging, and batch inference. The service is live at
              <strong>wordtoken.ygn.app</strong> and fronts a Hugging Face-backed
              XLM-RoBERTa + BiLSTM + CRF model behind Caddy HTTPS.
            </p>
            <p>
              Use this landing page for fast onboarding, and open the wiki for
              operational notes, deployment topology, and release guidance.
            </p>
            <div class="hero-actions">
              <a class="button button-primary" href="/docs">Interactive API docs</a>
              <a class="button button-secondary" href="/wiki">Operations wiki</a>
              <a class="button button-secondary" href="/health">Health JSON</a>
            </div>
          </div>
          {_status_panel(request)}
        </section>

        <section class="section-grid">
          <section class="panel section">
            <div class="pill">API Surface</div>
            <h2>What this service exposes</h2>
            <div class="card-grid">
              <article class="card">
                <div class="pill">POST /api/v1/segment</div>
                <h3>Segment Myanmar text</h3>
                <p>
                  Returns a list of token strings for a single input sentence.
                  Send <code>X-API-Key</code> when API key auth is enabled.
                </p>
                <pre class="code">curl -X POST https://wordtoken.ygn.app/api/v1/segment \\
  -H 'X-API-Key: YOUR_API_KEY' \\
  -H 'content-type: application/json' \\
  -d '{{"text":"ကျွန်တော်သည်ကျောင်းသွားသည်"}}'</pre>
              </article>
              <article class="card">
                <div class="pill">POST /api/v1/pos</div>
                <h3>Joint segmentation + POS</h3>
                <p>
                  Returns token/POS pairs in one call for downstream NLP apps.
                  API keys apply to this route as well.
                </p>
                <pre class="code">curl -X POST https://wordtoken.ygn.app/api/v1/pos \\
  -H 'X-API-Key: YOUR_API_KEY' \\
  -H 'content-type: application/json' \\
  -d '{{"text":"ကျွန်တော်သည်ကျောင်းသွားသည်"}}'</pre>
              </article>
              <article class="card">
                <div class="pill">POST /api/v1/batch</div>
                <h3>Batch processing</h3>
                <p>
                  Processes multiple strings per request with batch-size controls
                  and the same header-based auth.
                </p>
                <pre class="code">curl -X POST https://wordtoken.ygn.app/api/v1/batch \\
  -H 'X-API-Key: YOUR_API_KEY' \\
  -H 'content-type: application/json' \\
  -d '{{"texts":["မင်္ဂလာပါ","hello world"]}}'</pre>
              </article>
            </div>
          </section>

          <section class="panel section split">
            <div>
              <div class="pill">Quick Start</div>
              <h2>First requests</h2>
              <p>
                The API is HTTPS-first. Plain HTTP requests are redirected by
                Caddy, and the OpenAPI UIs stay available at <code>/docs</code>
                and <code>/redoc</code>.
              </p>
              <p>
                If <code>API_KEYS</code> is configured on the server, include
                <code>X-API-Key</code> on every <code>/api/v1/*</code> request.
              </p>
              <pre class="code">curl https://wordtoken.ygn.app/health

curl -X POST https://wordtoken.ygn.app/api/v1/segment \\
  -H 'X-API-Key: YOUR_API_KEY' \\
  -H 'content-type: application/json' \\
  -d '{{"text":"မြန်မာဘာသာသည်လှပသောဘာသာတစ်ခုဖြစ်သည်"}}'</pre>
            </div>
            <div>
              <div class="pill">Sample Output</div>
              <h2>Expected response shape</h2>
              <pre class="code">{{
  "input": "ကျွန်တော်သည်ကျောင်းသွားသည်",
  "words": [
    "ကျွန်တော်",
    "သည်",
    "ကျောင်း",
    "သွား",
    "သည်"
  ],
  "processing_time_ms": 105.12
}}</pre>
            </div>
          </section>

          <section class="panel section">
            <div class="pill">Links</div>
            <h2>Where to go next</h2>
            <table class="table">
              <thead>
                <tr>
                  <th>Resource</th>
                  <th>Purpose</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><a href="/wiki">/wiki</a></td>
                  <td>Operations handbook, deployment topology, and CI/CD notes.</td>
                </tr>
                <tr>
                  <td><a href="/docs">/docs</a></td>
                  <td>Swagger UI backed by the live OpenAPI schema.</td>
                </tr>
                <tr>
                  <td><a href="/redoc">/redoc</a></td>
                  <td>Alternative API reference with a long-form layout.</td>
                </tr>
                <tr>
                  <td><a href="https://github.com/sithu015/wordtoken">GitHub repository</a></td>
                  <td>Source code, deploy workflow, and infrastructure templates.</td>
                </tr>
              </tbody>
            </table>
          </section>
        </section>
        """
    )
    return HTMLResponse(
        _render_layout(
            title="Wordtoken Overview",
            summary="Public documentation for the Myanmar word segmentation and POS tagging API.",
            body=body,
            active_path="/",
        )
    )


@router.get("/wiki", response_class=HTMLResponse)
async def wiki(request: Request) -> HTMLResponse:
    """Serve the public operations wiki."""
    body = dedent(
        f"""\
        <section class="hero">
          <div class="panel hero-copy">
            <div class="kicker">Operations Wiki</div>
            <h1>Runbook, release flow, and infrastructure notes.</h1>
            <p>
              This page explains how the production instance is laid out, what
              the GitHub Actions deploy job does, and which commands are useful
              when something looks wrong.
            </p>
            <div class="hero-actions">
              <a class="button button-primary" href="/health">Check live health</a>
              <a class="button button-secondary" href="https://github.com/sithu015/wordtoken">Open repository</a>
            </div>
          </div>
          {_status_panel(request)}
        </section>

        <section class="section-grid">
          <section class="panel section split">
            <div>
              <div class="pill">Topology</div>
              <h2>Runtime chain</h2>
              <pre class="code">Client
  -> Caddy :443 / :80
  -> reverse_proxy 127.0.0.1:8000
  -> FastAPI container (wordtoken)
  -> Hugging Face cache volume
  -> XLM-RoBERTa + BiLSTM + CRF inference</pre>
              <p>
                The public site terminates TLS in Caddy. The API container is
                only bound on loopback, so traffic reaches it through the reverse
                proxy and not directly from the internet.
              </p>
            </div>
            <div>
              <div class="pill">Startup</div>
              <h2>Model lifecycle</h2>
              <ul class="list">
                <li>The first cold boot downloads roughly 1.1 GB of model artifacts from Hugging Face.</li>
                <li>Downloaded files are cached in the Docker volume <code>wordtoken_huggingface_cache</code>.</li>
                <li>The health endpoint stays degraded until the model is fully loaded.</li>
                <li>Subsequent restarts are much faster because the cache is reused.</li>
              </ul>
            </div>
          </section>

          <section class="panel section split">
            <div>
              <div class="pill">Release Flow</div>
              <h2>GitHub Actions deployment</h2>
              <ol class="list">
                <li>Push or merge a change into <code>main</code>.</li>
                <li>GitHub Actions checks out the repository and runs the test suite.</li>
                <li>The workflow syncs the repository to <code>/opt/wordtoken</code> over SSH.</li>
                <li>The remote deploy script rebuilds the Docker image and replaces the running container.</li>
                <li>The script waits for <code>http://127.0.0.1:8000/health</code> to return successfully before completing.</li>
              </ol>
            </div>
            <div>
              <div class="pill">Repo Secrets</div>
              <h2>Required GitHub configuration</h2>
              <table class="table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Use</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><code>DEPLOY_SSH_KEY</code></td>
                    <td>Private key used by the workflow to SSH into the production host.</td>
                  </tr>
                  <tr>
                    <td><code>DEPLOY_HOST</code></td>
                    <td>Production IPv4 or hostname.</td>
                  </tr>
                  <tr>
                    <td><code>DEPLOY_USER</code></td>
                    <td>SSH user on the target server.</td>
                  </tr>
                  <tr>
                    <td><code>DEPLOY_PORT</code></td>
                    <td>SSH port, typically <code>22</code>.</td>
                  </tr>
                  <tr>
                    <td><code>DEPLOY_PATH</code></td>
                    <td>Remote application directory, usually <code>/opt/wordtoken</code>.</td>
                  </tr>
                  <tr>
                    <td><code>SITE_URL</code></td>
                    <td>Public health-check target used after deploy, for example <code>https://wordtoken.ygn.app</code>.</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          <section class="panel section split">
            <div>
              <div class="pill">Troubleshooting</div>
              <h2>Useful commands</h2>
              <pre class="code">docker logs --tail=200 wordtoken
docker ps
curl -sS http://127.0.0.1:8000/health
systemctl status caddy
journalctl -u caddy -n 100 --no-pager</pre>
            </div>
            <div>
              <div class="pill">Recovery Notes</div>
              <h2>Common symptoms</h2>
              <ul class="list">
                <li><strong>502/connection reset:</strong> the container is still loading the model or crashed during startup.</li>
                <li><strong>Health says degraded:</strong> check container logs for Hugging Face download errors or missing runtime dependencies.</li>
                <li><strong>401 on API requests:</strong> verify the <code>X-API-Key</code> header matches one of the configured <code>API_KEYS</code>.</li>
                <li><strong>TLS problems:</strong> inspect <code>systemctl status caddy</code> and ensure DNS still points to the server.</li>
                <li><strong>Slow cold boots:</strong> keep the Hugging Face cache volume intact between releases.</li>
              </ul>
            </div>
          </section>
        </section>
        """
    )
    return HTMLResponse(
        _render_layout(
            title="Wordtoken Wiki",
            summary="Operations wiki for the Myanmar NLP API, including deployment and release guidance.",
            body=body,
            active_path="/wiki",
        )
    )
