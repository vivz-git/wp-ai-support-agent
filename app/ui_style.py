"""Shared CSS for the internal staff-facing pages (``/staff``, ``/simulate``).

One small design system so both pages look and behave the same, including on
a phone: a single scrolling column, no fixed pixel widths that force
horizontal scrolling, and tap targets big enough for touch.
"""

BASE_STYLE = """
:root { --bg:#f6f7f9; --card:#fff; --text:#1c1f24; --muted:#5f6670; --border:#dde1e6;
        --urgent:#c62828; --urgent-bg:#fdecea; --accent:#0b6e4f; --accent-bg:#e8f4ef;
        --bubble-patient:#fff; --bubble-agent:#dcf8c6; --bubble-pending:#eee; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#15171a; --card:#1f2226; --text:#e8eaed; --muted:#a0a6ad; --border:#33373d;
          --urgent:#ff6b6b; --urgent-bg:#3a1f1f; --accent:#4cc38a; --accent-bg:#1d3329;
          --bubble-patient:#262a2e; --bubble-agent:#12452f; --bubble-pending:#2a2d31; }
}
* { box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
html { -webkit-text-size-adjust:100%; }
body { margin:0; background:var(--bg); color:var(--text); overflow-x:hidden;
       font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans Devanagari",sans-serif; }
main { max-width:860px; margin:0 auto; padding:16px; }
h1 { font-size:1.3rem; margin:8px 0 4px; line-height:1.3; }
h2 { font-size:1.05rem; margin:28px 0 10px; color:var(--muted); }
.sub { color:var(--muted); margin:0 0 16px; }
.notice { background:var(--accent-bg); border:1px solid var(--accent); border-radius:8px; padding:10px 12px; margin-bottom:16px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:14px; margin-bottom:14px; }
.card.urgent { border:2px solid var(--urgent); background:var(--urgent-bg); }
.meta { display:flex; flex-wrap:wrap; gap:6px 14px; color:var(--muted); font-size:.85rem; margin-bottom:8px; }
.badge { background:var(--urgent); color:#fff; font-weight:700; border-radius:4px; padding:2px 8px; letter-spacing:.03em; font-size:.75rem; }
.label { font-size:.8rem; font-weight:600; color:var(--muted); margin:10px 0 4px; }
.patient { white-space:pre-wrap; overflow-wrap:anywhere; }
textarea, input[type=text] { width:100%; font:inherit; color:var(--text); background:var(--card);
           border:1px solid var(--border); border-radius:6px; padding:10px; }
textarea { min-height:140px; }
.actions { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
button, .btn { font:inherit; font-size:1rem; border-radius:6px; padding:10px 14px; cursor:pointer;
         border:1px solid var(--border); background:var(--card); color:var(--text); min-height:40px;
         text-decoration:none; display:inline-block; }
button.approve, .btn.approve { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
button.reject, .btn.reject { color:var(--urgent); border-color:var(--urgent); }
.empty { color:var(--muted); }
.recent { font-size:.9rem; }
.recent .card { padding:10px 12px; }
a { color:var(--accent); }
nav.top-links { display:flex; gap:14px; margin-bottom:6px; font-size:.9rem; }

@media (max-width: 480px) {
  main { padding:12px; }
  h1 { font-size:1.15rem; }
  .actions { flex-direction:column; }
  .actions button, .actions .btn { width:100%; }
}
"""
