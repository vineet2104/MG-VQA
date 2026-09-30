#!/usr/bin/env python3
"""Build a static HTML report for a rollout directory.

Produces:
  <rollout_dir>/index.html              — run summary + sortable/filterable episode table
  <rollout_dir>/ep_NNN_*/report.html    — per-episode page with video, step cards, etc.

Everything references existing files via relative paths (no copying).
Open index.html in any browser (file:// protocol is fine).

Usage:
  python scripts_new/build_report.py rollouts/rollout_vlm_20260417_030531
  python scripts_new/build_report.py rollouts/rollout_vlm_*                # multiple
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from jinja2 import Template

# --------------------------------------------------------------------------- #
#  Templates                                                                  #
# --------------------------------------------------------------------------- #

INDEX_TMPL = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{{ rollout_name }} — Rollout Report</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600; }
  .ok   { background: #d1fae5; color: #065f46; }
  .bad  { background: #fee2e2; color: #991b1b; }
  .mid  { background: #fef3c7; color: #92400e; }
  .nul  { background: #e5e7eb; color: #374151; }
  .chip { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 11px;
          background: #eef2ff; color: #3730a3; margin-right: 4px; }
</style>
</head>
<body class="bg-slate-50 text-slate-800">

<div class="max-w-7xl mx-auto p-6">

  <header class="mb-6">
    <h1 class="text-3xl font-bold text-slate-900">{{ rollout_name }}</h1>
    <div class="mt-2 text-sm text-slate-600">
      <span class="chip">model · {{ cfg.model_name }}</span>
      <span class="chip">mode · {{ cfg.mode }}</span>
      <span class="chip">seed · {{ cfg.base_seed }}</span>
      <span class="chip">pointing · {{ cfg.pointing_model }}</span>
      <span class="chip">renderer · {{ cfg.renderer }}</span>
      <span class="chip">max_actions · {{ cfg.max_actions }}</span>
      <span class="chip">{{ cfg.timestamp }}</span>
    </div>
  </header>

  <!-- Summary cards -->
  <div class="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
    <div class="bg-white rounded-lg shadow-sm p-4">
      <div class="text-xs text-slate-500">Episodes</div>
      <div class="text-2xl font-bold">{{ summary.n_total }}</div>
      <div class="text-xs text-slate-500 mt-1">answered {{ summary.n_answered }} · failed {{ summary.n_fail or 0 }} · err {{ summary.n_error or 0 }}</div>
    </div>
    <div class="bg-white rounded-lg shadow-sm p-4">
      <div class="text-xs text-slate-500">Accuracy</div>
      <div class="text-2xl font-bold">{{ "%.1f"|format(summary.accuracy or 0) }}%</div>
      <div class="text-xs text-slate-500 mt-1">{{ summary.n_correct }}/{{ summary.n_total }} correct
        <span class="text-slate-400">· of answered: {{ "%.1f"|format(summary.accuracy_answered or 0) }}%</span>
      </div>
    </div>
    <div class="bg-white rounded-lg shadow-sm p-4">
      <div class="text-xs text-slate-500">Avg actions / ep</div>
      <div class="text-2xl font-bold">{{ "%.1f"|format(summary.avg_actions or 0) }}</div>
      <div class="text-xs text-slate-500 mt-1">budget {{ cfg.max_actions }}</div>
    </div>
    {% if summary.manipulation %}
    <div class="bg-white rounded-lg shadow-sm p-4">
      <div class="text-xs text-slate-500">Grasp success</div>
      <div class="text-2xl font-bold">{{ "%.0f"|format(summary.manipulation.grasps.success_rate or 0) }}%</div>
      <div class="text-xs text-slate-500 mt-1">{{ summary.manipulation.grasps.successes }}/{{ summary.manipulation.grasps.attempts }}</div>
    </div>
    <div class="bg-white rounded-lg shadow-sm p-4">
      <div class="text-xs text-slate-500">Push success</div>
      <div class="text-2xl font-bold">{{ "%.0f"|format(summary.manipulation.pushes.success_rate or 0) }}%</div>
      <div class="text-xs text-slate-500 mt-1">{{ summary.manipulation.pushes.successes }}/{{ summary.manipulation.pushes.attempts }}</div>
    </div>
    {% endif %}
  </div>

  <!-- Per-type breakdown -->
  {% if summary.per_question_type %}
  <div class="bg-white rounded-lg shadow-sm p-4 mb-6">
    <h2 class="font-semibold text-sm text-slate-700 mb-3">Per-type accuracy</h2>
    <div class="grid grid-cols-2 md:grid-cols-6 gap-3">
      {% for qtype, stats in summary.per_question_type.items() %}
      <div>
        <div class="text-xs font-medium text-slate-600">{{ qtype }}</div>
        <div class="text-lg font-bold">{{ stats.correct }}/{{ stats.total }}</div>
        {% set acc = (stats.correct / stats.total * 100) if stats.total else 0 %}
        <div class="h-2 rounded-full bg-slate-200 mt-1 overflow-hidden">
          <div class="h-full {% if acc >= 70 %}bg-emerald-500{% elif acc >= 40 %}bg-amber-500{% else %}bg-rose-500{% endif %}"
               style="width: {{ acc }}%"></div>
        </div>
        <div class="text-xs text-slate-500 mt-1">{{ "%.0f"|format(acc) }}%</div>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}

  <!-- Filters -->
  <div class="bg-white rounded-lg shadow-sm p-4 mb-4 flex gap-3 flex-wrap items-center">
    <label class="text-sm text-slate-600">Type:</label>
    <select id="filter-type" class="border rounded px-2 py-1 text-sm">
      <option value="">all</option>
      {% for qt in qtypes %}<option>{{ qt }}</option>{% endfor %}
    </select>
    <label class="text-sm text-slate-600">Outcome:</label>
    <select id="filter-outcome" class="border rounded px-2 py-1 text-sm">
      <option value="">all</option>
      <option value="correct">correct</option>
      <option value="wrong">wrong</option>
      <option value="failed">failed / no answer</option>
    </select>
    <input id="filter-text" type="text" placeholder="search question text…"
           class="border rounded px-2 py-1 text-sm flex-1 min-w-[200px]" />
    <span id="row-count" class="text-xs text-slate-500 ml-auto"></span>
  </div>

  <!-- Episode table -->
  <div class="bg-white rounded-lg shadow-sm overflow-hidden">
    <table class="w-full text-sm" id="ep-table">
      <thead class="bg-slate-100 text-xs uppercase text-slate-600">
        <tr>
          <th class="px-3 py-2 text-left">#</th>
          <th class="px-3 py-2 text-left">Type</th>
          <th class="px-3 py-2 text-left">Question</th>
          <th class="px-3 py-2 text-left">GT</th>
          <th class="px-3 py-2 text-left">VLM</th>
          <th class="px-3 py-2 text-center">Result</th>
          <th class="px-3 py-2 text-center">Steps</th>
          <th class="px-3 py-2 text-center">G / P</th>
          <th class="px-3 py-2 text-left">Variant</th>
          <th class="px-3 py-2"></th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-100">
        {% for ep in episodes %}
        <tr class="hover:bg-slate-50 ep-row"
            data-type="{{ ep.question_type }}"
            data-outcome="{{ ep.outcome }}"
            data-text="{{ ep.question|lower }}">
          <td class="px-3 py-2 font-mono text-xs">{{ "%03d"|format(ep.episode) }}</td>
          <td class="px-3 py-2"><span class="chip">{{ ep.question_type }}</span></td>
          <td class="px-3 py-2 text-slate-700 truncate max-w-md" title="{{ ep.question }}">{{ ep.question_short }}</td>
          <td class="px-3 py-2 font-mono text-xs">{{ ep.gt_answer_short }}</td>
          <td class="px-3 py-2 font-mono text-xs">{{ ep.vlm_answer_short }}</td>
          <td class="px-3 py-2 text-center">
            {% if ep.outcome == 'correct' %}<span class="pill ok">✓</span>
            {% elif ep.outcome == 'partial' %}<span class="pill mid">◐</span>
            {% elif ep.outcome == 'wrong' %}<span class="pill bad">✗</span>
            {% else %}<span class="pill nul">{{ ep.status_short }}</span>{% endif %}
          </td>
          <td class="px-3 py-2 text-center text-xs">{{ ep.total_vlm_steps }}</td>
          <td class="px-3 py-2 text-center text-xs font-mono">
            {{ ep.grasp_successes }}/{{ ep.grasp_attempts }} · {{ ep.push_successes }}/{{ ep.push_attempts }}
          </td>
          <td class="px-3 py-2 text-xs text-slate-500">{{ ep.template_variant or '' }}</td>
          <td class="px-3 py-2 text-right">
            <a href="{{ ep.report_path }}" class="text-indigo-600 hover:underline font-medium">open →</a>
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <footer class="mt-6 text-xs text-slate-400">
    Generated by scripts_new/build_report.py · re-run to refresh after more episodes land.
  </footer>
</div>

<script>
function applyFilter() {
  const t = document.getElementById('filter-type').value;
  const o = document.getElementById('filter-outcome').value;
  const q = document.getElementById('filter-text').value.toLowerCase();
  const rows = document.querySelectorAll('.ep-row');
  let shown = 0;
  rows.forEach(r => {
    const matches =
      (!t || r.dataset.type === t) &&
      (!o || r.dataset.outcome === o) &&
      (!q || r.dataset.text.includes(q));
    r.style.display = matches ? '' : 'none';
    if (matches) shown++;
  });
  document.getElementById('row-count').textContent = `showing ${shown} / ${rows.length}`;
}
document.getElementById('filter-type').addEventListener('change', applyFilter);
document.getElementById('filter-outcome').addEventListener('change', applyFilter);
document.getElementById('filter-text').addEventListener('input', applyFilter);
applyFilter();
</script>

</body>
</html>
""")


EPISODE_TMPL = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ep_{{ "%03d"|format(ep.episode) }} · {{ ep.question_type }} · {{ rollout_name }}</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600; }
  .ok { background:#d1fae5; color:#065f46; }
  .bad { background:#fee2e2; color:#991b1b; }
  .mid { background:#fef3c7; color:#92400e; }
  .nul { background:#e5e7eb; color:#374151; }
  .chip { display:inline-block; padding:1px 6px; border-radius:4px; font-size:11px;
          background:#eef2ff; color:#3730a3; margin-right:4px; }
  pre.prompt { background:#fafafa; color:#1e293b; padding:12px; border-radius:6px;
               font-size:12px; overflow-x:auto; line-height:1.5; white-space:pre-wrap;
               border:1px solid #e2e8f0; max-height:500px; overflow-y:auto; }
  .step-tool { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
               background:#1e293b; color:#fef3c7; padding:3px 10px; border-radius:4px;
               font-size:13px; font-weight:600; display:inline-block; }
  .field-label { font-size:10px; font-weight:700; letter-spacing:0.05em;
                 color:#64748b; text-transform:uppercase; }
  .field-val { font-size:13px; color:#1e293b; line-height:1.5; white-space:pre-wrap; }
  .side-label { font-size:11px; font-weight:700; letter-spacing:0.03em;
                color:#475569; text-transform:uppercase; margin-bottom:6px; }
  details > summary { cursor:pointer; list-style:none; }
  details > summary::-webkit-details-marker { display:none; }
  details > summary::before { content:"▸ "; font-size:0.85em; color:#94a3b8;
                              transition: transform 0.15s; display:inline-block; }
  details[open] > summary::before { content:"▾ "; }
  details[open] > summary { margin-bottom: 8px; }
  img.preview { cursor:zoom-in; transition: transform 0.15s; }
  img.preview:hover { transform: scale(1.02); }
  .step-card { transition: box-shadow 0.15s; }
  .step-card[open] { box-shadow: 0 4px 12px -2px rgba(0,0,0,0.08); }
</style>
</head>
<body class="bg-slate-50 text-slate-800">

<div class="max-w-7xl mx-auto p-6">

  <!-- Top-bar: back + expand/collapse -->
  <div class="flex items-center justify-between mb-4">
    <a href="{{ back_to_index }}" class="text-indigo-600 hover:underline text-sm">← back to run</a>
    <div class="flex gap-2 text-xs">
      <button id="expand-all"   class="px-3 py-1 border border-slate-300 rounded hover:bg-slate-100">expand all steps</button>
      <button id="collapse-all" class="px-3 py-1 border border-slate-300 rounded hover:bg-slate-100">collapse all</button>
    </div>
  </div>

  <!-- Header: question + just a pass/fail badge on the right -->
  <header class="bg-white rounded-lg shadow-sm p-5 mb-6">
    <div class="flex items-start justify-between gap-6">
      <div class="flex-1">
        <div class="text-xs text-slate-500 mb-1">
          Episode {{ "%03d"|format(ep.episode) }} · seed {{ ep.episode_seed }}
          <span class="chip ml-2">{{ ep.question_type }}</span>
          {% if ep.question_subtype %}<span class="chip">{{ ep.question_subtype }}</span>{% endif %}
          {% if ep.template_variant %}<span class="chip">{{ ep.template_variant }}</span>{% endif %}
          {% if ep.template_category %}<span class="chip">{{ ep.template_category }}</span>{% endif %}
        </div>
        <h1 class="text-xl font-bold text-slate-900 leading-snug mt-1">{{ ep.question }}</h1>
        {% if ep.answer_options %}
        <div class="mt-2 text-sm text-slate-600">
          Options:
          {% for opt in ep.answer_options %}<span class="chip">{{ letter_or_num(loop.index0) }} · {{ opt }}</span>{% endfor %}
        </div>
        {% endif %}
      </div>
      <div class="shrink-0">
        {% if ep.outcome == 'correct' %}
          <div class="pill ok" style="font-size:14px; padding:6px 16px;">✓ SUCCESS</div>
        {% elif ep.outcome == 'partial' %}
          <div class="pill mid" style="font-size:14px; padding:6px 16px;">◐ PARTIAL</div>
        {% elif ep.outcome == 'wrong' %}
          <div class="pill bad" style="font-size:14px; padding:6px 16px;">✗ FAILURE</div>
        {% else %}
          <div class="pill nul" style="font-size:14px; padding:6px 16px;">{{ ep.status or 'unknown' }}</div>
        {% endif %}
      </div>
    </div>
  </header>

  <!-- Video + GT scene -->
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
    {% if ep.video %}
    <div class="bg-white rounded-lg shadow-sm p-3 lg:col-span-2">
      <div class="text-xs font-semibold text-slate-600 mb-2">Episode video (PyBullet — left: 3rd-person, right: top-down)</div>
      <video controls preload="metadata" class="w-full rounded">
        <source src="{{ ep.video }}" type="video/mp4">
      </video>
    </div>
    {% elif ep.manipulation_actions == 0 %}
    <div class="bg-white rounded-lg shadow-sm p-3 lg:col-span-2 flex items-center justify-center" style="min-height:200px;">
      <div class="text-center text-slate-400">
        <div class="text-3xl mb-2">🎬</div>
        <div class="text-sm font-medium">No manipulations executed</div>
        <div class="text-xs mt-1">VLM answered without grasping or pushing — no recorded video</div>
      </div>
    </div>
    {% endif %}
    {% if ep.scene_image %}
    <div class="bg-white rounded-lg shadow-sm p-3">
      <div class="text-xs font-semibold text-slate-600 mb-2">Scene (target marked)</div>
      <a href="{{ ep.scene_image }}" target="_blank">
        <img src="{{ ep.scene_image }}" class="preview w-full rounded border border-slate-200" loading="lazy">
      </a>
      {% if ep.scene_image_legend %}
      <div class="mt-2 text-xs text-slate-600 space-y-0.5">
        {% for item in ep.scene_image_legend %}
        <div><span style="color:{{ item.color }}; font-weight:700;">●</span> {{ item.label }}</div>
        {% endfor %}
      </div>
      {% endif %}
    </div>
    {% endif %}
  </div>

  <!-- Step cards (each is a <details> dropdown) -->
  <h2 class="text-lg font-semibold text-slate-800 mb-3">Reasoning Trace ({{ steps|length }} steps)</h2>

  {% for step in steps %}
  <details id="step-{{ step.n }}" class="step-card bg-white rounded-lg shadow-sm mb-3 overflow-hidden">
    <summary class="border-b border-slate-200 px-5 py-3 bg-slate-50 hover:bg-slate-100 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="text-sm font-bold text-slate-700">Step {{ step.n }}</div>
        <div class="step-tool">{{ step.tool_summary }}</div>
        {% if step.headline %}<div class="text-xs text-slate-500 italic">{{ step.headline }}</div>{% endif %}
      </div>
      {% if step.outcome %}<span class="pill {{ step.outcome_class }}">{{ step.outcome }}</span>{% endif %}
    </summary>

    <div class="p-5 grid grid-cols-1 lg:grid-cols-2 gap-5">

      <!-- LEFT column: input -->
      <div>
        <div class="side-label">▶ Input to the VLM at this step</div>
        {% if step.input_image %}
        <a href="{{ step.input_image.src }}" target="_blank">
          <img src="{{ step.input_image.src }}" class="preview w-full rounded border border-slate-200 mb-3" loading="lazy">
        </a>
        <div class="text-[11px] text-slate-500 mb-3">{{ step.input_image.label }}</div>
        {% endif %}
        {% if step.prompt %}
        <details class="bg-slate-50 border border-slate-200 rounded">
          <summary class="px-3 py-2 text-xs font-semibold text-slate-700">
            User prompt ({{ step.prompt|length }} chars)
          </summary>
          <pre class="prompt" style="border:0; border-radius:0;">{{ step.prompt }}</pre>
        </details>
        {% endif %}
      </div>

      <!-- RIGHT column: output -->
      <div>
        <div class="side-label">◀ Output from the VLM</div>

        {% if step.response %}
        <div class="space-y-3 bg-slate-50 border border-slate-200 rounded p-3 mb-3">
          {% if step.response.tool %}
          <div>
            <div class="field-label">Tool</div>
            <div class="field-val font-mono">{{ step.tool_summary }}</div>
          </div>
          {% endif %}
          {% if step.response.reasoning %}
          <div>
            <div class="field-label">Reasoning</div>
            <div class="field-val">{{ step.response.reasoning }}</div>
          </div>
          {% endif %}
          {% if step.response.plan %}
          <div>
            <div class="field-label">Plan</div>
            <div class="field-val">{{ step.response.plan }}</div>
          </div>
          {% endif %}
          {% if step.response.belief %}
          <div>
            <div class="field-label">Belief</div>
            <div class="field-val">{{ step.response.belief }}</div>
          </div>
          {% endif %}
          {% if step.extra_fields %}
          {% for k, v in step.extra_fields %}
          <div>
            <div class="field-label">{{ k }}</div>
            <div class="field-val font-mono">{{ v }}</div>
          </div>
          {% endfor %}
          {% endif %}
        </div>
        {% elif step.response_raw %}
        <pre class="prompt mb-3">{{ step.response_raw }}</pre>
        {% endif %}

        {% if step.output_images %}
        <div class="grid grid-cols-{{ step.output_images|length }} gap-2">
          {% for img in step.output_images %}
          <div>
            <div class="text-[11px] text-slate-500 mb-1 font-medium">{{ img.label }}</div>
            <a href="{{ img.src }}" target="_blank">
              <img src="{{ img.src }}" class="preview w-full rounded border border-slate-200" loading="lazy">
            </a>
          </div>
          {% endfor %}
        </div>
        {% endif %}
      </div>
    </div>
  </details>
  {% endfor %}

  {% if not steps %}
  <div class="bg-white rounded-lg shadow-sm p-8 text-center text-slate-500">
    No VLM steps recorded for this episode.
  </div>
  {% endif %}

  <!-- Final episode summary -->
  <div class="bg-white rounded-lg shadow-sm p-5 mt-6">
    <div class="flex items-center justify-between border-b border-slate-200 pb-3 mb-4">
      <h2 class="text-lg font-semibold text-slate-800">Episode outcome</h2>
      {% if ep.outcome == 'correct' %}
        <span class="pill ok" style="font-size:13px; padding:4px 14px;">✓ SUCCESS</span>
      {% elif ep.outcome == 'partial' %}
        <span class="pill mid" style="font-size:13px; padding:4px 14px;">◐ PARTIAL
          {%- if ep.question_type == 'relative_size' %} (letter correct, ratio wrong)
          {%- elif ep.question_type == 'existence' %} (option correct, point missed)
          {%- endif -%}
        </span>
      {% elif ep.outcome == 'wrong' %}
        <span class="pill bad" style="font-size:13px; padding:4px 14px;">✗ FAILURE</span>
      {% else %}
        <span class="pill nul" style="font-size:13px; padding:4px 14px;">{{ ep.status or 'unknown' }}</span>
      {% endif %}
    </div>

    <!-- Core answer block -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
      <div class="bg-emerald-50 border border-emerald-200 rounded-lg p-3">
        <div class="field-label text-emerald-800">✓ Correct answer</div>
        <div class="field-val font-mono font-bold text-emerald-900 text-lg">
          {{ ep.gt_answer or '—' }}{% if ep.gt_option_text and ep.gt_option_text != ep.gt_answer %} ({{ ep.gt_option_text }}){% endif %}
        </div>
        {% if ep.also_valid_display %}
        <div class="mt-2 border-t border-emerald-200 pt-2">
          <div class="text-xs font-semibold text-emerald-700">Also accepted:</div>
          {% for av in ep.also_valid_display %}
          <div class="text-sm text-emerald-600">{% if av.letter and av.letter != '—' %}{{ av.letter }}: {% endif %}{{ av.text }}</div>
          {% endfor %}
        </div>
        {% endif %}
      </div>
      <div class="{% if ep.outcome == 'correct' %}bg-emerald-50 border-emerald-200{% elif ep.outcome == 'partial' %}bg-amber-50 border-amber-200{% elif ep.outcome == 'wrong' %}bg-rose-50 border-rose-200{% else %}bg-slate-50 border-slate-200{% endif %} border rounded-lg p-3">
        <div class="field-label {% if ep.outcome == 'correct' %}text-emerald-800{% elif ep.outcome == 'partial' %}text-amber-800{% elif ep.outcome == 'wrong' %}text-rose-800{% else %}text-slate-600{% endif %}">
          {% if ep.outcome == 'correct' %}✓ VLM answered (correct){% elif ep.outcome == 'partial' %}◐ VLM answered (partial — letter OK, ratio wrong){% elif ep.outcome == 'wrong' %}✗ VLM answered (wrong){% else %}VLM answer{% endif %}
        </div>
        <div class="field-val font-mono font-bold text-lg
             {% if ep.outcome == 'correct' %}text-emerald-900
             {% elif ep.outcome == 'partial' %}text-amber-900
             {% elif ep.outcome == 'wrong' %}text-rose-900
             {% else %}text-slate-500{% endif %}">
          {{ ep.vlm_answer or '—' }}{% if ep.vlm_option_text and ep.vlm_option_text != ep.vlm_answer %} ({{ ep.vlm_option_text }}){% endif %}
        </div>
        {% if not ep.vlm_answer and ep.status %}
        <div class="text-sm text-slate-500 mt-1">Status: {{ ep.status }}</div>
        {% endif %}
      </div>
    </div>

    <!-- Question-type-specific extras -->
    {% if ep.answer_extras %}
    <div class="border-t border-slate-200 pt-3 mb-4">
      <div class="field-label mb-2">Additional answer details</div>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
        {% for k, v, extra_class in ep.answer_extras %}
        <div>
          <div class="text-xs font-semibold text-slate-600">{{ k }}</div>
          <div class="field-val font-mono {{ extra_class }}">{{ v }}</div>
        </div>
        {% endfor %}
      </div>
    </div>
    {% endif %}

    {% if ep.point_check_image %}
    <!-- Existence point check visualisation -->
    <div class="border-t border-slate-200 pt-3 mb-3">
      <div class="field-label mb-2">Point localisation (GT · green, Predicted · green/red)</div>
      <a href="{{ ep.point_check_image }}" target="_blank">
        <img src="{{ ep.point_check_image }}" class="preview rounded border border-slate-200" loading="lazy" style="max-width:400px;">
      </a>
    </div>
    {% endif %}

    <!-- Manipulation stats -->
    <div class="border-t border-slate-200 pt-3 text-sm">
      <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div><div class="field-label">VLM steps</div><div class="field-val font-mono">{{ ep.total_vlm_steps or 0 }}</div></div>
        <div><div class="field-label">Actions taken</div><div class="field-val font-mono">{{ ep.manipulation_actions or 0 }}</div></div>
        <div><div class="field-label">Grasps</div><div class="field-val font-mono">{{ ep.grasp_successes }}/{{ ep.grasp_attempts }}</div></div>
        <div><div class="field-label">Pushes</div><div class="field-val font-mono">{{ ep.push_successes }}/{{ ep.push_attempts }}</div></div>
        <div><div class="field-label">Objects removed</div><div class="field-val">{{ ep.objects_removed|join(', ') if ep.objects_removed else '—' }}</div></div>
      </div>
    </div>
  </div>

  <footer class="mt-6 text-xs text-slate-400">
    <a href="{{ back_to_index }}" class="text-indigo-600 hover:underline">← back to run</a>
  </footer>
</div>

<script>
document.getElementById('expand-all').addEventListener('click',
  () => document.querySelectorAll('details.step-card').forEach(d => d.open = true));
document.getElementById('collapse-all').addEventListener('click',
  () => document.querySelectorAll('details.step-card').forEach(d => d.open = false));
</script>

</body>
</html>
""")


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #

STEP_RX = re.compile(r"^step_(\d+)_(.+)$")


def truncate(s: str, n: int = 80) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def letter_or_num(i: int) -> str:
    return chr(ord("A") + i) if i < 26 else str(i)


def find_episode_dirs(rollout_dir: Path) -> list[Path]:
    """Find episode directories.  Supports three layouts:

    1. Flat:       rollout_dir/ep_NNN_*/
    2. Symlinked:  rollout_dir/ep_NNN_* -> worker_N/ep_NNN_*  (follows symlinks)
    3. Worker-only: rollout_dir/worker_N/ep_NNN_*  (no top-level entries)

    Top-level ep_* entries (including symlinks) are preferred.  Worker
    subdirectories are only scanned for episodes NOT found at the top level.
    """
    ep_dirs: dict[str, Path] = {}

    # Top-level ep_NNN_* (real dirs or symlinks — both work via is_dir())
    for p in rollout_dir.iterdir():
        if p.is_dir() and p.name.startswith("ep_"):
            ep_dirs[p.name] = p

    # Fallback: scan worker_N/ for any episodes missing from top level
    for w in sorted(rollout_dir.iterdir()):
        if w.is_dir() and w.name.startswith("worker_"):
            for p in w.iterdir():
                if p.is_dir() and p.name.startswith("ep_") and p.name not in ep_dirs:
                    ep_dirs[p.name] = p

    return sorted(ep_dirs.values(), key=lambda p: p.name)


def find_file(ep_dir: Path, suffix: str) -> Path | None:
    for p in ep_dir.iterdir():
        if p.is_file() and p.name.endswith(suffix):
            return p
    return None


def find_file_containing(ep_dir: Path, fragment: str) -> Path | None:
    for p in ep_dir.iterdir():
        if p.is_file() and fragment in p.name:
            return p
    return None


def load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"  [warn] could not read {p}: {e}")
        return None


def outcome_of(rec: dict) -> tuple[str, str]:
    """Return (outcome_key, status_short).

    For relative_size: if letter is correct but ratio is wrong → "partial".
    """
    status = (rec.get("status") or "").lower()
    if status == "answered":
        if rec.get("is_correct"):
            return ("correct", "")
        # Check for partial success: relative_size with correct letter but wrong ratio
        qtype = rec.get("question_type", "")
        if qtype == "relative_size" and rec.get("letter_correct"):
            return ("partial", "ratio ✗")
        # Check for partial success: existence with correct letter but wrong/missing point
        if qtype == "existence" and rec.get("letter_correct"):
            return ("partial", "point ✗")
        # V2 rearrangement: is_correct requires F1==1.0; any 0<F1<1 is a
        # partial (some but not all target objects correctly rearranged).
        if qtype == "rearrangement":
            _f1 = rec.get("rearrange_f1")
            if _f1 is not None and _f1 > 0.0:
                return ("partial", f"F1 {_f1:.2f}")
        return ("wrong", "")
    if status == "failed":
        return ("failed", "fail")
    if status == "error":
        return ("failed", "err")
    if status == "no_answer":
        return ("failed", "no ans")
    return ("failed", status or "?")


# Fields already rendered in the structured card — everything else goes into "extra_fields"
_CORE_RESPONSE_KEYS = {"tool", "reasoning", "plan", "belief"}


def build_steps(ep_dir: Path) -> list[dict]:
    """Scan vlm_steps/ and group files by step number."""
    vd = ep_dir / "vlm_steps"
    if not vd.is_dir():
        return []

    groups: dict[int, dict[str, Path]] = {}
    for p in vd.iterdir():
        if not p.is_file():
            continue
        m = STEP_RX.match(p.name)
        if not m:
            continue
        n = int(m.group(1))
        tag = m.group(2)  # e.g. "img1.png", "prompt.txt", "response.json", "point.png", "seg.png"
        groups.setdefault(n, {})[tag] = p

    steps: list[dict] = []
    for n in sorted(groups):
        g = groups[n]
        prompt_path = g.get("prompt.txt")
        resp_path = g.get("response.json")

        prompt = prompt_path.read_text() if prompt_path else ""
        response = None
        response_raw = ""
        if resp_path:
            raw = resp_path.read_text().strip()
            try:
                response = json.loads(raw)
                response_raw = json.dumps(response, indent=2)
            except Exception:
                response_raw = raw

        # Left-column input image (the RGB the VLM actually saw)
        input_image = None
        if "img1.png" in g:
            input_image = {
                "src": g["img1.png"].relative_to(ep_dir).as_posix(),
                "label": "Clean RGB (input to VLM)",
            }

        # Right-column output images (whatever the tool produced)
        output_images = []
        for tag, label in [
            ("img2.png",  "Labelled perception overlay"),
            ("point.png", "Pointing model output"),
            ("seg.png",   "SAM3 segmentation mask"),
        ]:
            if tag in g:
                output_images.append({
                    "label": label,
                    "src": g[tag].relative_to(ep_dir).as_posix(),
                })

        # Build tool summary and headline
        tool = (response or {}).get("tool") or "?"
        args_hint = ""
        headline = ""
        if response:
            if tool == "perceive_find_object" and response.get("object_name"):
                args_hint = f"(name={response['object_name']!r})"
                headline = f"looking for {response['object_name']!r}"
            elif tool == "perceive_get_obstructors" and response.get("object_name"):
                args_hint = f"(target={response['object_name']!r})"
                headline = f"finding blockers of {response['object_name']!r}"
            elif tool == "grasp_and_remove" and response.get("object_id") is not None:
                args_hint = f"(id={response['object_id']})"
                headline = f"grasp object #{response['object_id']}"
            elif tool == "push":
                parts = []
                if response.get("object_id") is not None: parts.append(f"id={response['object_id']}")
                if response.get("direction"): parts.append(f"dir={response['direction']}")
                if response.get("distance_cm"): parts.append(f"dist={response['distance_cm']}cm")
                if parts: args_hint = "(" + ", ".join(parts) + ")"
                headline = f"push #{response.get('object_id','?')} {response.get('direction','?')}"
            elif tool == "answer_question":
                args_hint = f"(answer={response.get('answer')!r})"
                headline = f"final answer: {response.get('answer')!r}"
            elif tool == "perceive_scene":
                headline = "full scene scan"

        tool_summary = f"{tool} {args_hint}".strip()

        # Extra fields: everything the VLM returned that isn't in the structured card
        extra_fields: list[tuple[str, str]] = []
        if response:
            for k, v in response.items():
                if k in _CORE_RESPONSE_KEYS:
                    continue
                # Pretty-print lists/dicts
                if isinstance(v, (list, dict)):
                    v_str = json.dumps(v, ensure_ascii=False)
                else:
                    v_str = str(v)
                extra_fields.append((k, v_str))

        steps.append({
            "n": n,
            "prompt": prompt,
            "response": response,
            "response_raw": response_raw,
            "input_image": input_image,
            "output_images": output_images,
            "tool_summary": tool_summary,
            "headline": headline,
            "extra_fields": extra_fields,
            "outcome": None,
            "outcome_class": "nul",
        })
    return steps


def _resolve_option_text(answer_letter: str, answer_options: list,
                         option_map: dict) -> str | None:
    """Given an answer like 'A' or 'B', resolve it to the option text."""
    if not answer_letter:
        return None
    # Try option_map first (letter → text)
    if option_map:
        text = option_map.get(answer_letter) or option_map.get(answer_letter.upper())
        if text:
            return str(text)
    # Try positional (A=0, B=1, C=2, D=3)
    if answer_options and len(answer_letter) == 1 and answer_letter.upper().isalpha():
        idx = ord(answer_letter.upper()) - ord('A')
        if 0 <= idx < len(answer_options):
            return str(answer_options[idx])
    # Maybe it IS the text itself
    if answer_options and answer_letter in answer_options:
        return answer_letter
    return None


_AB_POINT_RX = re.compile(r"^\s*([AB]):\s*point=\[(\d+),\s*(\d+)\]", re.MULTILINE)


def _parse_ab_points_from_prompt(ep_dir: Path) -> dict:
    """Extract A/B target points from step_1_prompt.txt.

    Points are on a [0,1000] normalized grid: [row, col].
    Returns {"A": (row, col), "B": (row, col)} or empty.
    """
    prompt_path = ep_dir / "vlm_steps" / "step_1_prompt.txt"
    if not prompt_path.is_file():
        return {}
    text = prompt_path.read_text()
    result = {}
    for m in _AB_POINT_RX.finditer(text):
        label = m.group(1)
        row, col = int(m.group(2)), int(m.group(3))
        result[label] = (row, col)
    return result


def _build_compare_marked_image(ep_dir: Path, rec: dict, qjson: dict) -> dict | None:
    """Highlight the three badged A/B/C objects for a compare episode.

    Draws a coloured box + large letter over each of the A/B/C objects on the
    clean step-1 frame, mirroring the human-review "Scene (target marked)"
    panel.  Object pixel boxes come from the step-1 simulator GT
    (``sim-gt-stats/step_1.json``): ``role_map`` maps obj_id→A/B/C and each
    ``objects[]`` entry carries an ``aabb_pixel``.  The GT-answer letter is
    outlined thicker and flagged in the legend.  Writes ``compare_marked.png``
    into the episode dir.  Returns {"path", "legend"} or None on failure.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    # Base: the clean frame the VLM actually saw at step 1.
    base_img = ep_dir / "vlm_steps" / "step_1_img1.png"
    if not base_img.is_file():
        base_img = ep_dir / "after_walls_topdown.png"
    if not base_img.is_file():
        return None

    # Step-1 simulator GT: role_map + per-object AABBs.
    sg_path = ep_dir / "sim-gt-stats" / "step_1.json"
    if not sg_path.is_file():
        return None
    try:
        sg = json.loads(sg_path.read_text())
    except Exception:
        return None
    role_map = {str(k): v for k, v in (sg.get("role_map") or {}).items()}
    aabb_by_id = {}
    for o in sg.get("objects", []):
        ab = o.get("aabb_pixel")
        if ab:
            aabb_by_id[str(o.get("obj_id"))] = (ab, o.get("name", ""))

    # Names + GT.
    marker_names = qjson.get("marker_obj_names") or {}
    gt_letter = str(rec.get("gt_answer") or qjson.get("gt_answer") or "").strip().upper()
    extremum = qjson.get("comparison_extremum") or "largest"

    img = cv2.imread(str(base_img))
    if img is None:
        return None
    img = np.ascontiguousarray(img)

    # Distinct, high-contrast colours per label (BGR for cv2 / hex for legend).
    colors_bgr = {"A": (60, 180, 250),    # amber/orange
                  "B": (240, 130, 40),    # blue
                  "C": (80, 200, 90)}     # green
    colors_hex = {"A": "#fab43c", "B": "#2882f0", "C": "#50c85a"}

    legend = []
    drawn_any = False
    for label in ("A", "B", "C"):
        # Find the obj_id assigned to this label via role_map.
        oid = next((k for k, v in role_map.items() if v == label), None)
        name = marker_names.get(label) or (aabb_by_id.get(oid, (None, ""))[1] if oid else "")
        is_gt = (label == gt_letter)
        if oid is None or oid not in aabb_by_id:
            legend.append({"color": colors_hex[label],
                           "label": f"{label}: {name or 'unknown'} "
                                    f"(not visible / removed)"
                                    + ("  ✓ GT" if is_gt else "")})
            continue
        ab, _nm = aabb_by_id[oid]
        x0, y0, x1, y1 = ab["x_min"], ab["y_min"], ab["x_max"], ab["y_max"]
        color = colors_bgr[label]
        thick = 7 if is_gt else 3
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thick, cv2.LINE_AA)
        # Big letter label with a filled tag at the top-left corner of the box.
        # NOTE: OpenCV's Hershey font is ASCII-only — use "= GT" (no ✓ glyph,
        # which renders as "???").
        tag = (label + " = GT") if is_gt else label
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
        ty = max(y0, th + 8)
        cv2.rectangle(img, (x0, ty - th - 8), (x0 + tw + 10, ty + 6),
                      color, -1)
        cv2.putText(img, tag, (x0 + 5, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        legend.append({"color": colors_hex[label],
                       "label": f"{label}: {name}"
                                + ("  ✓ GT (" + extremum + ")" if is_gt else "")})
        drawn_any = True

    if not drawn_any:
        return None

    out = ep_dir / "compare_marked.png"
    try:
        cv2.imwrite(str(out), img)
    except Exception:
        return None
    return {"path": out.relative_to(ep_dir).as_posix(), "legend": legend or None}


def _build_scene_image(ep_dir: Path, rec: dict, qjson: dict) -> dict | None:
    """Build annotated scene image with target points marked.

    Uses step_1_img1.png as base, draws point markers for targets.
    Returns {"path": relative_path, "legend": [...]} or None.
    """
    qtype = rec.get("question_type") or ""
    target_name = qjson.get("target_name") or rec.get("target_name")

    # ── marked_comparison (compare): highlight the three badged A/B/C objects,
    # like the human-review "Scene (target marked)" panel.  We draw coloured
    # boxes + big letters on the clean frame, using per-object AABBs from the
    # step-1 simulator GT (sim-gt-stats/step_1.json → objects[].aabb_pixel,
    # role_map[obj_id] = A/B/C).  This is the only reliable per-object pixel
    # source for compare (badges are baked into the render, not annotated).
    if qtype in ("marked_comparison", "compare"):
        marked = _build_compare_marked_image(ep_dir, rec, qjson)
        if marked is not None:
            return marked
        # else fall through to the generic handling below.

    # Prefer the demo.py-generated _gt_scene.png — it already has proper
    # bounding-box annotations (with buried-object dashed boxes) for
    # counting, existence, under_object, same_attribute, relative_size.
    # V2 task names reuse the same _gt_scene.png annotations produced by
    # core/gt_helpers._annotate_gt_scene (count→counting, find→existence,
    # beneath→under_object, plus a dedicated rearrangement branch).
    _V2_TO_V1 = {"count": "counting", "find": "existence", "beneath": "under_object"}
    _qtype_norm = _V2_TO_V1.get(qtype, qtype)

    gt_scene = find_file_containing(ep_dir, "_gt_scene.png")
    if gt_scene and (_qtype_norm in (
            "counting", "existence", "under_object",
            "same_attribute", "relative_size") or qtype == "rearrangement"):
        legend = []
        if _qtype_norm == "counting" and target_name:
            legend.append({"color": "#00c800",
                           "label": f"Target instances: {target_name}"})
        elif _qtype_norm == "existence" and target_name:
            gt_txt = (qjson.get("gt_answer_text") or "").lower()
            if gt_txt == "yes":
                legend.append({"color": "#00c800",
                               "label": f"Target (present): {target_name}"})
            else:
                legend.append({"color": "#dc3232",
                               "label": f"Target (absent from scene): {target_name}"})
        elif _qtype_norm == "under_object":
            cover = qjson.get("cover_object")
            hidden = target_name
            alts = qjson.get("also_valid") or []
            if cover:
                legend.append({"color": "#0078e6",
                               "label": f"Cover (blocking): {cover}"})
            if hidden:
                legend.append({"color": "#00c800",
                               "label": f"Hidden (answer): {hidden}"})
            for alt in alts:
                legend.append({"color": "#00ff8c",
                               "label": f"Also accepted: {alt}"})
        elif qtype == "rearrangement":
            # Reference (KEEP) drawn red; every GT-removal object drawn green.
            direction = qjson.get("direction") or rec.get("direction") or "?"
            ref_name = target_name
            rm_names = (qjson.get("gt_removal_names")
                        or rec.get("gt_removal_names") or [])
            if ref_name:
                legend.append({"color": "#dc3232",
                               "label": f"Reference (KEEP): {ref_name}"})
            for rn in rm_names:
                legend.append({"color": "#00c800",
                               "label": f"Remove (to the {direction}): {rn}"})
        elif _qtype_norm in ("same_attribute", "relative_size"):
            marker_names = qjson.get("marker_obj_names") or {}
            if marker_names.get("A"):
                legend.append({"color": "#ffc800",
                               "label": f"A: {marker_names['A']}"})
            if marker_names.get("B"):
                legend.append({"color": "#0078dc",
                               "label": f"B: {marker_names['B']}"})
        return {
            "path": gt_scene.relative_to(ep_dir).as_posix(),
            "legend": legend or None,
        }

    # Fall back to step_1_img1.png (clean Blender render) for the remaining
    # types (interactive_counting, etc.) or when the gt_scene file is missing.
    step1_img = ep_dir / "vlm_steps" / "step_1_img1.png"
    if not step1_img.is_file():
        if gt_scene:
            return {"path": gt_scene.relative_to(ep_dir).as_posix(), "legend": None}
        return None

    # For same_attribute / relative_size, parse A/B points from prompt (legacy path)
    ab_points = {}
    if qtype in ("same_attribute", "relative_size"):
        ab_points = _parse_ab_points_from_prompt(ep_dir)

    # Determine if we need annotation
    needs_annotation = bool(ab_points) or (qtype == "existence" and target_name)

    if not needs_annotation:
        return {
            "path": step1_img.relative_to(ep_dir).as_posix(),
            "legend": None,
        }

    # Generate annotated image
    try:
        import cv2
        import numpy as np

        img = cv2.imread(str(step1_img))
        if img is None:
            return {
                "path": step1_img.relative_to(ep_dir).as_posix(),
                "legend": None,
            }
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        legend = []

        if ab_points:
            # A/B target marking with crosshair
            # Get names from question json
            marker_names = qjson.get("marker_obj_names") or {}
            obj_a_name = marker_names.get("A") or qjson.get("target_name") or "Object A"
            obj_b_name = marker_names.get("B") or qjson.get("comparison_object") or "Object B"
            names = {"A": obj_a_name, "B": obj_b_name}

            colors = {"A": (0, 220, 0), "B": (220, 0, 220)}  # green, magenta (BGR)
            color_hex = {"A": "#00dc00", "B": "#dc00dc"}

            for label in ("A", "B"):
                if label not in ab_points:
                    legend.append({"color": color_hex[label],
                                   "label": f"{label}: {names[label]} (occluded)"})
                    continue
                row, col = ab_points[label]
                # Convert from [0,1000] grid to pixel coords
                px = int(col / 1000.0 * w)
                py = int(row / 1000.0 * h)

                color = colors[label]
                # Draw crosshair circle + letter
                cv2.circle(img, (px, py), 14, color, 3, cv2.LINE_AA)
                cv2.line(img, (px - 22, py), (px + 22, py), color, 2, cv2.LINE_AA)
                cv2.line(img, (px, py - 22), (px, py + 22), color, 2, cv2.LINE_AA)
                # Label with background for visibility
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
                lx, ly = px + 18, py - 10
                cv2.rectangle(img, (lx - 2, ly - th - 4), (lx + tw + 4, ly + 4),
                              (0, 0, 0), -1)
                cv2.putText(img, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
                legend.append({"color": color_hex[label],
                               "label": f"{label}: {names[label]}"})

        elif qtype == "existence" and target_name:
            # Try to find target point from the prompt's A/B block or occlusion data
            occ_path = find_file(ep_dir, "_occlusion.json")
            occ_data = load_json(occ_path) if occ_path else {}
            target_id = None
            target_centroid = None

            if occ_data:
                # Find the target's segmentation centroid from visible_area/full_area
                name_to_id = occ_data.get("name_to_id") or {}
                tid = name_to_id.get(target_name)
                if tid is not None:
                    target_id = tid
                # Also check if there's a centroid in the occlusion stats
                # We can approximate from the visible area center if pixel data is stored

            # Alternative: parse from step_1_prompt if existence uses point format
            # For existence, the prompt doesn't use A/B but may mention target indirectly
            # Just show the label without a point if we can't locate it
            legend.append({"color": "#00dc00",
                           "label": f"Target: {target_name}"})

        # Save annotated image
        annotated_path = ep_dir / "report_scene_annotated.png"
        cv2.imwrite(str(annotated_path), img)
        return {
            "path": annotated_path.relative_to(ep_dir).as_posix(),
            "legend": legend if legend else None,
        }

    except ImportError:
        # cv2 not available — just use raw image
        return {
            "path": step1_img.relative_to(ep_dir).as_posix(),
            "legend": None,
        }


def build_episode_record(rollout_dir: Path, ep_dir: Path, result_rec: dict | None) -> dict:
    """Combine results.jsonl record + on-disk files into a unified dict."""
    rec = dict(result_rec or {})

    qjson_path = find_file(ep_dir, "_question.json")
    qjson = load_json(qjson_path) if qjson_path else {}
    if qjson:
        # fill-in fields that results.jsonl sometimes omits
        for k in ("question_subtype", "target_name", "gt_answer_text",
                  "gt_ratio", "gt_volume_a", "gt_volume_b",
                  "answer_options", "option_map", "also_valid",
                  "marker_obj_ids", "ab_targets",
                  # V2 rearrangement GT lives only in _question.json
                  "reference_obj_id", "direction",
                  "gt_removal_ids", "gt_removal_names", "cover_object"):
            if rec.get(k) is None:
                rec[k] = qjson.get(k)

    # Video: prefer *_manipulation.mp4 (Blender-rendered, matches VLM view).
    # Fall back to *_vlm_video.mp4 (PyBullet dual-view: 3rd-person + top-down).
    # We show the PyBullet recording for ALL episodes — including ones with 0
    # manipulations (e.g. V2 count/find/beneath) — so every episode has a
    # video on the report.
    video = find_file(ep_dir, "_manipulation.mp4")
    if not video:
        video = find_file(ep_dir, "_vlm_video.mp4")
    rec["video"] = video.relative_to(ep_dir).as_posix() if video else None

    # Scene image: use Blender render from step 1 (clean RGB with proper lighting)
    # and generate a target-annotated version
    scene_image = _build_scene_image(ep_dir, rec, qjson)
    rec["scene_image"] = scene_image["path"] if scene_image else None
    rec["scene_image_legend"] = scene_image["legend"] if scene_image else None

    # Keep gt_scene_image for backwards compat but don't show it prominently
    gt_scene = find_file_containing(ep_dir, "_gt_scene.png")
    rec["gt_scene_image"] = gt_scene.relative_to(ep_dir).as_posix() if gt_scene else None

    # Point-check debug image (existence only — GT vs predicted point)
    point_check = find_file_containing(ep_dir, "_point_check.png")
    rec["point_check_image"] = (
        point_check.relative_to(ep_dir).as_posix() if point_check else None)

    outcome, status_short = outcome_of(rec)
    rec["outcome"] = outcome
    rec["status_short"] = status_short

    # Short display fields
    rec["question_short"] = truncate(rec.get("question", ""), 90)
    rec["gt_answer_short"] = truncate(str(rec.get("gt_answer") or ""), 24)
    rec["vlm_answer_short"] = truncate(str(rec.get("vlm_answer") or ""), 24)

    # Ensure numeric fields exist
    for k in ("grasp_attempts", "grasp_successes", "push_attempts", "push_successes",
              "manipulation_actions", "total_vlm_steps"):
        rec.setdefault(k, 0)

    # --- Resolve option text for GT and VLM answers ---
    answer_options = rec.get("answer_options") or qjson.get("answer_options") or []
    option_map = rec.get("option_map") or qjson.get("option_map") or {}
    gt_answer = str(rec.get("gt_answer") or "").strip()
    vlm_answer = str(rec.get("vlm_answer") or rec.get("answer") or "").strip()

    rec["gt_option_text"] = _resolve_option_text(gt_answer, answer_options, option_map)
    rec["vlm_option_text"] = _resolve_option_text(vlm_answer, answer_options, option_map)

    # ── V2 synthetic answer text for the "correct/predicted answer" block ──
    # These tasks have no letter/text GT, so build a human-readable summary.
    _qt = rec.get("question_type")
    if _qt == "rearrangement":
        _dir = rec.get("direction") or qjson.get("direction") or "?"
        _n = len(rec.get("gt_removal_ids") or qjson.get("gt_removal_ids") or [])
        _ref = rec.get("target_name") or qjson.get("target_name") or "the reference"
        rec["gt_answer_short"] = f"remove {_n} obj → {_dir} of {_ref}"
        rec["gt_option_text"] = None
        _f1 = rec.get("rearrange_f1")
        rec["vlm_answer_short"] = (f"F1={_f1:.2f}" if _f1 is not None
                                   else (rec.get("status") or "—"))
    elif _qt == "count":
        _gi = rec.get("gt_int")
        if _gi is None:
            try:
                _gi = int(gt_answer)
            except (TypeError, ValueError):
                _gi = gt_answer or "—"
        rec["gt_answer_short"] = str(_gi)
        rec["gt_option_text"] = None
        _pi = rec.get("predicted_int")
        rec["vlm_answer_short"] = str(_pi) if _pi is not None else (vlm_answer or "—")

    # also_valid display (for under_object etc.)
    also_valid = rec.get("also_valid") or qjson.get("also_valid") or []
    rec["also_valid_display"] = []
    if also_valid:
        for av in also_valid:
            av_str = str(av).strip()
            # Try to resolve as letter → text; if not a letter, it IS the text
            av_text = _resolve_option_text(av_str, answer_options, option_map)
            if av_text and av_text != av_str:
                rec["also_valid_display"].append({
                    "letter": av_str,
                    "text": av_text,
                })
            else:
                # It's already the text value (e.g. "lipstick box")
                # Find which letter it maps to, if any
                letter = ""
                for i, opt in enumerate(answer_options):
                    if opt.lower().strip() == av_str.lower().strip():
                        letter = chr(ord('A') + i)
                        break
                rec["also_valid_display"].append({
                    "letter": letter or "—",
                    "text": av_str,
                })

    # Answer extras — question-type-specific details surfaced in the final summary.
    # Each entry: (label, value, extra_css_class)
    qtype = rec.get("question_type")
    extras: list[tuple[str, str, str]] = []
    predicted_point = rec.get("predicted_point")  # may be None
    predicted_ratio = rec.get("predicted_ratio")

    # Fall back to parsing the final answer_question step if results.jsonl didn't store these
    if predicted_point is None or predicted_ratio is None:
        final_resp = _find_final_answer_response(ep_dir)
        if final_resp:
            if predicted_point is None:
                predicted_point = final_resp.get("point")
            if predicted_ratio is None:
                predicted_ratio = final_resp.get("ratio")

    if qtype in ("existence", "find"):
        vlm_ans_lower = str(rec.get("vlm_answer") or "").strip().lower()
        gt_ans_lower = str(rec.get("gt_answer") or "").strip().lower()
        # Show point info for "yes" answers
        if vlm_ans_lower in ("yes", "a") and predicted_point is not None:
            point_on_mask = rec.get("point_on_mask")
            point_label = f"{predicted_point}"
            if point_on_mask is True:
                point_label += " ✓ (on target mask)"
            elif point_on_mask is False:
                point_label += " ✗ (missed target mask)"
            extras.append(("Predicted point", point_label,
                           "text-emerald-700" if point_on_mask else
                           ("text-rose-700" if point_on_mask is False else "")))
        elif vlm_ans_lower in ("yes", "a") and predicted_point is None:
            extras.append(("Predicted point",
                           "— (not provided, required for 'yes' answers)",
                           "text-rose-700"))
    if qtype == "count":
        # Free-form integer: predicted vs GT.
        pred_int = rec.get("predicted_int")
        gt_int = rec.get("gt_int")
        if gt_int is None:
            try:
                gt_int = int(str(rec.get("gt_answer")).strip())
            except (TypeError, ValueError):
                gt_int = None
        ok = (pred_int is not None and gt_int is not None and pred_int == gt_int)
        extras.append(("Predicted count",
                       f"{pred_int if pred_int is not None else '—'} "
                       f"(GT: {gt_int if gt_int is not None else '—'}) "
                       f"{'✓' if ok else '✗'}",
                       "text-emerald-700" if ok else "text-rose-700"))
    if qtype == "rearrangement":
        # World-state F1 scoring: surface P / R / F1 + the removal sets.
        f1 = rec.get("rearrange_f1")
        prec = rec.get("rearrange_precision")
        rcl = rec.get("rearrange_recall")
        f1_ok = (f1 is not None and f1 >= 1.0)
        if f1 is not None:
            extras.append(("F1 (world-state)",
                           f"{f1:.3f}  (precision {prec:.3f}, recall {rcl:.3f})"
                           if prec is not None and rcl is not None
                           else f"{f1:.3f}",
                           "text-emerald-700" if f1_ok else "text-amber-700"))
        rm_ids = rec.get("gt_removal_ids") or qjson.get("gt_removal_ids") or []
        rm_names = rec.get("gt_removal_names") or qjson.get("gt_removal_names") or []
        direction = rec.get("direction") or qjson.get("direction") or "?"
        ref_name = rec.get("target_name") or qjson.get("target_name")
        if ref_name:
            extras.append(("Reference (must KEEP)", f"{ref_name}", "text-slate-700"))
        if rm_names:
            extras.append((f"Must remove (to the {direction})",
                           "; ".join(str(n) for n in rm_names), ""))
        elif rm_ids:
            extras.append((f"Must remove (to the {direction})",
                           ", ".join(f"id={i}" for i in rm_ids), ""))
        # What the agent actually did.
        removed = rec.get("removed_obj_ids")
        missed = rec.get("missed_ids")
        extra_rm = rec.get("extra_removed_ids")
        if removed is not None:
            extras.append(("Removed from view (correct)",
                           ", ".join(f"id={i}" for i in removed) if removed else "none",
                           "text-emerald-700" if removed else "text-slate-500"))
        if missed:
            extras.append(("Missed (should have removed)",
                           ", ".join(f"id={i}" for i in missed),
                           "text-rose-700"))
        if extra_rm:
            extras.append(("Over-removed (precision penalty)",
                           ", ".join(f"id={i}" for i in extra_rm),
                           "text-rose-700"))
        if rec.get("reference_removed"):
            extras.append(("⚠ Reference removed",
                           "the reference object was removed — invalid",
                           "text-rose-700"))
    if qtype == "under_object":
        # also_valid is now shown in the main answer block (also_valid_display),
        # but still surface match info here
        if rec.get("matched_also_valid"):
            extras.append(("Matched via",
                           "also_valid (alternative accepted answer)",
                           "text-emerald-700"))
    if qtype == "relative_size":
        gt_ratio = rec.get("gt_ratio")
        pred_ratio = predicted_ratio
        ratio_ok = rec.get("ratio_correct")
        letter_ok = rec.get("letter_correct")

        # Letter correctness (which object is larger)
        extras.append(("① Letter correct (which is larger?)",
                       ("yes ✓" if letter_ok else "no ✗") if letter_ok is not None else "—",
                       "text-emerald-700" if letter_ok else "text-rose-700"))

        # Ratio with acceptable range (±10% of GT)
        if gt_ratio is not None:
            try:
                gt_r = float(gt_ratio)
                lo = round(gt_r * 0.9, 3)
                hi = round(gt_r * 1.1, 3)
                extras.append(("② GT ratio (vol_A / vol_B)",
                               f"{gt_ratio}  [acceptable: {lo} – {hi}, ±10%]",
                               ""))
            except (ValueError, TypeError):
                extras.append(("② GT ratio (vol_A / vol_B)",
                               f"{gt_ratio}", ""))
        else:
            extras.append(("② GT ratio (vol_A / vol_B)", "—", ""))

        # Predicted ratio
        if pred_ratio is not None:
            ratio_label = f"{pred_ratio}"
            if ratio_ok:
                ratio_label += " ✓ (within ±10%)"
            else:
                ratio_label += " ✗ (outside ±10%)"
            extras.append(("② Predicted ratio",
                           ratio_label,
                           "text-emerald-700" if ratio_ok else "text-rose-700"))
        else:
            extras.append(("② Predicted ratio", "— (not provided)", "text-slate-500"))

        # Why this episode was marked correct/partial/wrong
        if letter_ok and ratio_ok:
            extras.append(("Verdict logic",
                           "Letter ✓ + Ratio ✓ → SUCCESS",
                           "text-emerald-700"))
        elif letter_ok and not ratio_ok:
            extras.append(("Verdict logic",
                           "Letter ✓ but Ratio ✗ → PARTIAL (counted as failure)",
                           "text-amber-700"))
        elif not letter_ok:
            extras.append(("Verdict logic",
                           "Letter ✗ → FAILURE (wrong object identified as larger)",
                           "text-rose-700"))

        gv_a = rec.get("gt_volume_a")
        gv_b = rec.get("gt_volume_b")
        if gv_a is not None and gv_b is not None:
            extras.append(("Object volumes (GT)",
                           f"A = {gv_a} cm³, B = {gv_b} cm³",
                           ""))
    rec["answer_extras"] = extras

    return rec


def _find_final_answer_response(ep_dir: Path) -> dict | None:
    """Return the parsed response of the last answer_question step, if any."""
    vd = ep_dir / "vlm_steps"
    if not vd.is_dir():
        return None
    # Find highest N for which response.json contains tool=answer_question
    best: tuple[int, dict] | None = None
    for p in vd.glob("step_*_response.json"):
        m = STEP_RX.match(p.name)
        if not m:
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("tool") == "answer_question":
            n = int(m.group(1))
            if best is None or n > best[0]:
                best = (n, d)
    return best[1] if best else None


def write_episode_report(rollout_dir: Path, ep_dir: Path, ep_rec: dict) -> None:
    steps = build_steps(ep_dir)

    # Mark answer_question steps with the final verdict
    if steps and ep_rec.get("outcome"):
        for s in steps:
            if s["response"] and s["response"].get("tool") == "answer_question":
                if ep_rec["outcome"] == "correct":
                    s["outcome"] = "answer ✓"; s["outcome_class"] = "ok"
                elif ep_rec["outcome"] == "wrong":
                    s["outcome"] = "answer ✗"; s["outcome_class"] = "bad"

    # Compute relative path back to rollout root for the "back" link
    depth = len(ep_dir.relative_to(rollout_dir).parts)  # 1 for ep_NNN, 2 for worker_N/ep_NNN
    back_to_index = "../" * depth + "index.html"

    html = EPISODE_TMPL.render(
        ep=ep_rec,
        steps=steps,
        rollout_name=rollout_dir.name,
        letter_or_num=letter_or_num,
        back_to_index=back_to_index,
    )
    out = ep_dir / "report.html"
    out.write_text(html)
    print(f"  wrote {out.relative_to(rollout_dir.parent)}  ({len(steps)} steps)")


def build_rollout(rollout_dir: Path) -> None:
    rollout_dir = rollout_dir.resolve()
    if not rollout_dir.is_dir():
        print(f"[error] not a directory: {rollout_dir}")
        return

    print(f"\n=== {rollout_dir.name} ===")

    cfg_path = rollout_dir / "run_config.json"
    summary_path = rollout_dir / "summary.json"
    results_path = rollout_dir / "results.jsonl"

    cfg = load_json(cfg_path) if cfg_path.is_file() else {}
    summary = load_json(summary_path) if summary_path.is_file() else {}

    results_by_ep: dict[int, dict] = {}
    if results_path.is_file():
        for line in results_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                results_by_ep[rec["episode"]] = rec
            except Exception as e:
                print(f"  [warn] bad results.jsonl line: {e}")

    ep_dirs = find_episode_dirs(rollout_dir)
    if not ep_dirs:
        print("  [warn] no episode directories found")
        return

    episodes: list[dict] = []
    for ep_dir in ep_dirs:
        m = re.match(r"ep_(\d+)_", ep_dir.name)
        if not m:
            continue
        ep_n = int(m.group(1))
        rec = results_by_ep.get(ep_n)
        ep_rec = build_episode_record(rollout_dir, ep_dir, rec)
        ep_rec["episode"] = ep_n
        ep_rec["report_path"] = f"{ep_dir.relative_to(rollout_dir).as_posix()}/report.html"
        episodes.append(ep_rec)
        write_episode_report(rollout_dir, ep_dir, ep_rec)

    qtypes = sorted({e.get("question_type", "") for e in episodes if e.get("question_type")})

    # Ensure summary has fields we reference in the template.
    # Always RECOMPUTE from episode records — the on-disk summary.json
    # stores accuracy as a decimal (0.5) but the template expects a
    # percentage (50.0).
    sm = dict(summary or {})
    n_total = len(episodes)
    n_answered = sum(1 for e in episodes if e.get("status") == "answered")
    n_correct = sum(1 for e in episodes if e.get("outcome") == "correct")
    n_fail = sum(1 for e in episodes if e.get("outcome") in ("failed",))
    n_error = sum(1 for e in episodes if (e.get("status") or "").lower() == "error")
    sm["n_total"] = n_total
    sm["n_answered"] = n_answered
    sm["n_correct"] = n_correct
    sm["n_fail"] = n_fail
    sm["n_error"] = n_error
    sm["accuracy"] = 100.0 * n_correct / max(1, n_total)
    sm["accuracy_answered"] = 100.0 * n_correct / max(1, n_answered)
    sm.setdefault("avg_actions",
                  sum(e.get("manipulation_actions", 0) for e in episodes) / max(1, n_total))

    # Fix manipulation success rates: on-disk values may be decimals (0-1)
    # but the template expects percentages (0-100).
    manip = sm.get("manipulation")
    if manip:
        for section_key in ("grasps", "pushes"):
            section = manip.get(section_key, {})
            sr = section.get("success_rate", 0)
            if sr is not None and 0 < sr < 1.01:
                section["success_rate"] = sr * 100.0

    # Rebuild per_question_type from episode records so template gets correct values
    per_type: dict = {}
    for ep in episodes:
        qt = ep.get("question_type")
        if not qt:
            continue
        per_type.setdefault(qt, {"total": 0, "correct": 0})
        per_type[qt]["total"] += 1
        if ep.get("outcome") == "correct":
            per_type[qt]["correct"] += 1
    sm["per_question_type"] = per_type

    html = INDEX_TMPL.render(
        rollout_name=rollout_dir.name,
        cfg=cfg,
        summary=sm,
        episodes=episodes,
        qtypes=qtypes,
    )
    (rollout_dir / "index.html").write_text(html)
    print(f"  wrote {(rollout_dir / 'index.html').relative_to(rollout_dir.parent)}")
    print(f"  → open {rollout_dir / 'index.html'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rollouts", nargs="+", type=Path,
                    help="One or more rollout directories (e.g. rollouts/rollout_vlm_XXX).")
    args = ap.parse_args()

    for r in args.rollouts:
        build_rollout(r)


if __name__ == "__main__":
    main()
