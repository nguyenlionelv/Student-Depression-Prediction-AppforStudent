/* ─── Step navigation ──────────────────────────────────────────────────────── */
let currentStep = 1;
const TOTAL_STEPS = 3;

function showStep(n) {
  for (let i = 1; i <= TOTAL_STEPS; i++) {
    document.getElementById(`step-${i}`).style.display = i === n ? '' : 'none';
    const pill = document.getElementById(`pill-${i}`);
    if (i < n) pill.className = 'step-pill done';
    else if (i === n) pill.className = 'step-pill active';
    else pill.className = 'step-pill';
  }
  currentStep = n;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ─── Validation ─────────────────────────────────────────────────────────────── */
function clearValidationErrors(step) {
  step.querySelectorAll('select, input').forEach(el => el.style.borderColor = '');
  step.querySelectorAll('.scale-group').forEach(g => { g.style.outline = ''; g.style.padding = ''; });
  step.querySelectorAll('.form-group').forEach(g => { g.style.outline = ''; g.style.padding = ''; });
}

function validateStep(n) {
  const step = document.getElementById(`step-${n}`);
  clearValidationErrors(step);
  let ok = true;
  let firstError = null;

  // 1. Visible inputs: select, number, text
  step.querySelectorAll('select, input[type=number], input[type=text]').forEach(el => {
    if (el.hasAttribute('required') && !el.value.trim()) {
      el.style.borderColor = 'var(--warn)';
      ok = false;
      if (!firstError) firstError = el;
    }
  });

  // 2. Hidden inputs backed by scale buttons
  step.querySelectorAll('input[type=hidden]').forEach(el => {
    if (el.hasAttribute('required') && !el.value.trim()) {
      ok = false;
      const name = el.name;
      const group = step.querySelector(`.scale-group[data-name="${name}"]`);
      if (group) {
        group.style.outline = '2px solid var(--warn)';
        group.style.borderRadius = '8px';
        group.style.padding = '4px';
        if (!firstError) firstError = group;
      }
    }
  });

  // 3. Radio groups (pill buttons)
  const radioNames = new Set();
  step.querySelectorAll('input[type=radio][required]').forEach(el => radioNames.add(el.name));
  radioNames.forEach(name => {
    if (!step.querySelector(`input[name="${name}"]:checked`)) {
      ok = false;
      const firstRadio = step.querySelector(`input[name="${name}"]`);
      const formGroup = firstRadio?.closest('.form-group');
      if (formGroup) {
        formGroup.style.outline = '2px solid var(--warn)';
        formGroup.style.borderRadius = '8px';
        formGroup.style.padding = '4px';
        if (!firstError) firstError = formGroup;
      }
    }
  });

  if (!ok && firstError) firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
  return ok;
}

function nextStep(n) {
  if (validateStep(n)) showStep(n + 1);
}
function prevStep(n) { showStep(n - 1); }

/* ─── Scale buttons ──────────────────────────────────────────────────────────── */
document.querySelectorAll('.scale-group').forEach(group => {
  const name = group.dataset.name;
  const input = document.querySelector(`input[name="${name}"]`);

  group.querySelectorAll('.scale-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      group.querySelectorAll('.scale-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (input) {
        input.value = btn.dataset.val;
        // Clear validation highlight
        group.style.outline = '';
        group.style.padding = '';
      }
    });
  });
});

/* ─── Form submit ─────────────────────────────────────────────────────────────── */
document.getElementById('survey-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!validateStep(3)) return;

  // Build payload
  const fd = new FormData(e.target);
  const data = Object.fromEntries(fd.entries());

  // Coerce numeric fields
  ['age', 'cgpa', 'academic_pressure', 'work_pressure',
    'study_satisfaction', 'job_satisfaction',
    'work_study_hours', 'financial_stress'].forEach(k => {
      if (data[k] !== undefined) data[k] = parseFloat(data[k]) || 0;
    });

  console.log('[MindCheck] Submitting payload:', data);
  showLoading(true);

  // Support both same-origin (port 8000) and cross-origin (VS Code Live Server etc.)
  const API_BASE = (window.location.port === '8000' || window.location.port === '')
    ? ''
    : 'http://localhost:8000';

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });

    console.log('[MindCheck] Response status:', res.status);

    let result;
    const text = await res.text();
    console.log('[MindCheck] Raw response:', text.slice(0, 200));

    try {
      result = JSON.parse(text);
    } catch {
      throw new Error(`Server trả về response không hợp lệ (${res.status}): ${text.slice(0, 100)}`);
    }

    if (!res.ok) {
      throw new Error(result.detail || `Lỗi server ${res.status}`);
    }

    showLoading(false);
    renderResult(result);

  } catch (err) {
    showLoading(false);
    console.error('[MindCheck] Error:', err);
    alert(`❌ ${err.message}`);
  }
});

/* ─── Render result ───────────────────────────────────────────────────────────── */
function renderResult(r) {
  console.log('[MindCheck] Rendering result:', r);

  // Hide all form steps
  for (let i = 1; i <= TOTAL_STEPS; i++)
    document.getElementById(`step-${i}`).style.display = 'none';
  document.getElementById('pill-4').className = 'step-pill active';

  const section = document.getElementById('result-section');
  section.style.display = 'block';
  setTimeout(() => section.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);

  // Emoji
  const emojiMap = { 'Thấp': '😊', 'Trung bình': '😐', 'Cao': '😟', 'Rất cao': '😢' };
  document.getElementById('result-emoji').textContent = emojiMap[r.risk_level] || '🧠';

  // Risk badge
  const badge = document.getElementById('result-badge');
  badge.textContent = `Mức nguy cơ: ${r.risk_level}`;
  badge.className = `risk-badge risk-${r.risk_level.split(' ')[0]}`;

  // Probability bar
  const pct = Math.round((r.probability || 0) * 100);
  document.getElementById('prob-text').textContent = `${pct}%`;
  setTimeout(() => {
    const bar = document.getElementById('prob-bar');
    if (bar) bar.style.width = `${pct}%`;
  }, 150);

  // Explanation
  document.getElementById('explanation-text').innerHTML = r.explanation || '';

  // // Factors
  // const factorsList = document.getElementById('factors-list');
  // factorsList.innerHTML = '';
  // (r.top_factors || []).forEach(f => {
  //   const isUp = (f.direction || '').includes('tăng');
  //   const chip = document.createElement('div');
  //   chip.className = 'factor-chip';
  //   chip.innerHTML = `
  //     <span class="arrow ${isUp ? 'up' : 'down'}">${isUp ? '▲' : '▼'}</span>
  //     <span><strong>${f.label_vi || f.feature}</strong></span>
  //     <span class="text-muted" style="font-size:.82rem">— ${f.direction}, mức ${f.impact_level}</span>
  //   `;
  //   factorsList.appendChild(chip);
  // });

  // Recommendations
  const recsList = document.getElementById('recs-list');
  recsList.innerHTML = '';
  (r.recommendations || []).forEach(rec => {
    const li = document.createElement('li');
    li.textContent = rec;
    recsList.appendChild(li);
  });
}

/* ─── Reset ───────────────────────────────────────────────────────────────────── */
function resetForm() {
  document.getElementById('survey-form').reset();
  document.querySelectorAll('.scale-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('result-section').style.display = 'none';
  document.getElementById('pill-4').className = 'step-pill';
  showStep(1);
}

/* ─── Loading ─────────────────────────────────────────────────────────────────── */
function showLoading(show) {
  document.getElementById('loading').style.display = show ? 'flex' : 'none';
}
