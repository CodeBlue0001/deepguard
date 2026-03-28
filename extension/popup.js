// DeepGuard — Popup Script

const LABELS = {
  safe:    'All clear',
  warning: 'Suspicious content',
  danger:  'Deepfake detected!',
  idle:    'No video playing'
};

const SUBS = {
  safe:    'No manipulation signals detected',
  warning: 'Some signals triggered — watch closely',
  danger:  'High confidence fake video — be cautious',
  idle:    'DeepGuard will activate when a video plays'
};

const SIGNAL_KEYS = ['visual', 'audio', 'lipsync'];

// ── DOM refs ──

const body       = document.body;
const statusLabel = document.getElementById('status-label');
const statusSub   = document.getElementById('status-sub');
const scorePct    = document.getElementById('score-pct');
const scoreFill   = document.getElementById('score-bar-fill');
const noVideoMsg  = document.getElementById('no-video-msg');
const signalsSec  = document.getElementById('signals-section');
const historyList = document.getElementById('history-list');
const historyEmpty = document.getElementById('history-empty');

// ── Update UI ──

function applyState(state, score, signals) {
  // Body class for colour theming
  body.className = `state-${state}`;

  statusLabel.textContent = LABELS[state] || 'Scanning...';
  statusSub.textContent   = SUBS[state]   || '';

  const pct = Math.round((score || 0) * 100);
  scorePct.textContent  = `${pct}%`;
  scoreFill.style.width = `${pct}%`;

  // Show/hide sections
  if (state === 'idle') {
    noVideoMsg.classList.remove('hidden');
    signalsSec.classList.add('hidden');
  } else {
    noVideoMsg.classList.add('hidden');
    signalsSec.classList.remove('hidden');
  }

  // Update signal bars
  if (signals && signals.length) {
    signals.forEach((sig, i) => {
      const key  = SIGNAL_KEYS[i];
      const row  = document.querySelector(`.signal-row[data-key="${key}"]`);
      if (!row) return;
      const fill = row.querySelector('.signal-bar-fill');
      const val  = row.querySelector('.signal-val');
      const pctSig = Math.round(sig.score * 100);
      fill.style.width = `${pctSig}%`;
      fill.style.background = pctSig > 72 ? '#d63939' : pctSig > 45 ? '#e89e1a' : '#27a96c';
      val.textContent = `${pctSig}%`;
    });
  }
}

function renderHistory(items) {
  historyList.innerHTML = '';
  if (!items || items.length === 0) {
    historyEmpty.classList.remove('hidden');
    return;
  }
  historyEmpty.classList.add('hidden');
  items.slice(0, 10).forEach(item => {
    const div = document.createElement('div');
    div.className = 'history-item';

    const hostname = (() => {
      try { return new URL(item.url).hostname; } catch { return item.url || 'unknown'; }
    })();

    div.innerHTML = `
      <div class="history-dot ${item.state}"></div>
      <span class="history-url" title="${item.url}">${hostname}</span>
      <span class="history-score">${Math.round(item.score * 100)}%</span>
    `;
    historyList.appendChild(div);
  });
}

// ── Init ──

async function init() {
  // Get current tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;

  // Get current state from background
  chrome.runtime.sendMessage({ type: 'GET_STATE' }, (res) => {
    if (chrome.runtime.lastError) return;
    applyState(res?.state || 'idle', res?.score || 0, res?.signals);
  });

  // Get history
  chrome.runtime.sendMessage({ type: 'GET_HISTORY' }, (items) => {
    if (chrome.runtime.lastError) return;
    renderHistory(items || []);
  });

  // Live updates from background while popup is open
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === 'STATE_UPDATE') {
      applyState(msg.state, msg.score, msg.signals);
    }
  });
}

init();
