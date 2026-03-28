// DeepGuard v2 — Content Script
// Cloud-connected detection + human-in-the-loop feedback system.

(function () {
  'use strict';
  if (window.__deepguardV2) return;
  window.__deepguardV2 = true;

  // ─── Config ────────────────────────────────────────────────────────────────
  const API_BASE       = 'https://your-deepguard-api.com';  // ← set your deployed URL
  const SAMPLE_MS      = 1500;     // send a frame to the cloud every 1.5s
  const SESSION_ID     = _getSessionId();
  const PROMPT_COOLDOWN_MS = 60000; // don't re-prompt the same video for 60s

  // ─── State ─────────────────────────────────────────────────────────────────
  const videoState   = new WeakMap(); // video → { detectionId, score, state, lastPrompt }
  const monitored    = new WeakSet();

  // ─── Session ID (anonymous, per-install) ────────────────────────────────
  function _getSessionId() {
    let id = localStorage.getItem('__dg_sid');
    if (!id) { id = crypto.randomUUID(); localStorage.setItem('__dg_sid', id); }
    return id;
  }

  // ─── Canvas frame capture ──────────────────────────────────────────────────
  function captureFrameB64(video, w = 224, h = 224) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    c.getContext('2d').drawImage(video, 0, 0, w, h);
    return c.toDataURL('image/jpeg', 0.7);
  }

  // ─── Audio feature extraction ──────────────────────────────────────────────
  const audioCtxMap = new WeakMap();

  function getAudioFeatures(video) {
    try {
      if (!audioCtxMap.has(video)) {
        const ctx      = new AudioContext();
        const src      = ctx.createMediaElementSource(video);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        src.connect(analyser);
        analyser.connect(ctx.destination);
        audioCtxMap.set(video, analyser);
      }
      const analyser = audioCtxMap.get(video);
      const buf      = new Float32Array(analyser.frequencyBinCount);
      analyser.getFloatFrequencyData(buf);
      return Array.from(buf);
    } catch { return []; }
  }

  // ─── Lip-sync data (basic) ──────────────────────────────────────────────
  function getLipSyncData(video) {
    // In a full implementation, track mouth landmarks via MediaPipe
    // and correlate with audio peaks. This sends a stub for now.
    return { desync_frames: 0, total_frames: 1 };
  }

  // ─── API calls ─────────────────────────────────────────────────────────────
  async function callDetect(video) {
    const frame_b64     = captureFrameB64(video);
    const audio_features = getAudioFeatures(video);
    const lipsync_data  = getLipSyncData(video);

    const res = await fetch(`${API_BASE}/detect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: SESSION_ID,
        url:        location.href,
        frame_b64,
        audio_features,
        lipsync_data
      })
    });
    if (!res.ok) throw new Error(`Detect HTTP ${res.status}`);
    return res.json();
  }

  async function callFeedback(detectionId, label, trigger, video) {
    const frame_b64      = video ? captureFrameB64(video) : null;
    const audio_features = video ? getAudioFeatures(video) : [];
    const res = await fetch(`${API_BASE}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        detection_id:   detectionId,
        session_id:     SESSION_ID,
        user_label:     label,
        trigger,
        frame_b64,
        audio_features
      })
    });
    return res.ok ? res.json() : null;
  }

  // ─── CSS injection ────────────────────────────────────────────────────────
  function injectStyles() {
    if (document.getElementById('__dg2-css')) return;
    const s = document.createElement('style');
    s.id = '__dg2-css';
    s.textContent = `
      .__dg2-wrap { position: relative !important; display: inline-block !important; }

      .__dg2-badge {
        position: absolute; top: 10px; left: 10px; z-index: 2147483640;
        display: flex; align-items: center; gap: 7px;
        padding: 5px 10px 5px 8px; border-radius: 20px;
        font: 600 12px/1 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        transition: background .4s, border-color .4s; cursor: default;
        border: 1.5px solid transparent; pointer-events: auto;
      }
      .__dg2-badge.safe    { background: rgba(21,128,61,.9);  border-color: rgba(134,239,172,.4); }
      .__dg2-badge.warning { background: rgba(161,98,7,.93);  border-color: rgba(253,211,77,.4); }
      .__dg2-badge.danger  { background: rgba(153,27,27,.96); border-color: rgba(252,165,165,.5);
                             animation: __dg2-pulse 1.5s ease-in-out infinite; }
      .__dg2-badge.scanning { background: rgba(30,30,40,.7); border-color: rgba(255,255,255,.15); }

      @keyframes __dg2-pulse {
        0%,100% { border-color: rgba(252,165,165,.4); }
        50%      { border-color: rgba(252,165,165,1); }
      }

      .__dg2-dot {
        width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
        background: #fff; opacity: .7;
      }
      .__dg2-badge.warning .__dg2-dot,
      .__dg2-badge.danger  .__dg2-dot { animation: __dg2-blink .7s step-end infinite; }
      @keyframes __dg2-blink { 50% { opacity: 0; } }

      .__dg2-text { color: #fff; white-space: nowrap; }
      .__dg2-score { font-size: 10px; color: rgba(255,255,255,.65); margin-left: 2px; }

      .__dg2-report-btn {
        position: absolute; top: 10px; right: 10px; z-index: 2147483640;
        padding: 4px 10px; border-radius: 6px; font: 500 11px -apple-system,sans-serif;
        background: rgba(0,0,0,.55); color: rgba(255,255,255,.8);
        border: 1px solid rgba(255,255,255,.2); cursor: pointer;
        transition: background .15s; display: none;
      }
      .__dg2-report-btn:hover { background: rgba(0,0,0,.8); color: #fff; }

      .__dg2-prompt {
        position: absolute; bottom: 14px; left: 50%; transform: translateX(-50%);
        z-index: 2147483641; background: rgba(15,15,20,.97);
        border: 1px solid rgba(255,255,255,.1); border-radius: 14px;
        padding: 14px 16px; min-width: 290px; max-width: 380px;
        font: 13px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        color: #e8e8e8; animation: __dg2-slide-up .25s ease;
      }
      @keyframes __dg2-slide-up {
        from { opacity:0; transform: translateX(-50%) translateY(10px); }
        to   { opacity:1; transform: translateX(-50%) translateY(0); }
      }
      .__dg2-prompt-title { font-weight: 600; font-size: 13px; color: #fff; margin-bottom: 5px; }
      .__dg2-prompt-sub   { font-size: 11px; color: #888; margin-bottom: 12px; line-height: 1.5; }
      .__dg2-prompt-btns  { display: flex; gap: 8px; }

      .__dg2-btn {
        flex: 1; padding: 7px 0; border-radius: 8px; font: 500 12px sans-serif;
        border: 1px solid rgba(255,255,255,.12); cursor: pointer;
        transition: background .15s, border-color .15s; color: #fff; background: transparent;
      }
      .__dg2-btn:hover { background: rgba(255,255,255,.08); border-color: rgba(255,255,255,.25); }
      .__dg2-btn.fake  { background: rgba(153,27,27,.6);  border-color: rgba(252,165,165,.4); }
      .__dg2-btn.fake:hover  { background: rgba(153,27,27,.9); }
      .__dg2-btn.real  { background: rgba(21,128,61,.5);  border-color: rgba(134,239,172,.4); }
      .__dg2-btn.real:hover  { background: rgba(21,128,61,.85); }
      .__dg2-btn.skip  { color: #666; font-size: 11px; flex: 0.5; }

      .__dg2-thanks {
        font-size: 12px; color: #27a96c; text-align: center;
        padding: 6px 0; animation: __dg2-fade-in .3s ease;
      }
      .__dg2-training { font-size: 10px; color: #e89e1a; text-align: center; margin-top: 4px; }
      @keyframes __dg2-fade-in { from { opacity: 0; } to { opacity: 1; } }
    `;
    document.head.appendChild(s);
  }

  // ─── Overlay DOM ──────────────────────────────────────────────────────────
  function createOverlay(video) {
    const parent  = video.parentElement;
    const wrapper = document.createElement('div');
    wrapper.className = '__dg2-wrap';

    const badge = document.createElement('div');
    badge.className = '__dg2-badge scanning';
    badge.innerHTML = `<div class="__dg2-dot"></div><span class="__dg2-text">Scanning…</span>`;

    const reportBtn = document.createElement('button');
    reportBtn.className = '__dg2-report-btn';
    reportBtn.textContent = 'Report as fake';

    parent.insertBefore(wrapper, video);
    wrapper.appendChild(video);
    wrapper.appendChild(badge);
    wrapper.appendChild(reportBtn);

    // Show report button on hover
    wrapper.addEventListener('mouseenter', () => { reportBtn.style.display = 'block'; });
    wrapper.addEventListener('mouseleave', () => { reportBtn.style.display = 'none'; });

    return { wrapper, badge, reportBtn };
  }

  function updateBadge(badge, state, score) {
    badge.className = `__dg2-badge ${state}`;
    const labels = { safe: 'All clear', warning: 'Suspicious', danger: 'Deepfake detected', scanning: 'Scanning…' };

    // Update or create inner HTML
    let dotEl   = badge.querySelector('.__dg2-dot');
    let textEl  = badge.querySelector('.__dg2-text');
    let scoreEl = badge.querySelector('.__dg2-score');

    if (!dotEl)  { dotEl  = document.createElement('div');  dotEl.className  = '__dg2-dot';   badge.appendChild(dotEl);  }
    if (!textEl) { textEl = document.createElement('span'); textEl.className = '__dg2-text';  badge.appendChild(textEl); }
    if (!scoreEl){ scoreEl= document.createElement('span'); scoreEl.className= '__dg2-score'; badge.appendChild(scoreEl);}

    textEl.textContent  = labels[state] || state;
    scoreEl.textContent = score != null ? `${Math.round(score * 100)}%` : '';
  }

  // ─── Feedback prompt ──────────────────────────────────────────────────────
  function showFeedbackPrompt(video, wrapper, detectionId, trigger, score) {
    // Remove existing prompt if any
    wrapper.querySelector('.__dg2-prompt')?.remove();

    const pct  = Math.round(score * 100);
    const isUncertain = trigger === 'auto_prompt';
    const title = isUncertain
      ? `Uncertain — is this video AI-generated?`
      : `You reported this as fake. Confirm?`;
    const sub = isUncertain
      ? `Our model is ${pct}% confident this might be a deepfake. Your feedback trains the model in real time.`
      : `Your feedback helps DeepGuard learn and improve for everyone.`;

    const prompt = document.createElement('div');
    prompt.className = '__dg2-prompt';
    prompt.innerHTML = `
      <div class="__dg2-prompt-title">${title}</div>
      <div class="__dg2-prompt-sub">${sub}</div>
      <div class="__dg2-prompt-btns">
        <button class="__dg2-btn fake" data-label="fake">Yes, it's fake</button>
        <button class="__dg2-btn real" data-label="real">No, it's real</button>
        <button class="__dg2-btn skip" data-label="unsure">Skip</button>
      </div>
    `;

    wrapper.appendChild(prompt);

    // Button handlers
    prompt.querySelectorAll('.__dg2-btn').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const label = btn.dataset.label;

        // Show loading state
        prompt.innerHTML = `<div class="__dg2-thanks">Submitting feedback…</div>`;

        const fbResult = await callFeedback(detectionId, label, trigger, video);

        // Show result
        let msg = 'Feedback recorded. Thank you!';
        let trainingMsg = '';
        if (fbResult) {
          if (fbResult.instant_updated)    msg = 'Model adapted instantly to your feedback!';
          if (fbResult.retrain_triggered)  trainingMsg = 'Full model retrain triggered in background.';
        }

        prompt.innerHTML = `
          <div class="__dg2-thanks">${msg}</div>
          ${trainingMsg ? `<div class="__dg2-training">${trainingMsg}</div>` : ''}
        `;

        // Notify background
        chrome.runtime.sendMessage({ type: 'FEEDBACK_SUBMITTED', label, fbResult });

        setTimeout(() => prompt.remove(), 3000);
      });
    });

    // Auto-close after 20s
    setTimeout(() => prompt.remove(), 20000);
  }

  // ─── Detection loop ────────────────────────────────────────────────────────
  async function runDetection(video, overlay) {
    const { badge, reportBtn, wrapper } = overlay;
    if (video.paused || video.ended || video.readyState < 2) return;
    if (video.videoWidth < 20) return;

    try {
      const result = await callDetect(video);

      updateBadge(badge, result.state, result.score);

      const vs = videoState.get(video) || {};
      vs.detectionId = result.detection_id;
      vs.score       = result.score;
      vs.state       = result.state;
      videoState.set(video, vs);

      // Report to background for toolbar icon
      chrome.runtime.sendMessage({
        type:    'DETECTION_UPDATE',
        score:   result.score,
        state:   result.state,
        signals: result.signals,
        url:     location.href
      }).catch(() => {});

      // Auto-prompt if model says uncertain and we haven't prompted recently
      const now = Date.now();
      if (result.should_prompt && (!vs.lastPrompt || now - vs.lastPrompt > PROMPT_COOLDOWN_MS)) {
        vs.lastPrompt = now;
        videoState.set(video, vs);
        showFeedbackPrompt(video, wrapper, result.detection_id, 'auto_prompt', result.score);
      }
    } catch (err) {
      // Fallback: show scanning badge on API error
      updateBadge(badge, 'scanning', null);
    }
  }

  // ─── Attach to video ──────────────────────────────────────────────────────
  function attachToVideo(video) {
    if (monitored.has(video)) return;
    if ((video.videoWidth || 0) < 20 && (video.offsetWidth || 0) < 20) return;
    monitored.add(video);

    const overlay = createOverlay(video);
    videoState.set(video, {});

    // Manual report button
    overlay.reportBtn.addEventListener('click', () => {
      const vs = videoState.get(video) || {};
      if (vs.detectionId) {
        showFeedbackPrompt(video, overlay.wrapper, vs.detectionId, 'manual_report', vs.score || 0);
      }
    });

    // Start detection loop
    const interval = setInterval(() => runDetection(video, overlay), SAMPLE_MS);

    video.addEventListener('ended', () => {
      clearInterval(interval);
      overlay.badge?.remove();
      chrome.runtime.sendMessage({ type: 'VIDEO_GONE' }).catch(() => {});
    });
  }

  // ─── DOM watching ─────────────────────────────────────────────────────────
  function scanVideos() {
    document.querySelectorAll('video').forEach(v => attachToVideo(v));
  }

  injectStyles();
  scanVideos();

  const obs = new MutationObserver(scanVideos);
  obs.observe(document.body, { childList: true, subtree: true });
  window.addEventListener('load', scanVideos);
  setTimeout(scanVideos, 2000);
  setTimeout(scanVideos, 5000);

})();
