// DeepGuard — Background Service Worker
// Manages global detection state, toolbar icon, badge, and detection history.

const STATE = {
  SAFE: 'safe',
  WARNING: 'warning',
  DANGER: 'danger',
  IDLE: 'idle'
};

const ICON_PATHS = {
  [STATE.SAFE]:    { icon: 'icons/icon-green', title: 'DeepGuard — All clear' },
  [STATE.WARNING]: { icon: 'icons/icon-amber', title: 'DeepGuard — Suspicious content' },
  [STATE.DANGER]:  { icon: 'icons/icon-red',   title: 'DeepGuard — Deepfake detected!' },
  [STATE.IDLE]:    { icon: 'icons/icon-gray',   title: 'DeepGuard — No video detected' }
};

// Per-tab detection state
const tabStates = new Map();
const detectionHistory = [];

// ----- Icon & badge helpers -----

function setTabIcon(tabId, state) {
  const cfg = ICON_PATHS[state] || ICON_PATHS[STATE.IDLE];

  chrome.action.setIcon({
    tabId,
    path: {
      16:  `${cfg.icon}-16.png`,
      32:  `${cfg.icon}-32.png`,
      48:  `${cfg.icon}-48.png`,
      128: `${cfg.icon}-128.png`
    }
  });

  chrome.action.setTitle({ tabId, title: cfg.title });

  // Badge text
  const badgeMap = {
    [STATE.SAFE]:    { text: '',    color: '#27A96C' },
    [STATE.WARNING]: { text: '!',   color: '#E89E1A' },
    [STATE.DANGER]:  { text: '!!!', color: '#D63939' },
    [STATE.IDLE]:    { text: '',    color: '#888888' }
  };
  const badge = badgeMap[state];
  chrome.action.setBadgeText({ tabId, text: badge.text });
  chrome.action.setBadgeBackgroundColor({ tabId, color: badge.color });
}

// ----- Message handler (from content.js) -----

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  const tabId = sender.tab?.id;
  if (!tabId) return;

  if (msg.type === 'DETECTION_UPDATE') {
    const { score, state, signals, url } = msg;

    const prev = tabStates.get(tabId);
    tabStates.set(tabId, { score, state, signals, url, ts: Date.now() });

    setTabIcon(tabId, state);

    // Log to history (keep last 100 events)
    if (state === STATE.WARNING || state === STATE.DANGER) {
      detectionHistory.unshift({ tabId, score, state, signals, url, ts: Date.now() });
      if (detectionHistory.length > 100) detectionHistory.pop();
    }

    // Notify popup if open
    chrome.runtime.sendMessage({ type: 'STATE_UPDATE', tabId, score, state, signals }).catch(() => {});
    sendResponse({ ok: true });
  }

  if (msg.type === 'VIDEO_GONE') {
    tabStates.set(tabId, { state: STATE.IDLE });
    setTabIcon(tabId, STATE.IDLE);
    sendResponse({ ok: true });
  }

  if (msg.type === 'GET_STATE') {
    sendResponse(tabStates.get(tabId) || { state: STATE.IDLE });
  }

  if (msg.type === 'GET_HISTORY') {
    sendResponse(detectionHistory.slice(0, 20));
  }

  return true; // keep channel open for async
});

// ----- Tab lifecycle cleanup -----

chrome.tabs.onRemoved.addListener((tabId) => {
  tabStates.delete(tabId);
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (changeInfo.status === 'loading') {
    tabStates.set(tabId, { state: STATE.IDLE });
    setTabIcon(tabId, STATE.IDLE);
  }
});
