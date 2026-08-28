/**
 * PAC Help Desk — Admin common JS
 * Live badges, theme, mobile sidebar, toasts
 *
 * Needs in HTML:
 *   #unansweredBadge, #reportsBadge, #sidebar, #sidebarOverlay
 *   #themeIcon, #themeLabel, #themeToggle (optional)
 */
(function () {
  'use strict';

  // Badge polling interval. The backend endpoint (/api/admin/realtime)
  // caches its own result for ~8s server-side and is shared by every open
  // tab/admin, so this can stay reasonably tight without hammering the
  // (small, 3-connection-on-Vercel) DB pool — see routes/api.py.
  const POLL_MS = 15000;
  const ACTIVE_MAP = {
    dashboard: '/dashboard',
    unanswered: '/unanswered',
    holidays: '/holidays',
    handbook: '/handbook',
    reports: '/reports',
    corrections: '/corrections',
    analytics: '/analytics',
    ratings: '/ratings',
    alert_log: '/alert_log',
    profile: '/profile',
  };

  function $(sel, root) {
    return (root || document).querySelector(sel);
  }
  function $$(sel, root) {
    return Array.from((root || document).querySelectorAll(sel));
  }

  function esc(s) {
    if (s == null) return '';
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  window.PACAdmin = window.PACAdmin || {};
  window.PACAdmin.esc = esc;

  function ensureToastWrap() {
    let w = $('#toastWrap');
    if (!w) {
      w = document.createElement('div');
      w.id = 'toastWrap';
      w.className = 'toast-wrap';
      document.body.appendChild(w);
    }
    return w;
  }

  function showToast(msg, type) {
    type = type || 'success';
    const icons = {
      success: 'check-circle-fill',
      danger: 'x-circle-fill',
      warning: 'exclamation-triangle-fill',
      info: 'info-circle-fill',
    };
    const wrap = ensureToastWrap();
    const el = document.createElement('div');
    el.className = 'toast-item ' + type;
    el.innerHTML =
      '<i class="bi bi-' +
      (icons[type] || icons.info) +
      '"></i><span>' +
      esc(msg) +
      '</span>';
    wrap.appendChild(el);
    setTimeout(function () {
      el.style.opacity = '0';
      el.style.transform = 'translateY(8px)';
      el.style.transition = 'all 0.25s ease';
      setTimeout(function () { el.remove(); }, 260);
    }, 3400);
  }
  window.showToast = showToast;
  window.PACAdmin.showToast = showToast;

  function updateThemeUI(theme) {
    const icon = $('#themeIcon');
    const label = $('#themeLabel');
    if (icon) icon.className = theme === 'dark' ? 'bi bi-moon-fill' : 'bi bi-sun-fill';
    if (label) label.textContent = theme === 'dark' ? 'Dark mode' : 'Light mode';
  }

  function loadTheme() {
    const t = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-bs-theme', t);
    updateThemeUI(t);
  }

  function toggleTheme() {
    const html = document.documentElement;
    const cur = html.getAttribute('data-bs-theme') || 'dark';
    const next = cur === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-bs-theme', next);
    localStorage.setItem('theme', next);
    updateThemeUI(next);
  }
  window.toggleTheme = toggleTheme;

  function toggleSidebar() {
    const sb = $('#sidebar');
    const ov = $('#sidebarOverlay');
    if (!sb) return;
    const open = sb.classList.toggle('open');
    if (ov) ov.classList.toggle('show', open);
    document.body.style.overflow = open ? 'hidden' : '';
  }

  function closeSidebar() {
    const sb = $('#sidebar');
    const ov = $('#sidebarOverlay');
    if (sb) sb.classList.remove('open');
    if (ov) ov.classList.remove('show');
    document.body.style.overflow = '';
  }
  window.toggleSidebar = toggleSidebar;
  window.closeSidebar = closeSidebar;

  const BADGE_KEY = {
    unansweredBadge: 'unanswered',
    unansweredBadge2: 'unanswered',
    reportsBadge: 'reports',
    alertsBadge: 'alerts',
    ratingsBadge: 'ratings',
  };
  const lastBadge = { unanswered: null, reports: null, alerts: null, ratings: null };

  function setBadge(id, count) {
    const el = document.getElementById(id);
    if (!el) return;
    const n = Number(count) || 0;
    const key = BADGE_KEY[id] || id;
    const prev = lastBadge[key];
    el.textContent = n > 99 ? '99+' : n;
    el.style.display = n > 0 ? 'inline-block' : 'none';
    if (prev !== null && prev !== n && n > 0) {
      el.classList.remove('pulse');
      void el.offsetWidth;
      el.classList.add('pulse');
      setTimeout(function () { el.classList.remove('pulse'); }, 600);
    }
    lastBadge[key] = n;
  }
  window.PACAdmin.setBadge = setBadge;

  async function fetchJSON(url, opts) {
    const r = await fetch(url, Object.assign({ credentials: 'same-origin' }, opts || {}));
    if (!r.ok) throw new Error(String(r.status));
    return r.json();
  }
  window.PACAdmin.fetchJSON = fetchJSON;

  // BUG FIX: this used to try /api/admin/realtime, and on ANY failure
  // (including a plain network hiccup) silently fall back to two more
  // requests AND leave the alerts/ratings badges stuck at 0 forever,
  // since there was never a working fallback for those two. The backend
  // endpoint is now a single cached, rollback-safe source of truth (see
  // routes/api.py's _get_admin_badge_counts), so this is just one
  // request — no fallback chain needed. An AbortController guards
  // against a slow response still being in flight when the next poll
  // tick fires (was possible to stack requests on a bad connection,
  // which is its own source of "laggy").
  let liveAbort = null;

  async function refreshAdminLive() {
    if (document.hidden) return;
    if (liveAbort) liveAbort.abort();
    liveAbort = new AbortController();
    try {
      const d = await fetchJSON('/api/admin/realtime', { signal: liveAbort.signal });
      const unanswered = d.unanswered || 0;
      const reports = d.reports || 0;
      const alertsNew = d.alerts_new || 0;
      const lowRatings = d.low_ratings || 0;

      setBadge('unansweredBadge', unanswered);
      setBadge('unansweredBadge2', unanswered);
      setBadge('reportsBadge', reports);
      setBadge('alertsBadge', alertsNew);
      setBadge('ratingsBadge', lowRatings);

      const su = $('#statUnanswered');
      if (su && su.textContent !== String(unanswered)) {
        su.textContent = unanswered;
        su.classList.remove('updated');
        void su.offsetWidth;
        su.classList.add('updated');
      }

      $$('.live-dot').forEach(function (dot) {
        dot.classList.toggle('degraded', !!d.degraded);
        dot.title = d.degraded ? 'Live data delayed — showing last known values' : 'Live';
      });
      $$('.degraded-banner').forEach(function (b) {
        b.classList.toggle('show', !!d.degraded);
      });

      if (typeof window.onAdminLiveUpdate === 'function') {
        window.onAdminLiveUpdate({
          unanswered: unanswered,
          reports: reports,
          alertsNew: alertsNew,
          lowRatings: lowRatings,
          degraded: !!d.degraded,
        });
      }
    } catch (e) {
      if (e.name !== 'AbortError') console.debug('admin live', e);
    }
  }
  window.refreshAdminLive = refreshAdminLive;

  // Shared helper for page-specific polling (analytics cache stats,
  // ratings badge, alert log, etc). Every page that used to roll its own
  // setInterval gets the same hidden-tab pause and jitter (spreads
  // requests out so N open tabs don't all hit the DB in the same
  // 100ms window every refresh) for free.
  function createPoller(fn, ms, opts) {
    opts = opts || {};
    let timer = null;
    let stopped = false;

    function tick() {
      if (stopped) return;
      if (!document.hidden) {
        try { fn(); } catch (e) { console.debug('poller', e); }
      }
      schedule();
    }
    function schedule() {
      if (stopped) return;
      const jitter = Math.floor(Math.random() * Math.min(1500, ms * 0.15));
      timer = setTimeout(tick, ms + jitter);
    }
    function start() {
      stopped = false;
      if (opts.immediate !== false) fn();
      schedule();
    }
    function stop() {
      stopped = true;
      if (timer) clearTimeout(timer);
      timer = null;
    }
    document.addEventListener('visibilitychange', function () {
      if (!document.hidden) fn();
    });
    start();
    return { stop: stop, start: start, runNow: fn };
  }
  window.PACAdmin.createPoller = createPoller;

  function exportData() {
    showToast('Export started…', 'info');
    window.location.href = '/api/export/faq';
  }
  window.exportData = exportData;

  /* Optional: body data-inject-sidebar="1" rebuilds sidebar */
  function injectSidebarIfNeeded() {
    const body = document.body;
    if (!body || body.getAttribute('data-inject-sidebar') !== '1') return;
    const page = body.getAttribute('data-admin-page') || '';
    const path = window.location.pathname;
    function isActive(href) {
      if (page && ACTIVE_MAP[page] === href) return true;
      return path === href || path.startsWith(href + '/');
    }
    function nav(href, icon, label, badgeId, badgeClass) {
      const active = isActive(href) ? ' active' : '';
      const badge = badgeId
        ? '<span class="nav-badge ' + badgeClass + '" id="' + badgeId + '" style="display:none">0</span>'
        : '';
      return (
        '<a href="' + href + '" class="nav-link' + active + '">' +
        '<i class="bi bi-' + icon + '"></i> ' + label + badge + '</a>'
      );
    }
    const html =
      '<div class="sidebar-brand">' +
      '<div class="brand-orb"><img src="/static/logo.png" alt="PAC" onerror="this.parentElement.innerHTML=\'🎓\'"></div>' +
      '<div><div class="brand-text">PAC Help Desk</div><div class="brand-sub">Admin Panel</div></div></div>' +
      '<nav class="sidebar-nav">' +
      '<div class="nav-section">Management</div>' +
      nav('/dashboard', 'grid-fill', 'FAQ Management') +
      nav('/unanswered', 'question-circle', 'Unanswered', 'unansweredBadge', 'badge-red') +
      nav('/holidays', 'calendar-event', 'Holidays &amp; Events') +
      nav('/handbook', 'journal-bookmark-fill', 'Student Handbook') +
      nav('/reports', 'flag', 'Reports', 'reportsBadge', 'badge-yellow') +
      nav('/corrections', 'wrench-adjustable', 'Corrections') +
      '<div class="nav-section">Analytics</div>' +
      nav('/analytics', 'graph-up', 'System Analytics') +
      nav('/ratings', 'star-half', 'Chat Ratings', 'ratingsBadge', 'badge-yellow') +
      nav('/alert_log', 'bell', 'Alert Log', 'alertsBadge', 'badge-red') +
      '<div class="nav-section">Account</div>' +
      nav('/profile', 'person-circle', 'My Profile') +
      '</nav>' +
      '<div class="sidebar-footer">' +
      '<button type="button" class="theme-toggle-sidebar" id="themeToggle" title="Toggle theme">' +
      '<span class="theme-switch-label"><i class="bi bi-moon-fill" id="themeIcon"></i><span id="themeLabel">Dark mode</span></span>' +
      '<i class="bi bi-circle-half" style="opacity:.5;font-size:.85rem"></i></button>' +
      '<button type="button" class="logout-btn" id="exportBtn"><i class="bi bi-download"></i> Export Backup</button>' +
      '<a href="/admin/logout" class="logout-btn"><i class="bi bi-box-arrow-right"></i> Log Out</a>' +
      '<a href="/" class="logout-btn"><i class="bi bi-house"></i> Back to Help Desk</a></div>';

    let sb = $('#sidebar');
    if (!sb) {
      sb = document.createElement('aside');
      sb.className = 'sidebar';
      sb.id = 'sidebar';
      document.body.insertBefore(sb, document.body.firstChild);
    }
    sb.innerHTML = html;
    if (!$('#sidebarOverlay')) {
      const ov = document.createElement('div');
      ov.className = 'sidebar-overlay';
      ov.id = 'sidebarOverlay';
      ov.onclick = closeSidebar;
      document.body.insertBefore(ov, sb);
    }
  }

  function softLoadHTML(msg) {
    return '<div class="soft-load"><span class="spinner"></span> ' + esc(msg || 'Loading…') + '</div>';
  }
  window.PACAdmin.softLoadHTML = softLoadHTML;

  let livePoller = null;

  function startPolling() {
    if (livePoller) return; // createPoller already self-pauses on hidden tabs
    livePoller = createPoller(refreshAdminLive, POLL_MS);
  }

  function bindUI() {
    const themeBtn = $('#themeToggle');
    if (themeBtn && !themeBtn._bound) {
      themeBtn.addEventListener('click', toggleTheme);
      themeBtn._bound = true;
    }
    const exportBtn = $('#exportBtn');
    if (exportBtn && !exportBtn._bound) {
      exportBtn.addEventListener('click', exportData);
      exportBtn._bound = true;
    }
    $$('.sidebar .nav-link').forEach(function (link) {
      link.addEventListener('click', function () {
        if (window.innerWidth <= 900) closeSidebar();
      });
    });
    const toggle = $('#sidebarToggle');
    if (toggle && !toggle._bound) {
      toggle.addEventListener('click', toggleSidebar);
      toggle._bound = true;
    }
    const ov = $('#sidebarOverlay');
    if (ov && !ov._bound) {
      ov.addEventListener('click', closeSidebar);
      ov._bound = true;
    }
  }

  function boot() {
    injectSidebarIfNeeded();
    loadTheme();
    bindUI();
    startPolling();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();