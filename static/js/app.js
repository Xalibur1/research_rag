const API_BASE = '/api';

/* =============================================
   INJECT TRANSITION & ALIGNMENT CSS
   ============================================= */
(function injectStyles() {
    const style = document.createElement('style');
    style.textContent = `
        /* Page entrance */
        @keyframes pageIn {
            from { opacity: 0; transform: translateY(14px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes pageFadeIn {
            from { opacity: 0; }
            to   { opacity: 1; }
        }
        main, .page-content {
            animation: pageIn 0.32s cubic-bezier(0.22, 1, 0.36, 1) both;
        }
        /* Sidebar brand */
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 24px;
        }
        .sidebar-brand-icon {
            width: 40px; height: 40px;
            border-radius: 12px;
            background: linear-gradient(135deg, #7339c8 0%, #8c55e3 100%);
            display: flex; align-items: center; justify-content: center;
            color: white;
            box-shadow: 0 4px 12px rgba(115, 57, 200, 0.25);
            flex-shrink: 0;
        }
        /* Nav active state */
        [data-nav].active {
            background: white;
            color: #7339c8;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }
        /* Page exit */
        body.page-exiting main,
        body.page-exiting .page-content {
            animation: none;
            transition: opacity 0.18s ease, transform 0.18s ease;
            opacity: 0;
            transform: translateY(-8px);
        }
        /* Stat card hover lift */
        section > div {
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        section > div:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(115,57,200,0.08);
        }
        /* Toast slide-in */
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(24px); }
            to   { opacity: 1; transform: translateX(0); }
        }
        /* Upload spinner */
        @keyframes spin { to { transform: rotate(360deg); } }
        .animate-spin { animation: spin 0.7s linear infinite; }
    `;
    document.head.appendChild(style);
})();

/* =============================================
   NAVIGATION — update active link per page
   ============================================= */
function setActiveNav() {
    const path = window.location.pathname.replace(/\/$/, '') || '/index.html';
    const navMap = {
        '/index.html':    '[data-nav="dashboard"]',
        '/':              '[data-nav="dashboard"]',
        '/analysis.html': '[data-nav="analysis"]',
        '/compare.html':  '[data-nav="compare"]',
        '/io.html':       '[data-nav="io"]',
    };
    const selector = navMap[path];
    if (selector) {
        document.querySelectorAll('[data-nav]').forEach(el => {
            el.classList.remove('bg-white', 'text-primary', 'shadow-sm');
            el.classList.add('text-slate-600', 'hover:bg-[#e3e0f5]');
        });
        const active = document.querySelector(selector);
        if (active) {
            active.classList.add('bg-white', 'text-primary', 'shadow-sm');
            active.classList.remove('text-slate-600', 'hover:bg-[#e3e0f5]');
        }
    }
}

/* =============================================
   TOAST — inline styles so no external CSS needed
   ============================================= */
function showToast(message, isError = false) {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.style.cssText = 'position:fixed;bottom:32px;right:32px;z-index:9999;display:flex;flex-direction:column;gap:8px;';
        document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.style.cssText = `
        padding: 12px 20px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 600;
        color: white;
        background: ${isError ? '#ba1a1a' : '#7339c8'};
        box-shadow: 0 8px 32px rgba(0,0,0,0.18);
        display: flex; align-items: center; gap: 10px;
        min-width: 260px; max-width: 400px;
        animation: slideIn 0.25s ease;
    `;
    toast.innerHTML = `<span style="font-size:18px;">${isError ? '⚠️' : '✅'}</span> ${message}`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(20px)';
        setTimeout(() => toast.remove(), 300);
    }, 4500);
}

/* =============================================
   DASHBOARD — load real stats from backend
   ============================================= */
async function loadDashboardStats() {
    try {
        const response = await fetch(`${API_BASE}/dashboard/stats`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        const m = data.metrics;
        document.getElementById('stat-total-papers').textContent  = m.total_papers ?? '—';
        document.getElementById('stat-inputs-mapped').textContent = m.graph_nodes ?? '—';
        document.getElementById('stat-flagged').textContent       = m.limitations_found ?? '—';

        const list = document.getElementById('recent-verifications-list');
        if (!list) return;
        list.innerHTML = '';

        if (!data.recent_verifications || data.recent_verifications.length === 0) {
            list.innerHTML = `
                <div class="p-6 text-center text-on-surface-variant text-sm font-body">
                    <span class="material-symbols-outlined text-4xl block mb-2 opacity-30">inbox</span>
                    No papers processed yet. Upload a PDF using the button above.
                </div>`;
            return;
        }

        data.recent_verifications.forEach(paper => {
            const item = document.createElement('div');
            item.innerHTML = `
                <div class="group flex items-center gap-4 p-4 bg-white hover:bg-surface-container-low transition-colors rounded-xl cursor-pointer border border-outline-variant/10"
                     onclick="window.location.href='/analysis.html?id=${encodeURIComponent(paper.id)}'">
                    <div class="w-10 h-10 flex-shrink-0 bg-surface-container-high rounded-lg flex items-center justify-center text-primary">
                        <span class="material-symbols-outlined text-lg">description</span>
                    </div>
                    <div class="flex-1 min-w-0">
                        <h4 class="text-sm font-bold truncate font-headline group-hover:text-primary transition-colors">${paper.title || paper.id}</h4>
                        <p class="text-xs text-on-surface-variant font-body mt-0.5">ID: ${paper.id}&nbsp;&bull;&nbsp;Inputs: ${paper.inputs}</p>
                    </div>
                    <div class="flex-shrink-0 px-3 py-1 rounded-full bg-emerald-50 text-emerald-700 font-label text-[10px] font-black uppercase tracking-wider flex items-center gap-1">
                        <span class="material-symbols-outlined text-xs" style="font-variation-settings:'FILL' 1;">check_circle</span>
                        ${paper.status}
                    </div>
                </div>`;
            list.appendChild(item);
        });
    } catch (err) {
        console.error('Dashboard stats error:', err);
        const list = document.getElementById('recent-verifications-list');
        if (list) list.innerHTML = `<p class="text-error text-sm p-4">Failed to load dashboard data. Is the server running?</p>`;
    }
}

/* =============================================
   UPLOAD WIDGET
   ============================================= */
function initUploadWidget() {
    const btn   = document.getElementById('btn-upload');
    const input = document.getElementById('pdf-upload');
    if (!btn || !input) return;

    btn.addEventListener('click', () => input.click());

    input.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            showToast('Only PDF files are supported.', true);
            return;
        }

        const originalHTML = btn.innerHTML;
        btn.innerHTML = `
            <svg class="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
            </svg>
            <span>Extracting…</span>`;
        btn.disabled = true;

        const form = new FormData();
        form.append('file', file);

        try {
            const res  = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form });
            const data = await res.json();
            if (data.error) {
                showToast('Upload failed: ' + data.error, true);
            } else {
                showToast(`"${data.paper_id}" successfully processed!`);
                loadDashboardStats();
            }
        } catch (err) {
            showToast('Network error — check server connection.', true);
            console.error(err);
        } finally {
            btn.innerHTML  = originalHTML;
            btn.disabled   = false;
            input.value    = '';
        }
    });
}

/* =============================================
   PAPER ANALYSIS PAGE
   ============================================= */
async function loadPaperAnalysis() {
    const paperId = new URLSearchParams(window.location.search).get('id');

    const titleEl    = document.getElementById('paper-title');
    const abstractEl = document.getElementById('paper-abstract');
    const metaEl     = document.getElementById('paper-metadata');
    const entEl      = document.getElementById('paper-entities');
    const resEl      = document.getElementById('paper-results');

    if (!paperId) {
        if (titleEl) titleEl.textContent = 'No paper selected — go back to Dashboard.';
        return;
    }

    if (titleEl) titleEl.textContent = 'Loading…';

    try {
        const response = await fetch(`${API_BASE}/papers/${encodeURIComponent(paperId)}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const d = await response.json();

        if (d.error) {
            if (titleEl) titleEl.textContent = 'Error: ' + d.error;
            return;
        }

        if (titleEl)    titleEl.textContent    = d.title || paperId;
        if (abstractEl) abstractEl.textContent = d.abstract || 'No abstract extracted for this paper.';

        // Metadata row
        if (metaEl) {
            const authors = (d.authors || []).join(', ') || '—';
            const year    = d.year  || '—';
            const venue   = d.venue || '—';
            metaEl.innerHTML = `
                <div>
                    <p class="text-[10px] font-label font-bold text-slate-400 uppercase tracking-widest mb-1">Authors</p>
                    <p class="text-sm font-bold text-on-background">${authors}</p>
                </div>
                <div>
                    <p class="text-[10px] font-label font-bold text-slate-400 uppercase tracking-widest mb-1">Venue / Year</p>
                    <p class="text-sm font-bold text-on-background">${venue} ${year}</p>
                </div>
                <div>
                    <p class="text-[10px] font-label font-bold text-slate-400 uppercase tracking-widest mb-1">Inputs Mapped</p>
                    <p class="text-sm font-bold text-on-background">${d.inputs ?? '—'}</p>
                </div>`;
        }

        // Extracted entities (glossary terms)
        if (entEl) {
            const terms = (d.entities || []).filter(Boolean);
            if (terms.length > 0) {
                entEl.innerHTML = terms.map(t => `
                    <div class="bg-white p-4 rounded-lg flex justify-between items-center hover:translate-x-1 transition-transform border border-outline-variant/10">
                        <div>
                            <p class="text-xs font-bold text-slate-800">${t}</p>
                            <p class="text-[10px] text-slate-500 font-label">Glossary Entity</p>
                        </div>
                        <span class="material-symbols-outlined text-emerald-400" style="font-variation-settings:'FILL' 1;">verified</span>
                    </div>`).join('');
            } else {
                entEl.innerHTML = `<p class="text-sm text-slate-400 italic">No glossary entities extracted for this paper.</p>`;
            }
        }

        // Key results
        if (resEl) {
            const method = d.method || {};
            const rows   = [];

            if (method.core_idea)            rows.push({ k: 'Core Idea',       v: method.core_idea });
            if (method.name)                 rows.push({ k: 'Method Name',     v: method.name });
            if ((method.algorithm_steps||[]).length)
                rows.push({ k: 'Algorithm Steps', v: method.algorithm_steps.join(' → ') });
            if ((method.assumptions||[]).length)
                rows.push({ k: 'Assumptions',    v: method.assumptions.join('; ') });

            const evaluation = d.evaluation || {};
            if ((evaluation.datasets||[]).length)
                rows.push({ k: 'Datasets',    v: evaluation.datasets.join(', ') });
            if ((evaluation.metrics||[]).length)
                rows.push({ k: 'Metrics',     v: evaluation.metrics.join(', ') });
            (evaluation.results||[]).slice(0,3).forEach(r => {
                if (r.metric) rows.push({ k: r.metric, v: `${r.value || '—'} (${r.dataset || 'unknown'})` });
            });

            if (rows.length > 0) {
                resEl.innerHTML = rows.map(r => `
                    <div class="bg-white p-4 rounded-lg border-l-4 border-primary flex justify-between items-start hover:translate-x-1 transition-transform mb-3">
                        <div class="flex-1">
                            <p class="text-xs font-bold text-slate-800">${r.k}</p>
                            <p class="text-[11px] text-slate-600 mt-1 leading-relaxed">${r.v}</p>
                        </div>
                    </div>`).join('');
            } else {
                resEl.innerHTML = `<p class="text-sm text-slate-400 italic">No structured results extracted.</p>`;
            }
        }

    } catch (err) {
        console.error('Analysis load error:', err);
        if (titleEl) titleEl.textContent = 'Failed to load paper data.';
    }
}

/* =============================================
   COMPARE PAGE — full implementation
   ============================================= */
function initComparison() {
    // ── State ──────────────────────────────────
    let allPapers       = [];    // [{id, title}] from API
    let selectedIds     = new Set(); // currently selected paper IDs
    let collections     = [];    // [{id, name, paper_ids[]}]
    let modalSelection  = new Set(); // temp selection in picker modal

    const ACCENT_COLORS = ['#7339c8','#a83900','#7d4887','#3a5fa8','#2f8a4d','#8a6b2f','#ba1a1a','#067474'];

    // ── DOM refs ──────────────────────────────
    const modal         = document.getElementById('paper-picker-modal');
    const modalClose    = document.getElementById('modal-close');
    const modalSearch   = document.getElementById('modal-search');
    const modalList     = document.getElementById('modal-paper-list');
    const modalCount    = document.getElementById('modal-count');
    const modalConfirm  = document.getElementById('modal-confirm');

    const saveModal     = document.getElementById('collection-save-modal');
    const saveCancel    = document.getElementById('collection-save-cancel');
    const saveConfirm   = document.getElementById('collection-save-confirm');
    const saveNameInput = document.getElementById('collection-name-input');

    const chipsEl       = document.getElementById('active-paper-chips');
    const selectedList  = document.getElementById('selected-papers-list');
    const selCount      = document.getElementById('selection-count');
    const noHint        = document.getElementById('no-selection-hint');
    const scopeBanner   = document.getElementById('scope-banner');
    const scopeText     = document.getElementById('scope-text');
    const outputScopeTag= document.getElementById('output-scope-tag');
    const collectionsList = document.getElementById('collections-list');
    const sidebarCols   = document.getElementById('sidebar-collections');

    const form          = document.getElementById('chat-form');
    const chatInput     = document.getElementById('chat-input');
    const chatOutput    = document.getElementById('chat-output');
    const chatSubmitBtn = document.getElementById('chat-submit');
    const chatSubmitTxt = document.getElementById('chat-submit-text');
    const reasoningEl   = document.getElementById('reasoning-output');
    const reasoningSec  = document.getElementById('reasoning-section');

    // ── Helpers ───────────────────────────────
    function openModal()  { modal.classList.add('open'); }
    function closeModal() { modal.classList.remove('open'); }

    function renderSelectedPapersUI() {
        // Chips in header
        if (chipsEl) {
            chipsEl.innerHTML = [...selectedIds].map((id, i) => {
                const paper = allPapers.find(p => p.id === id);
                const title = paper ? paper.title : id;
                const color = ACCENT_COLORS[i % ACCENT_COLORS.length];
                return `<span class="paper-chip" style="background:${color}20;color:${color}" data-id="${id}" title="${title}">
                    <span style="max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${title}</span>
                    <span class="remove-chip" data-remove="${id}">✕</span>
                </span>`;
            }).join('');
            chipsEl.querySelectorAll('.remove-chip').forEach(btn => {
                btn.addEventListener('click', e => { e.stopPropagation(); removePaper(btn.dataset.remove); });
            });
        }

        // Left panel list
        if (selectedList) {
            const items = [...selectedIds].map((id, i) => {
                const paper = allPapers.find(p => p.id === id);
                const title = paper ? paper.title : id;
                const color = ACCENT_COLORS[i % ACCENT_COLORS.length];
                return `<div class="p-3 bg-surface-container-low rounded-lg relative overflow-hidden flex items-start gap-2">
                    <div style="width:3px;min-height:40px;border-radius:2px;background:${color};flex-shrink:0;margin-top:2px"></div>
                    <div class="flex-1 min-w-0">
                        <p class="text-xs font-bold line-clamp-2 text-on-surface">${title}</p>
                        <p class="text-[10px] text-slate-400 mt-0.5 font-mono">${id}</p>
                    </div>
                    <button class="remove-selected shrink-0 opacity-40 hover:opacity-100 transition-opacity" data-remove="${id}">
                        <span class="material-symbols-outlined text-sm text-red-400">close</span>
                    </button>
                </div>`;
            }).join('');
            selectedList.innerHTML = items || '';
            if (noHint) noHint.style.display = selectedIds.size === 0 ? 'block' : 'none';
            selectedList.querySelectorAll('.remove-selected').forEach(btn => {
                btn.addEventListener('click', () => removePaper(btn.dataset.remove));
            });
        }

        // Counter badge
        const n = selectedIds.size;
        if (selCount) selCount.textContent = `${n} paper${n !== 1 ? 's' : ''}`;

        // Scope banner
        if (scopeBanner) {
            if (n > 0) {
                scopeBanner.classList.remove('hidden');
                scopeBanner.classList.add('flex');
                const names = [...selectedIds].map(id => {
                    const p = allPapers.find(x => x.id === id);
                    return p ? p.title.split(':')[0] : id;
                }).join(', ');
                if (scopeText) scopeText.textContent = `Scoped to: ${names}`;
                if (outputScopeTag) outputScopeTag.textContent = `${n} Paper${n !== 1 ? 's' : ''}`;
            } else {
                scopeBanner.classList.add('hidden');
                scopeBanner.classList.remove('flex');
                if (outputScopeTag) outputScopeTag.textContent = 'All Papers';
            }
        }
    }

    function removePaper(id) {
        selectedIds.delete(id);
        renderSelectedPapersUI();
    }

    function addPapers(ids) {
        ids.forEach(id => selectedIds.add(id));
        renderSelectedPapersUI();
    }

    // ── Modal logic ───────────────────────────
    function refreshModalList(searchTerm = '') {
        const term = searchTerm.toLowerCase();
        const filtered = allPapers.filter(p =>
            p.title.toLowerCase().includes(term) || p.id.toLowerCase().includes(term)
        );
        if (filtered.length === 0) {
            modalList.innerHTML = `<p class="text-sm text-slate-400 italic text-center py-4">No papers found.</p>`;
            return;
        }
        modalList.innerHTML = filtered.map(p => {
            const checked = modalSelection.has(p.id) ? 'checked' : '';
            return `<label class="flex items-start gap-3 p-3 rounded-xl hover:bg-surface-container cursor-pointer transition-colors">
                <input type="checkbox" ${checked} data-pid="${p.id}" class="mt-0.5 w-4 h-4 rounded accent-purple-600 flex-shrink-0" />
                <div class="min-w-0">
                    <p class="text-sm font-semibold text-on-surface leading-snug">${p.title}</p>
                    <p class="text-[10px] text-slate-400 font-mono mt-0.5">${p.id}</p>
                </div>
            </label>`;
        }).join('');
        modalList.querySelectorAll('input[type=checkbox]').forEach(cb => {
            cb.addEventListener('change', () => {
                if (cb.checked) modalSelection.add(cb.dataset.pid);
                else modalSelection.delete(cb.dataset.pid);
                if (modalCount) modalCount.textContent = `${modalSelection.size} selected`;
            });
        });
        if (modalCount) modalCount.textContent = `${modalSelection.size} selected`;
    }

    document.querySelectorAll('#btn-add-paper, #btn-add-paper-2').forEach(btn => {
        btn && btn.addEventListener('click', () => {
            modalSelection = new Set(selectedIds);
            refreshModalList();
            openModal();
        });
    });
    if (modalClose) modalClose.addEventListener('click', closeModal);
    modal && modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });
    if (modalSearch) modalSearch.addEventListener('input', () => refreshModalList(modalSearch.value));
    if (modalConfirm) {
        modalConfirm.addEventListener('click', () => {
            addPapers([...modalSelection]);
            closeModal();
        });
    }

    // ── Scope banner clear ────────────────────
    const scopeClear = document.getElementById('scope-clear');
    if (scopeClear) scopeClear.addEventListener('click', () => {
        selectedIds.clear();
        renderSelectedPapersUI();
    });

    // ── Clear all ─────────────────────────────
    const clearAll = document.getElementById('btn-clear-all');
    if (clearAll) clearAll.addEventListener('click', () => {
        selectedIds.clear();
        renderSelectedPapersUI();
    });

    // ── Collections logic ─────────────────────
    async function loadCollections() {
        try {
            const res = await fetch(`${API_BASE}/collections`);
            collections = await res.json();
            renderCollections();
        } catch (e) { console.error('Collections load failed', e); }
    }

    function renderCollections() {
        const render = (container) => {
            if (!container) return;
            if (collections.length === 0) {
                container.innerHTML = `<p class="text-[11px] text-slate-400 italic px-4 py-2">No collections yet</p>`;
                return;
            }
            container.innerHTML = collections.map(col => `
                <div class="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-surface-container-low group transition-colors cursor-pointer collection-item" data-col-id="${col.id}" data-col-paper-ids='${JSON.stringify(col.paper_ids)}'>
                    <span class="material-symbols-outlined text-sm text-primary" style="font-variation-settings:'FILL' 1">folder</span>
                    <div class="flex-1 min-w-0">
                        <p class="text-xs font-semibold text-on-surface truncate">${col.name}</p>
                        <p class="text-[10px] text-slate-400">${col.paper_ids.length} paper${col.paper_ids.length !== 1 ? 's' : ''}</p>
                    </div>
                    <button class="delete-col opacity-0 group-hover:opacity-60 hover:!opacity-100 transition-opacity shrink-0" data-del-col="${col.id}">
                        <span class="material-symbols-outlined text-sm text-red-400">delete</span>
                    </button>
                </div>`
            ).join('');

            container.querySelectorAll('.collection-item').forEach(item => {
                item.addEventListener('click', e => {
                    if (e.target.closest('.delete-col')) return;
                    const ids = JSON.parse(item.dataset.colPaperIds || '[]');
                    selectedIds = new Set(ids);
                    renderSelectedPapersUI();
                    showToast(`Loaded collection "${collections.find(c=>c.id===item.dataset.colId)?.name}"`);
                });
            });
            container.querySelectorAll('.delete-col').forEach(btn => {
                btn.addEventListener('click', async e => {
                    e.stopPropagation();
                    const colId = btn.dataset.delCol;
                    await fetch(`${API_BASE}/collections/${colId}`, { method: 'DELETE' });
                    await loadCollections();
                });
            });
        };
        render(collectionsList);
        render(sidebarCols);
    }

    // Save Collection modal
    const btnSave = document.getElementById('btn-save-collection');
    const btnNewCol = document.getElementById('btn-new-collection');
    const btnNewColSidebar = document.getElementById('btn-new-collection-sidebar');

    function openSaveModal() {
        if (saveNameInput) saveNameInput.value = '';
        if (saveModal) { saveModal.classList.remove('hidden'); saveModal.classList.add('flex'); }
    }
    [btnSave, btnNewCol, btnNewColSidebar].forEach(b => b && b.addEventListener('click', openSaveModal));
    if (saveCancel) saveCancel.addEventListener('click', () => { saveModal.classList.add('hidden'); saveModal.classList.remove('flex'); });
    if (saveConfirm) {
        saveConfirm.addEventListener('click', async () => {
            const name = (saveNameInput?.value || '').trim();
            if (!name) { showToast('Please enter a collection name.', true); return; }
            const paperIds = [...selectedIds];
            await fetch(`${API_BASE}/collections`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, paper_ids: paperIds })
            });
            saveModal.classList.add('hidden');
            saveModal.classList.remove('flex');
            await loadCollections();
            showToast(`Collection "${name}" saved!`);
        });
    }

    // ── Chat form ─────────────────────────────
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const query = chatInput?.value.trim();
            if (!query) return;

            if (chatSubmitBtn) chatSubmitBtn.disabled = true;
            if (chatSubmitTxt) chatSubmitTxt.textContent = 'Thinking…';

            chatOutput.innerHTML = `
                <div class="flex items-center gap-3 text-primary font-label text-sm p-4">
                    <svg class="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
                    </svg>
                    Querying Hybrid RAG engine${selectedIds.size > 0 ? ` (scoped to ${selectedIds.size} paper${selectedIds.size !== 1 ? 's' : ''})` : ''}…
                </div>`;
            if (reasoningSec) reasoningSec.classList.add('hidden');

            try {
                const body = { query };
                if (selectedIds.size > 0) body.paper_ids = [...selectedIds];

                const res = await fetch(`${API_BASE}/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();

                if (data.error) {
                    chatOutput.innerHTML = `<p class="text-error text-sm p-4">${data.error}</p>`;
                    return;
                }

                const answer = data.narrative_answer || data.answer || 'No answer returned.';
                let html = `<p class="text-sm leading-relaxed text-on-surface">${answer.replace(/\n/g, '<br>')}</p>`;

                if (data.structured_summary?.key_findings?.length) {
                    html += `<div class="mt-5 p-4 bg-surface-container-low rounded-xl">
                        <p class="text-[10px] font-label font-bold uppercase tracking-widest text-on-surface-variant mb-2">Key Findings</p>
                        <ul class="space-y-1">${data.structured_summary.key_findings.map(f =>
                            `<li class="text-xs text-on-surface flex gap-2"><span class="text-primary">▸</span>${f}</li>`
                        ).join('')}</ul></div>`;
                }

                if (data.structured_summary?.comparisons?.length) {
                    html += `<div class="mt-3 p-4 bg-surface-container-low rounded-xl">
                        <p class="text-[10px] font-label font-bold uppercase tracking-widest text-on-surface-variant mb-2">Comparisons</p>
                        <ul class="space-y-1">${data.structured_summary.comparisons.map(c =>
                            `<li class="text-xs text-on-surface flex gap-2"><span class="text-secondary">↔</span>${c}</li>`
                        ).join('')}</ul></div>`;
                }

                if (data.evidence?.length) {
                    html += `<div class="mt-3 p-4 bg-surface-container-low rounded-xl">
                        <p class="text-[10px] font-label font-bold uppercase tracking-widest text-on-surface-variant mb-2">Evidence</p>
                        <ul class="space-y-2">${data.evidence.map(ev =>
                            `<li class="text-xs border-l-2 border-primary pl-3 py-1">
                                <strong class="text-on-background">${ev.paper_id}</strong>
                                <span class="text-on-surface-variant"> (p.${(ev.page_numbers||[]).join(',')}):</span>
                                ${ev.snippet || ''}
                            </li>`
                        ).join('')}</ul></div>`;
                }

                chatOutput.innerHTML = html;

                if (data.reasoning_summary && reasoningSec && reasoningEl) {
                    reasoningEl.textContent = data.reasoning_summary;
                    reasoningSec.classList.remove('hidden');
                }

                // Save for verify
                window.lastQuery = query;
                window.lastAnswer = answer;
                const verifyOut = document.getElementById('verify-output');
                if (verifyOut) {
                    verifyOut.classList.add('hidden');
                    verifyOut.innerHTML = '';
                }

            } catch (err) {
                chatOutput.innerHTML = `<p class="text-error text-sm p-4">Error: ${err.message}</p>`;
                console.error(err);
            } finally {
                if (chatSubmitBtn) chatSubmitBtn.disabled = false;
                if (chatSubmitTxt) chatSubmitTxt.textContent = 'Analyse';
            }
        });
    }

    // ── Verify Grounding ──────────────────────
    const btnVerify = document.getElementById('btn-verify-grounding');
    const verifyOut = document.getElementById('verify-output');
    if (btnVerify && verifyOut) {
        btnVerify.addEventListener('click', async () => {
            if (!window.lastQuery || !window.lastAnswer) {
                showToast("Perform an analysis first to verify it.", true);
                return;
            }

            verifyOut.classList.remove('hidden');
            verifyOut.innerHTML = `
                <div class="flex items-center gap-3 text-secondary font-label text-sm">
                    <svg class="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
                    </svg>
                    Checking hallucination against source chunks...
                </div>
            `;

            try {
                const body = { 
                    query: window.lastQuery, 
                    answer: window.lastAnswer 
                };
                if (selectedIds.size > 0) body.paper_ids = [...selectedIds];

                const res = await fetch(`${API_BASE}/verify`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();

                if (data.error) throw new Error(data.error);

                let scoreColor = "text-secondary";
                if (data.grounding_score > 0.8) scoreColor = "text-emerald-500";
                if (data.grounding_score < 0.5) scoreColor = "text-red-500";

                let html = `<div class="flex items-center justify-between mb-3 border-b border-outline-variant/10 pb-2">
                                <h5 class="text-xs font-bold uppercase tracking-widest text-on-surface-variant">Grounding Report</h5>
                                <span class="font-bold ${scoreColor} rounded px-2 py-1 bg-surface-container">Score: ${(data.grounding_score * 100).toFixed(0)}%</span>
                            </div>`;

                if (data.hallucinated_claims?.length) {
                    html += `<p class="text-[10px] uppercase font-bold text-red-500 tracking-wider mb-1 mt-2">Hallucinated / Unsupported</p>
                             <ul class="space-y-1 mb-3">${data.hallucinated_claims.map(c => `<li class="text-xs text-red-600 flex items-start gap-2"><span class="mt-0.5">⚠️</span> <span>${c}</span></li>`).join('')}</ul>`;
                }

                if (data.grounded_claims?.length) {
                    html += `<p class="text-[10px] uppercase font-bold text-emerald-500 tracking-wider mb-1 mt-2">Grounded Claims</p>
                             <ul class="space-y-1">${data.grounded_claims.map(c => `<li class="text-xs text-emerald-700 flex items-start gap-2"><span class="mt-0.5">✓</span> <span>${c}</span></li>`).join('')}</ul>`;
                }

                verifyOut.innerHTML = html;

            } catch (err) {
                verifyOut.innerHTML = `<p class="text-error text-xs">Verification failed: ${err.message}</p>`;
            }
        });
    }



    // ── Bootstrap ─────────────────────────────
    (async () => {
        try {
            const res = await fetch(`${API_BASE}/papers`);
            allPapers = await res.json();
            // normalise: backend returns [{id, title}]
            allPapers = allPapers.map(p => ({
                id: p.id || p.paper_id,
                title: typeof p.title === 'string' ? p.title : (p.title?.[0] || p.id)
            }));
        } catch (e) { console.error('Failed to load papers', e); }
        await loadCollections();
        renderSelectedPapersUI();
    })();
}

/* =============================================
   BOOT
   ============================================= */
document.addEventListener('DOMContentLoaded', () => {
    // Fix all nav hrefs using data-nav attributes in every page
    const navLinks = {
        'dashboard': '/index.html',
        'analysis':  '/analysis.html',
        'compare':   '/compare.html',
        'io':        '/io.html',
    };

    document.querySelectorAll('[data-nav]').forEach(el => {
        const key = el.getAttribute('data-nav');
        if (navLinks[key]) el.href = navLinks[key];
    });

    setActiveNav();

    // ── Page exit transitions ──────────────────
    document.querySelectorAll('[data-nav]').forEach(link => {
        link.addEventListener('click', (e) => {
            const dest = link.href;
            if (!dest || dest === window.location.href) return;

            e.preventDefault();
            const mainEl = document.querySelector('main') || document.querySelector('.page-content');

            if (mainEl) {
                mainEl.style.transition = 'opacity 0.18s ease, transform 0.18s ease';
                mainEl.style.opacity    = '0';
                mainEl.style.transform  = 'translateY(-8px)';
                setTimeout(() => { window.location.href = dest; }, 190);
            } else {
                window.location.href = dest;
            }
        });
    });

    // ── Page init ─────────────────────────────
    if (document.getElementById('dashboard-page')) {
        loadDashboardStats();
        initUploadWidget();
    }
    if (document.getElementById('analysis-page')) {
        loadPaperAnalysis();
    }
    if (document.getElementById('compare-page')) {
        initComparison();
    }
});
