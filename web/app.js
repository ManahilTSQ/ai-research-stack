/* ==========================================================================
   ACADEMIC AI RESEARCH STACK — CLIENT SPA ROUTER & CONTROLLER
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    // API Endpoint Base — empty string means all /api/* calls go to the same origin
    const API_BASE = "";

    // App State — track running citation job and current source chunks
    let activeCitationRunId = null;
    let citationPollInterval = null;
    let activeChatSources = [];
    // Persisted conversational memory used for server-side multi-turn RAG context.
    let chatHistoryTurns = []; // [{role: "user"|"assistant", content: "..."}]
    // Persisted per-message source payload so restored "Show retrieved chunks" keeps working after refresh.
    let sourcesByBubbleId = {}; // { [bubbleId]: Array<sourceChunk> }

    // Paginated search state
    let currentSearchQuery = "";
    let currentSearchLimit = 10;
    let currentSearchOffset = 0;
    let allFetchedPapers = [];    // All papers fetched (grows with Load More)
    let hasMorePapers = true;     // Whether more pages exist

    // Ingested papers cache — populated by fetchLocalPDFs(), used for duplicate detection
    let ingestedPapers = [];      // {title, doi, authors, year, paper_id} for 'success' entries
    let editingPromptName = null; // When set, prompt editor is updating an existing template
    let localManifestFiles = [];  // Raw list of all manifest files (includes pending, failed, success)
    const CHAT_STORAGE_KEY = "cite_rag_chat_history_v2";

    // Initialize SPA tabs routing
    initTabs();

    // Start health check polling (every 15 seconds)
    checkHealth();
    setInterval(checkHealth, 15000);

    // Initial data loading on page load
    fetchLocalPDFs();
    fetchPrompts();
    fetchReports();
    restoreChatHistory();

    // Register all event listeners for forms and buttons
    document.getElementById("paper-search-form").addEventListener("submit", handlePaperSearch);
    document.getElementById("rag-query-form").addEventListener("submit", handleRAGQuery);
    document.getElementById("btn-sync-pdfs").addEventListener("click", fetchLocalPDFs);
    document.getElementById("btn-scan-pending").addEventListener("click", handleScanPending);
    document.getElementById("citation-analysis-form").addEventListener("submit", handleCitationAnalysis);
    document.getElementById("close-sources-btn").addEventListener("click", () => {
        document.getElementById("retrieved-sources-panel").classList.add("hidden");
    });

    // Abstract modal closing logic
    const abstractModal = document.getElementById("abstract-modal");
    if (abstractModal) {
        document.getElementById("btn-close-modal").addEventListener("click", () => {
            abstractModal.classList.add("hidden");
        });
        abstractModal.addEventListener("click", (e) => {
            if (e.target === abstractModal) {
                abstractModal.classList.add("hidden");
            }
        });
    }

    // Sync the RAG context-limit slider label with the slider value in real time
    const limitSlider = document.getElementById("rag-limit-slider");
    const limitLabel = document.getElementById("lbl-rag-limit");
    limitSlider.addEventListener("input", (e) => {
        limitLabel.textContent = `Context: ${e.target.value} chunks`;
    });

    // Show compare / Hassan variable panels when the RAG template dropdown changes
    const ragTemplateSelect = document.getElementById("rag-template-select");
    if (ragTemplateSelect) {
        ragTemplateSelect.addEventListener("change", updateRagTemplateUi);
    }

    document.getElementById("prompt-save-form")?.addEventListener("submit", handlePromptSave);
    document.getElementById("pe-clear")?.addEventListener("click", clearPromptEditorForm);
    document.getElementById("pe-new")?.addEventListener("click", () => {
        editingPromptName = null;
        clearPromptEditorForm();
    });

    // Upload button opens the native file picker
    document.getElementById("btn-upload-pdfs").addEventListener("click", () => {
        document.getElementById("pdf-file-input").click();
    });
    // When files are selected, start uploading immediately
    document.getElementById("pdf-file-input").addEventListener("change", handleUploadPDFs);

    // Load More button
    document.getElementById("btn-load-more").addEventListener("click", handleLoadMore);

    // Real-time filter and sort bindings
    ["filter-author", "filter-venue", "filter-year-min", "filter-year-max", "sort-results"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener("input", renderFilteredAndSortedPapers);
    });


    /* ==========================================================================
       HEALTH CHECK & UTILS
       ========================================================================== */

    /**
     * Check backend health (Ollama + ChromaDB) and update the status indicator badges.
     * Also populates the Knowledge Base stats (paper count + chunk count).
     */
    async function checkHealth() {
        try {
            const resp = await fetch(`${API_BASE}/api/health`);
            const data = await resp.json();

            // Update the Ollama and ChromaDB status dot + label in the header
            updateStatusIndicator("status-ollama", data.ollama === "online", `Ollama: ${data.ollama.toUpperCase()}`);
            updateStatusIndicator("status-db", data.vector_db === "online", `ChromaDB: ${data.vector_db.toUpperCase()}`);

            // Populate the stats badges in the Knowledge Base sidebar
            if (data.db_stats) {
                document.getElementById("stat-papers-count").textContent = data.db_stats.total_papers || 0;
                document.getElementById("stat-chunks-count").textContent = data.db_stats.total_chunks || 0;
            }
        } catch (err) {
            // Backend unreachable — mark both services as offline
            updateStatusIndicator("status-ollama", false, "Ollama: OFFLINE");
            updateStatusIndicator("status-db", false, "ChromaDB: OFFLINE");
        }
    }

    /**
     * Update a status indicator element's CSS class and label text.
     * @param {string} elementId - ID of the status indicator element.
     * @param {boolean} isOnline - True if the service is online.
     * @param {string} labelText - Text to display in the label span.
     */
    function updateStatusIndicator(elementId, isOnline, labelText) {
        const el = document.getElementById(elementId);
        if (!el) return;
        if (isOnline) {
            el.classList.remove("offline");
            el.classList.add("online");
        } else {
            el.classList.remove("online");
            el.classList.add("offline");
        }
        el.querySelector(".status-label").textContent = labelText;
    }

    /**
     * Format a raw byte count into a human-readable string (e.g. "2.34 MB").
     * @param {number} bytes - Raw byte count.
     * @param {number} decimals - Number of decimal places (default: 2).
     * @returns {string} Formatted size string.
     */
    function formatBytes(bytes, decimals = 2) {
        if (!bytes) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    /**
     * Escape a string for safe insertion into HTML content.
     * Prevents XSS when rendering user-supplied filenames or metadata.
     * @param {string} str - Raw string to escape.
     * @returns {string} HTML-safe string.
     */
    function escapeHTML(str) {
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    /* ==========================================================================
       TAB ROUTING SYSTEM
       ========================================================================== */

    /**
     * Initialise the SPA tab navigation system.
     * Clicking a tab hides all others, shows the selected one, and syncs data.
     */
    function initTabs() {
        const navButtons = document.querySelectorAll(".nav-btn");
        const panes = document.querySelectorAll(".tab-pane");

        function activateTab(targetTab) {
            navButtons.forEach(b => b.classList.remove("active"));
            panes.forEach(p => p.classList.remove("active"));

            const matchingBtn = [...navButtons].find(b => b.getAttribute("data-tab") === targetTab);
            if (matchingBtn) matchingBtn.classList.add("active");

            const pane = document.getElementById(targetTab);
            if (pane) pane.classList.add("active");

            // Persist choice in both localStorage AND URL hash (survives refresh + allows bookmarking)
            try { localStorage.setItem("cite_active_tab", targetTab); } catch (_) {}
            try {
                // Update hash without triggering a page scroll
                history.replaceState(null, "", `#${targetTab}`);
            } catch (_) {}

            // Refresh relevant data when switching tabs
            if (targetTab === "tab-rag") fetchLocalPDFs();
            else if (targetTab === "tab-citations") fetchReports();
        }

        navButtons.forEach(btn => {
            btn.addEventListener("click", () => activateTab(btn.getAttribute("data-tab")));
        });

        // Priority 1: URL hash (most reliable — survives hard refresh, supports bookmarks)
        try {
            const hash = window.location.hash.replace("#", "");
            if (hash && document.getElementById(hash)) {
                activateTab(hash);
                return;
            }
        } catch (_) {}

        // Priority 2: localStorage fallback
        try {
            const saved = localStorage.getItem("cite_active_tab");
            if (saved && document.getElementById(saved)) {
                activateTab(saved);
                return;
            }
        } catch (_) {}

        // Default: Paper Discovery
        activateTab("tab-discovery");
    }

    /* ==========================================================================
       TAB 1: PAPER DISCOVERY
       ========================================================================== */

    /**
     * Handle the paper search form submission.
     * Queries the backend /api/search endpoint and renders result cards.
     */
    async function handlePaperSearch(e) {
        e.preventDefault();
        const query = document.getElementById("search-input").value.trim();
        const limit = parseInt(document.getElementById("search-limit").value) || 10;
        const statusBox = document.getElementById("search-status-message");
        const resultsList = document.getElementById("search-results-list");
        const loadMoreContainer = document.getElementById("load-more-container");
        const filterBar = document.getElementById("filter-sort-bar");

        if (!query) return;

        // Reset pagination state for a fresh search
        currentSearchQuery = query;
        currentSearchLimit = limit;
        currentSearchOffset = 0;
        allFetchedPapers = [];
        hasMorePapers = true;

        // Reset filter inputs
        ["filter-author", "filter-venue", "filter-year-min", "filter-year-max"].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.value = "";
        });
        const sortEl = document.getElementById("sort-results");
        if (sortEl) sortEl.value = "relevance";

        // Show loading spinner while waiting for results
        statusBox.classList.remove("hidden");
        resultsList.innerHTML = "";
        loadMoreContainer.classList.add("hidden");
        filterBar.classList.add("hidden");

        try {
            const exactAuthor = document.getElementById("search-exact-author")?.checked ? "true" : "false";
            const resp = await fetch(
                `${API_BASE}/api/search?q=${encodeURIComponent(query)}&limit=${limit}&offset=0&exact_author=${exactAuthor}`
            );
            const papers = await resp.json();
            statusBox.classList.add("hidden");

            if (!papers || papers.length === 0) {
                resultsList.innerHTML = `
                    <div class="placeholder-card glass-card">
                        <i class="fa-solid fa-face-frown placeholder-icon"></i>
                        <h3>No Papers Found</h3>
                        <p>No papers found for that query on Semantic Scholar. Try different keywords.</p>
                    </div>
                `;
                return;
            }

            allFetchedPapers = papers;
            currentSearchOffset = papers.length;
            hasMorePapers = papers.length >= limit;

            renderFilteredAndSortedPapers();

            // Show filter bar and Load More button
            filterBar.classList.remove("hidden");
            if (hasMorePapers) loadMoreContainer.classList.remove("hidden");

        } catch (err) {
            statusBox.classList.add("hidden");
            resultsList.innerHTML = `
                <div class="placeholder-card glass-card">
                    <i class="fa-solid fa-circle-exclamation placeholder-icon text-crimson"></i>
                    <h3>Search Failed</h3>
                    <p>An error occurred while calling the search service. Please check your backend logs.</p>
                </div>
            `;
        }
    }

    /**
     * Fetch the next page of results and append to allFetchedPapers.
     */
    async function handleLoadMore() {
        const btn = document.getElementById("btn-load-more");
        const statusBox = document.getElementById("search-status-message");
        if (!currentSearchQuery) return;

        btn.disabled = true;
        btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Loading...`;
        statusBox.classList.remove("hidden");

        try {
            const exactAuthor = document.getElementById("search-exact-author")?.checked ? "true" : "false";
            const resp = await fetch(
                `${API_BASE}/api/search?q=${encodeURIComponent(currentSearchQuery)}&limit=${currentSearchLimit}&offset=${currentSearchOffset}&exact_author=${exactAuthor}`
            );
            const papers = await resp.json();
            statusBox.classList.add("hidden");

            if (papers && papers.length > 0) {
                allFetchedPapers = allFetchedPapers.concat(papers);
                currentSearchOffset += papers.length;
                hasMorePapers = papers.length >= currentSearchLimit;
                renderFilteredAndSortedPapers();
            } else {
                hasMorePapers = false;
            }

            btn.disabled = false;
            btn.innerHTML = `<span>Load More Papers</span>`;
            if (!hasMorePapers) {
                document.getElementById("load-more-container").classList.add("hidden");
            }
        } catch (err) {
            statusBox.classList.add("hidden");
            btn.disabled = false;
            btn.innerHTML = `<span>Load More Papers</span>`;
        }
    }

    /**
     * Re-render the search results grid based on current filter inputs and sort selection.
     * Operates purely on allFetchedPapers — no network call needed.
     */
    function renderFilteredAndSortedPapers() {
        const resultsList = document.getElementById("search-results-list");
        const authorFilter = (document.getElementById("filter-author")?.value || "").trim().toLowerCase();
        const venueFilter  = (document.getElementById("filter-venue")?.value  || "").trim().toLowerCase();
        const yearMin = parseInt(document.getElementById("filter-year-min")?.value) || 0;
        const yearMax = parseInt(document.getElementById("filter-year-max")?.value) || 9999;
        const sortBy  = document.getElementById("sort-results")?.value || "relevance";

        let filtered = allFetchedPapers.filter(paper => {
            const authors  = formatAuthors(paper.authors).toLowerCase();
            const venue    = (paper.venue || "").toLowerCase();
            const year     = parseInt(paper.year) || 0;

            if (authorFilter && !authors.includes(authorFilter)) return false;
            if (venueFilter  && !venue.includes(venueFilter))   return false;
            if (year && (year < yearMin || year > yearMax))       return false;
            return true;
        });

        // Sort
        if (sortBy === "citations") {
            filtered.sort((a, b) => (b.citationCount || 0) - (a.citationCount || 0));
        } else if (sortBy === "year-desc") {
            filtered.sort((a, b) => (parseInt(b.year) || 0) - (parseInt(a.year) || 0));
        } else if (sortBy === "year-asc") {
            filtered.sort((a, b) => (parseInt(a.year) || 0) - (parseInt(b.year) || 0));
        }
        // "relevance" keeps original API order

        resultsList.innerHTML = "";

        if (filtered.length === 0) {
            resultsList.innerHTML = `
                <div class="placeholder-card glass-card">
                    <i class="fa-solid fa-filter-circle-xmark placeholder-icon"></i>
                    <h3>No Matching Papers</h3>
                    <p>No papers match the current filter criteria. Try adjusting the filters or loading more results.</p>
                </div>
            `;
            return;
        }

        filtered.forEach(paper => resultsList.appendChild(createPaperCard(paper)));
    }

    /**
     * Build a paper result card DOM element from a paper metadata object.
     * @param {Object} paper - Paper metadata from the /api/search response.
     * @returns {HTMLElement} The card element.
     */
    function createPaperCard(paper) {
        const card = document.createElement("div");
        card.className = "paper-card glass-card";

        const authorsStr = formatAuthors(paper.authors);
        const doiLabel = paper.doi !== "N/A" ? paper.doi : "None";
        // Highlight highly-cited papers in purple vs standard indigo
        const citationBadgeColor = paper.citationCount > 100 ? "var(--accent-purple)" : "var(--accent-indigo)";

        // Generate a unique DOM ID for the abstract toggle (prevents ID collisions in long result lists)
        const absId = `abs-${Math.random().toString(36).substring(2, 9)}`;

        // --- Duplicate ingest detection ---
        // Check if this paper is already in the local knowledge base (by DOI or normalised title)
        const normTitle = (s) => (s || "").toLowerCase().replace(/[^a-z0-9]/g, "");
        const paperNormTitle = normTitle(paper.title);
        const alreadyIngested = ingestedPapers.some(ip => {
            if (paper.paperId && ip.paper_id && ip.paper_id === paper.paperId) return true;
            if (paper.doi && paper.doi !== "N/A" && ip.doi && ip.doi !== "N/A") {
                return ip.doi.toLowerCase() === paper.doi.toLowerCase();
            }
            return normTitle(ip.title) === paperNormTitle;
        });

        // If there is no OA PDF and no abstract snippet, ingestion cannot proceed meaningfully.
        const canIngest = paper.has_pdf || !!(paper.abstract && paper.abstract.trim());
        const ingestButtonHtml = alreadyIngested
            ? `<button class="btn btn-secondary btn-download-ingest" disabled style="opacity:0.65; cursor:default;">
                <i class="fa-solid fa-circle-check text-emerald"></i> Already in Knowledge Base
               </button>`
            : !canIngest
            ? `<button class="btn btn-secondary btn-download-ingest" disabled title="No open-access PDF or abstract snippet was available from metadata." style="opacity:0.65; cursor:not-allowed;">
                <i class="fa-solid fa-ban"></i> No PDF/Abstract Available
               </button>`
            : `<button class="btn btn-primary btn-download-ingest" id="btn-ingest-${paper.paperId}">
                <i class="fa-solid fa-cloud-arrow-down"></i> ${paper.has_pdf ? 'Download & Ingest PDF' : 'Ingest Abstract Only'}
               </button>`;

        card.innerHTML = `
            <div class="paper-card-header">
                <div>
                    <h3 class="paper-title">${paper.title}</h3>
                    <div class="paper-meta">
                        <span><i class="fa-solid fa-users"></i> ${authorsStr}</span>
                        <span><i class="fa-solid fa-calendar"></i> ${paper.year}</span>
                        <span><i class="fa-solid fa-hotel"></i> ${paper.venue}</span>
                        <span style="font-weight: 600; color: ${citationBadgeColor}">
                            <i class="fa-solid fa-quote-left"></i> Citations: ${paper.citationCount}
                        </span>
                    </div>
                </div>
                <div>
                    ${paper.has_pdf ?
                '<span class="badge badge-oa"><i class="fa-solid fa-unlock-keyhole"></i> PDF Available</span>' :
                '<span class="badge badge-paywall"><i class="fa-solid fa-file-invoice"></i> Abstract Only</span>'
            }
                </div>
            </div>

            <div class="paper-abstract" id="${absId}">
                <strong>Abstract summary:</strong> ${paper.abstract ? paper.abstract : "No abstract snippet indexed (cannot ingest abstract-only for this paper)."}
            </div>

            <div class="paper-footer">
                <span class="report-meta">DOI: ${doiLabel} | arXiv: ${paper.arxiv}</span>
                ${paper.article_url ? `<a class="btn btn-secondary btn-icon paper-article-link" href="${paper.article_url}" target="_blank" rel="noopener noreferrer" title="Open on Semantic Scholar"><i class="fa-solid fa-arrow-up-right-from-square"></i> View on Semantic Scholar</a>` : ""}
                ${ingestButtonHtml}
            </div>
        `;

        // Wire up the ingestion button click handler (only if not already ingested)
        if (!alreadyIngested && canIngest) {
            const btn = card.querySelector(`.btn-download-ingest`);
            if (btn) btn.addEventListener("click", () => triggerIngestion(paper, btn));
        }

        return card;
    }

    /**
     * Format a list of author objects into a readable string.
     * Shows up to 3 names, then appends "et al." for longer lists.
     * @param {Array} authors - Array of author objects with a "name" key.
     * @returns {string} Formatted author string.
     */
    function formatAuthors(authors) {
        if (!authors || authors.length === 0) return "Unknown Authors";
        const names = authors.map(a => a.name).filter(Boolean);
        if (names.length > 3) {
            return names.slice(0, 3).join(", ") + " et al.";
        }
        return names.join(", ");
    }

    /**
     * Trigger the download + ingestion pipeline for a paper.
     * Sends a POST to /api/download and updates the button state to reflect progress.
     * @param {Object} paper - Paper metadata object.
     * @param {HTMLButtonElement} button - The ingest button element.
     */
    async function triggerIngestion(paper, button) {
        // Guard against abstract-only ingestion when no abstract is actually available.
        if (!paper.has_pdf && !(paper.abstract && paper.abstract.trim())) {
            alert("This paper has no open-access PDF and no abstract snippet available to ingest.");
            return;
        }
        button.disabled = true;
        button.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> ${paper.has_pdf ? "Downloading PDF..." : "Ingesting abstract..."}`;

        try {
            const resp = await fetch(`${API_BASE}/api/download`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    title: paper.title,
                    authors: paper.authors,
                    venue: paper.venue,
                    year: paper.year !== "N/A" ? parseInt(paper.year) : null,
                    externalIds: {
                        DOI: paper.doi !== "N/A" ? paper.doi : null,
                        ArXiv: paper.arxiv !== "N/A" ? paper.arxiv : null
                    },
                    abstract: paper.abstract,
                    citationCount: paper.citationCount,
                    paperId: paper.paperId || null
                })
            });
            // Refresh duplicate-detection cache immediately after ingest starts
            if (paper.paperId) {
                ingestedPapers.push({
                    title: paper.title,
                    doi: paper.doi,
                    paper_id: paper.paperId
                });
            }

            const result = await resp.json();

            if (result.success) {
                // Show success state — distinguish full PDF from abstract-only
                button.className = "btn btn-secondary";
                if (result.mode === "pdf") {
                    button.innerHTML = `<i class="fa-solid fa-circle-check text-emerald"></i> PDF Ingested ✓`;
                } else if (result.mode === "abstract") {
                    button.innerHTML = `<i class="fa-solid fa-file-lines"></i> Abstract Only ⚠`;
                    button.title = "No open-access PDF found — abstract stored instead.";
                } else {
                    button.innerHTML = `<i class="fa-solid fa-circle-check text-emerald"></i> Ingested ✓`;
                }
                checkHealth();      // Refresh paper/chunk count badges
                fetchLocalPDFs();   // Refresh the manifest list in the KnowledgeBase tab
            } else {
                throw new Error("Ingestion aborted by vector DB handler");
            }
        } catch (err) {
            button.disabled = false;
            button.innerHTML = `<i class="fa-solid fa-circle-exclamation text-crimson"></i> Retry Ingestion`;
            alert(`Ingestion failed: ${err.message || err}`);
        }
    }


    /* ==========================================================================
       TAB 2: RAG KNOWLEDGE BASE
       ========================================================================== */

    /**
     * Fetch the current PDF manifest from /api/pdfs and render the file list.
     * Shows ingestion status badges (success / pending / failed) for each file.
     */
    async function fetchLocalPDFs() {
        const listDiv = document.getElementById("local-files-list");
        try {
            const resp = await fetch(`${API_BASE}/api/pdfs`);
            const files = await resp.json();

            // Update the global ingested-papers cache used for duplicate detection in Paper Discovery
            localManifestFiles = files;
            ingestedPapers = files.filter(f => f.status === "success").map(f => ({
                title: f.title || "",
                doi: f.doi || "N/A",
                authors: f.authors,
                year: f.year,
                paper_id: f.paper_id || "",
            }));

            // Always refresh the paper filter dropdown whenever the manifest is loaded
            populatePaperFilter(files);

            if (files.length === 0) {
                listDiv.innerHTML = `
                    <div class="list-empty">
                        <i class="fa-solid fa-box-open" style="font-size:24px; margin-bottom:8px; opacity:0.5;"></i>
                        <p>No PDFs saved in papers/ directory.</p>
                    </div>
                `;
                return;
            }

            listDiv.innerHTML = "";
            files.forEach(file => {
                const item = document.createElement("div");
                item.className = "file-item";

                // Choose status badge based on manifest status field
                let statusBadge = "";
                if (file.status === "success") {
                    statusBadge = `<span class="badge badge-success" style="font-size: 9px;"><i class="fa-solid fa-circle-check"></i> Ingested</span>`;
                } else if (file.status === "pending") {
                    statusBadge = `<span class="badge badge-pending" style="font-size: 9px;"><i class="fa-solid fa-spinner fa-spin"></i> Pending</span>`;
                } else {
                    statusBadge = `<span class="badge badge-failed" style="font-size: 9px;"><i class="fa-solid fa-triangle-exclamation"></i> Error</span>`;
                }

                const sidebarLabel = formatSidebarLabel(file.authors, file.year, file.title);

                item.innerHTML = `
                    <div class="file-item-main" title="${file.title}">
                        <div class="file-name-row">
                            ${file.status === "success" 
                                ? `<a class="sidebar-paper-link" href="${API_BASE}/api/papers/${encodeURIComponent(file.filename)}" target="_blank" title="Open PDF in new tab">${sidebarLabel}</a>`
                                : `<span class="sidebar-paper-inactive" style="opacity: 0.6;">${sidebarLabel}</span>`
                            }
                        </div>
                        <div class="file-meta-row">
                            <span>${formatBytes(file.size_bytes)}</span>
                            ${statusBadge}
                        </div>
                    </div>
                    <div class="file-item-actions" style="display: flex; gap: 4px; align-items: center; flex-shrink: 0;">
                        <button class="btn-view-abstract btn-icon" title="View Ingested Abstract" style="padding: 6px; border-radius: 6px; color: var(--accent-indigo) !important; opacity: 0.7; transition: opacity 0.2s, transform 0.2s;">
                            <i class="fa-solid fa-file-lines"></i>
                        </button>
                        <button class="btn-delete-file btn-icon" data-filename="${file.filename}" title="Delete Paper">
                            <i class="fa-solid fa-trash-can text-crimson"></i>
                        </button>
                    </div>
                `;

                // Wire up view abstract button
                const viewBtn = item.querySelector(".btn-view-abstract");
                if (viewBtn) {
                    viewBtn.addEventListener("click", (e) => {
                        e.stopPropagation();
                        showAbstractModal(file);
                    });
                }

                listDiv.appendChild(item);
            });

            // Wire up deletion for each button — no confirmation dialog, no alert pop-ups
            listDiv.querySelectorAll(".btn-delete-file").forEach(btn => {
                btn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    const filename = btn.getAttribute("data-filename");
                    try {
                        btn.disabled = true;
                        btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i>`;
                        // Await the delete to fully complete before refreshing stats
                        await fetch(`${API_BASE}/api/papers/${filename}`, { method: "DELETE" });
                    } catch (delErr) {
                        // Continue even on network error
                    } finally {
                        // Small delay to let ChromaDB commit the deletion before stat refresh
                        await new Promise(r => setTimeout(r, 600));
                        await checkHealth();
                        await fetchLocalPDFs();
                    }
                });
            });

        } catch (err) {
            listDiv.innerHTML = `<div class="list-empty text-crimson">Failed to load manifest.</div>`;
        }
    }

    /**
     * Display the abstract viewer modal for a given local paper entry.
     * @param {Object} file - The file object from the manifest.
     */
    function showAbstractModal(file) {
        const modal = document.getElementById("abstract-modal");
        if (!modal) return;
        document.getElementById("modal-title").textContent = file.title || "Paper Abstract";
        document.getElementById("modal-authors").textContent = file.authors || "Unknown Authors";
        document.getElementById("modal-year").textContent = file.year && file.year !== "None" && file.year !== "N/A" ? file.year : "N/A";
        
        const doiRow = document.getElementById("modal-doi-row");
        const doiLink = document.getElementById("modal-doi-link");
        if (file.doi && file.doi !== "N/A" && file.doi !== "None") {
            doiLink.href = `https://doi.org/${file.doi}`;
            doiLink.textContent = file.doi;
            doiRow.style.display = "block";
        } else {
            doiRow.style.display = "none";
        }

        const abstractText = document.getElementById("modal-abstract-text");
        if (file.status === "pending") {
            abstractText.innerHTML = `<span style="font-style: italic; color: #fbbf24;"><i class="fa-solid fa-spinner fa-spin"></i> Ingestion Pending... The server has not finished extracting text and resolving metadata for this file yet. Please click 'Scan & Ingest Folder' above or wait for the upload queue to finish.</span>`;
        } else if (file.status === "failed") {
            const errDetail = file.error || "Unknown extraction error.";
            abstractText.innerHTML = `<span style="font-style: italic; color: var(--accent-crimson);"><i class="fa-solid fa-triangle-exclamation"></i> Ingestion Failed. Error details: ${errDetail}</span>`;
        } else if (file.abstract && file.abstract.trim()) {
            abstractText.textContent = file.abstract;
        } else {
            abstractText.innerHTML = `<span style="font-style: italic; opacity: 0.6;">No abstract stored in manifest. You can RAG query this document to see extract details.</span>`;
        }

        modal.classList.remove("hidden");
    }

    /**
     * Populate the paper filter dropdown in the RAG tab.
     * Adds one <option> per successfully ingested paper, plus an "All Papers" default.
     * @param {Array} files - Array of file objects from /api/pdfs response.
     */
    function populatePaperFilter(files) {
        const sel = document.getElementById("rag-paper-filter");
        const selB = document.getElementById("rag-paper-filter-b");
        if (!sel) return;

        const prevVal = sel.value;
        const prevValB = selB ? selB.value : "";

        const fillSelect = (selectEl, includeEmpty, emptyLabel) => {
            if (!selectEl) return;
            selectEl.innerHTML = includeEmpty
                ? `<option value="">${emptyLabel}</option>`
                : "";
            const ingested = files.filter(f => f.status === "success");
            ingested.forEach(f => {
                const label = formatSidebarLabel(f.authors, f.year, f.title);
                const opt = document.createElement("option");
                opt.value = f.title;
                opt.textContent = label;
                opt.title = f.title;
                selectEl.appendChild(opt);
            });
        };

        fillSelect(sel, true, "All Papers in Knowledge Base");
        fillSelect(selB, true, "— Select second paper —");

        if ([...sel.options].some(o => o.value === prevVal)) sel.value = prevVal;
        if (selB && [...selB.options].some(o => o.value === prevValB)) selB.value = prevValB;
    }

    /**
     * Show/hide comparative second-paper selector and Hassan template variable fields.
     */
    function updateRagTemplateUi() {
        const template = document.getElementById("rag-template-select")?.value || "";
        const compareWrap = document.getElementById("rag-compare-b-wrap");
        const varsPanel = document.getElementById("rag-template-vars-panel");

        if (compareWrap) {
            compareWrap.classList.toggle("hidden", template !== "comparative_analysis");
        }
        if (varsPanel) {
            varsPanel.classList.toggle("hidden", template !== "hassanian_article");
        }
    }

    /**
     * Collect optional template variable fields for custom prompts (Hassan-style, etc.).
     */
    function collectTemplateVars() {
        const map = {
            phenomenon: document.getElementById("tv-phenomenon")?.value?.trim() || "",
            central_thesis: document.getElementById("tv-central-thesis")?.value?.trim() || "",
            generative_practice: document.getElementById("tv-generative-practice")?.value?.trim() || "",
            paradigm_level: document.getElementById("tv-paradigm-level")?.value?.trim() || "",
            stance: document.getElementById("tv-stance")?.value?.trim() || "",
            native_construct: document.getElementById("tv-native-construct")?.value?.trim() || "",
            target_journal: document.getElementById("tv-target-journal")?.value?.trim() || "",
            coauthors: document.getElementById("tv-coauthors")?.value?.trim() || "",
        };
        const out = {};
        Object.entries(map).forEach(([k, v]) => { if (v) out[k] = v; });
        return Object.keys(out).length ? out : null;
    }

    /**
     * APA-style label for a retrieved chunk in the sources panel (not "Source N").
     */
    function formatChunkCitationLabel(meta, index) {
        const authors = meta.authors || "Unknown Authors";
        const year = meta.year || "N/A";
        let pages = meta.pages;
        let pagePart = "";
        if (pages && pages !== "N/A") {
            pagePart = `, p. ${pages}`;
        }
        return `(${authors}, ${year}${pagePart})`;
    }

    /**
     * Build a compact academic-style sidebar label from authors + year.
     * Format: "LastName, Year" | "LastName & LastName2, Year" | "LastName et al., Year"
     * Falls back to a shortened title if authors/year are not available.
     * @param {string} authorsStr - Authors string from manifest (e.g. "Smith, J., Doe, A. et al.")
     * @param {string} year - Publication year (e.g. "2026") or "N/A".
     * @param {string} title - Full paper title (used as fallback).
     * @returns {string} Formatted label.
     */
    function formatSidebarLabel(authorsStr, year, title) {
        const hasYear = year && year !== "N/A" && year !== "None";
        let hasAuthors = authorsStr && authorsStr !== "Unknown Authors";

        // Some manifest rows store "Alzubaidi et al., 2021" inside the title field only.
        if (!hasAuthors && title) {
            const embedded = title.match(/^([A-Za-z][A-Za-z\-']+(?:\s+[A-Za-z][A-Za-z\-']+)?)\s+et\s+al\.?,?\s*(\d{4})?/i);
            if (embedded) {
                authorsStr = embedded[1].trim() + " et al.";
                if (!hasYear && embedded[2]) year = embedded[2];
                hasAuthors = true;
            }
        }

        if (!hasAuthors && !hasYear) {
            // Pure fallback: humanise technical filenames into readable labels.
            const pretty = (title || "Unknown Paper")
                .replace(/[_-]+/g, " ")
                .replace(/\s+/g, " ")
                .replace(/\.pdf$/i, "")
                .trim();
            return pretty.length > 35 ? pretty.slice(0, 33) + "…" : pretty;
        }

        // Extract last names from a formatted authors string.
        // Handles common patterns like:
        // - "N. Hassan"
        // - "Smith, John"
        // - "Smith, J., Doe, A."
        // - "Smith et al."
        const extractLastNames = (str) => {
            const cleaned = String(str || "").replace(/\s*et al\.?$/i, "").trim();
            if (!cleaned) return [];
            const semicolonSplit = cleaned.split(/\s*;\s*/).filter(Boolean);
            const rawParts = semicolonSplit.length > 1 ? semicolonSplit : cleaned.split(/\s*,\s*/).filter(Boolean);
            const names = rawParts.map(part => {
                const tokenised = part.trim().split(/\s+/).filter(Boolean);
                if (tokenised.length === 0) return "";
                // Prefer the last non-initial token as the surname.
                const nonInitial = tokenised.filter(t => !/^[A-Z]\.?$/i.test(t));
                const candidate = (nonInitial[nonInitial.length - 1] || tokenised[tokenised.length - 1] || "").replace(/\./g, "");
                return candidate;
            }).filter(Boolean);
            return names.slice(0, 3);
        };

        const hasEtAl = /et al\.?/i.test(authorsStr);
        const lastNames = hasAuthors ? extractLastNames(authorsStr) : [];
        const yearPart = hasYear ? year : "";

        let namePart;
        if (lastNames.length === 0) {
            namePart = "";
        } else if (lastNames.length === 1) {
            namePart = hasEtAl ? `${lastNames[0]} et al.` : lastNames[0];
        } else if (lastNames.length === 2) {
            namePart = `${lastNames[0]} & ${lastNames[1]}`;
        } else {
            namePart = `${lastNames[0]} et al.`;
        }

        if (namePart && yearPart) return `${namePart}, ${yearPart}`;
        if (namePart) return namePart;
        return yearPart || title;
    }

    /**
     * Handle the "Scan & Ingest Folder" button click.
     * Calls /api/ingest-pending and reports results to the user.
     */
    async function handleScanPending() {
        const btn = document.getElementById("btn-scan-pending");
        btn.disabled = true;
        btn.innerHTML = `<i class="fa-solid fa-arrows-spin fa-spin"></i> Scanning folder...`;

        try {
            const resp = await fetch(`${API_BASE}/api/ingest-pending`, { method: "POST" });
            const result = await resp.json();

            btn.disabled = false;
            btn.innerHTML = `<i class="fa-solid fa-arrows-spin"></i> Scan & Ingest Folder`;

            // Refresh the file list to show "pending" or new status
            fetchLocalPDFs();
            checkHealth();

            // Set up a temporary shorter polling interval for health and files
            // to show progress in real time (every 4 seconds, for 1 minute)
            let pollCount = 0;
            const tempPollInterval = setInterval(async () => {
                fetchLocalPDFs();
                checkHealth();
                pollCount++;
                if (pollCount >= 15) { // 15 * 4s = 60s
                    clearInterval(tempPollInterval);
                }
            }, 4000);

        } catch (err) {
            btn.disabled = false;
            btn.innerHTML = `<i class="fa-solid fa-arrows-spin"></i> Scan & Ingest Folder`;
        }
    }

    /**
     * Handle PDF file upload from the user's local computer.
     * Triggered when files are selected via the hidden <input type="file">.
     * Uploads each file individually to /api/upload, showing per-file progress rows.
     * After all uploads finish, refreshes the manifest and stats counters.
     */
    async function handleUploadPDFs(e) {
        let files = Array.from(e.target.files);
        if (!files.length) return;

        const statusList = document.getElementById("upload-status-list");
        const subfolder = document.getElementById("upload-subfolder").value.trim();
        const uploadBtn = document.getElementById("btn-upload-pdfs");

        // --- Check for duplicates ---
        if (localManifestFiles && localManifestFiles.length > 0) {
            const existingNames = new Set(localManifestFiles.map(f => {
                const parts = f.filename.split(/[\/\\]/);
                return parts[parts.length - 1].toLowerCase();
            }));
            const duplicates = files.filter(file => existingNames.has(file.name.toLowerCase()));
            if (duplicates.length > 0) {
                const dupNames = duplicates.map(file => file.name).join(", ");
                const proceed = confirm(`The following PDF(s) already exist in your Knowledge Base:\n\n${dupNames}\n\nDo you want to re-upload and re-ingest them? Click Cancel to skip these duplicates.`);
                if (!proceed) {
                    const dupSet = new Set(duplicates.map(f => f.name));
                    files = files.filter(file => !dupSet.has(file.name));
                    if (files.length === 0) {
                        e.target.value = "";
                        return;
                    }
                }
            }
        }

        uploadBtn.disabled = true;
        uploadBtn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Uploading...`;
        statusList.innerHTML = "";

        // Create a status row for each file immediately
        const rowIds = {};
        files.forEach(file => {
            const rowId = `up-${Math.random().toString(36).substring(2, 8)}`;
            rowIds[file.name] = rowId;
            const row = document.createElement("div");
            row.className = "upload-row upload-row-pending";
            row.id = rowId;
            row.innerHTML = `
                <i class="fa-solid fa-spinner fa-spin upload-row-icon"></i>
                <span class="upload-row-name" title="${escapeHTML(file.name)}">${escapeHTML(file.name.length > 30 ? file.name.slice(0, 28) + '…' : file.name)}</span>
                <span class="upload-row-size">${formatBytes(file.size)}</span>
            `;
            statusList.appendChild(row);
        });

        // Upload all files together as a single multipart POST
        try {
            const formData = new FormData();
            files.forEach(file => formData.append("files", file));
            if (subfolder) formData.append("subfolder", subfolder);

            const resp = await fetch(`${API_BASE}/api/upload`, {
                method: "POST",
                body: formData
                // Do NOT set Content-Type header — browser sets it with the multipart boundary
            });

            const result = await resp.json();

            // Mark each file row as succeeded or failed based on the result
            const uploadedNames = new Set((result.uploaded || []).map(u => u.filename.split(/[\/\\]/).pop()));
            const rejectedNames = new Set(result.rejected || []);

            files.forEach(file => {
                const rowEl = document.getElementById(rowIds[file.name]);
                if (!rowEl) return;
                if (uploadedNames.has(file.name)) {
                    rowEl.className = "upload-row upload-row-success";
                    rowEl.querySelector(".upload-row-icon").className = "fa-solid fa-circle-check upload-row-icon";
                } else if (rejectedNames.has(file.name)) {
                    rowEl.className = "upload-row upload-row-failed";
                    rowEl.querySelector(".upload-row-icon").className = "fa-solid fa-circle-xmark upload-row-icon";
                } else {
                    rowEl.className = "upload-row upload-row-failed";
                    rowEl.querySelector(".upload-row-icon").className = "fa-solid fa-triangle-exclamation upload-row-icon";
                }
            });

        } catch (err) {
            // Mark all rows as failed on network error
            files.forEach(file => {
                const rowEl = document.getElementById(rowIds[file.name]);
                if (rowEl) {
                    rowEl.className = "upload-row upload-row-failed";
                    rowEl.querySelector(".upload-row-icon").className = "fa-solid fa-circle-xmark upload-row-icon";
                }
            });
        } finally {
            // Reset the button and file input for the next upload
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = `<i class="fa-solid fa-file-arrow-up"></i> Select PDF Files...`;
            e.target.value = "";

            // Poll the manifest for a minute to reflect ingestion progress
            await fetchLocalPDFs();
            await checkHealth();
            let pollCount = 0;
            const poll = setInterval(async () => {
                await fetchLocalPDFs();
                await checkHealth();
                if (++pollCount >= 15) clearInterval(poll);
            }, 4000);
        }
    }

    /**
     * Handle the RAG chat form submission.
     * Sends the query to /api/query-rag and renders the grounded answer as a chat bubble.
     */
    async function handleRAGQuery(e) {
        e.preventDefault();
        const inputEl = document.getElementById("rag-input");
        const query = inputEl.value.trim();
        const limit = document.getElementById("rag-limit-slider").value;
        const template = document.getElementById("rag-template-select").value;
        const messagesDiv = document.getElementById("chat-messages");
        const submitBtn = document.getElementById("rag-submit-btn");

        if (!query) return;

        // Show the user's message as a chat bubble immediately
        appendChatBubble("user", query);
        inputEl.value = "";

        // Show a loading indicator while waiting for the LLM response
        const loadId = appendChatBubble("bot", `<i class="fa-solid fa-ellipsis fa-bounce"></i> Thinking...`, [], true);
        submitBtn.disabled = true;

        try {
            // Get optional paper filter — empty string means query all papers
            const paperFilter = document.getElementById("rag-paper-filter")?.value || "";
            const paperFilterB = document.getElementById("rag-paper-filter-b")?.value || "";
            const templateVars = collectTemplateVars();
            const payload = {
                query: query,
                limit: parseInt(limit),
                prompt_template: template ? template : null,
                filter_title: paperFilter || null,
                // Include recent turns so backend RAG can preserve memory after refresh.
                conversation_history: chatHistoryTurns.slice(-12),
            };
            if (template === "comparative_analysis" && paperFilter && paperFilterB) {
                payload.filter_title_b = paperFilterB;
            }
            if (templateVars) payload.template_vars = templateVars;

            const resp = await fetch(`${API_BASE}/api/query-rag`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!resp.ok) {
                const errData = await resp.json();
                throw new Error(errData.detail || "Server error");
            }

            const data = await resp.json();
            submitBtn.disabled = false;

            // Remove the loading bubble and render the actual answer
            document.getElementById(loadId).remove();

            // Store sources for the "Show retrieved chunks" button
            activeChatSources = data.sources || [];
            appendChatBubble("bot", data.answer, activeChatSources);

        } catch (err) {
            submitBtn.disabled = false;
            document.getElementById(loadId).remove();
            appendChatBubble("bot", `<span class="text-crimson"><i class="fa-solid fa-triangle-exclamation"></i> Error generating answer: ${escapeHTML(err.message || err)}</span>`, [], true);
        }
    }

    /**
     * Append a chat bubble (user or bot) to the messages container.
     * @param {string} sender - "user" or "bot".
     * @param {string} text - Message text (may contain HTML for bot messages).
     * @param {Array} sources - Retrieved context chunks (only for bot messages).
     * @param {boolean} isHtml - If true, bypasses parsing and renders text directly as HTML.
     * @returns {string} The generated bubble DOM ID (for later removal).
     */
    function appendChatBubble(sender, text, sources = [], isHtml = false) {
        const messagesDiv = document.getElementById("chat-messages");
        const bubble = document.createElement("div");
        const bubbleId = `msg-${Math.random().toString(36).substring(2, 9)}`;
        bubble.id = bubbleId;
        bubble.className = `chat-bubble ${sender}-message`;

        const avatarIcon = sender === "user" ? "fa-user" : "fa-microchip-ai";

        // Parse markdown for bot responses; escape HTML for user input (XSS prevention)
        const parsedText = sender === "bot" ? (isHtml ? text : parseMarkdown(text)) : escapeHTML(text);

        bubble.innerHTML = `
            <div class="bubble-avatar"><i class="fa-solid ${avatarIcon}"></i></div>
            <div class="bubble-content">
                ${parsedText}
                ${sources.length > 0 ? `<div style="margin-top: 10px; border-top:1px dashed var(--border-glass); padding-top:6px;"><button class="btn btn-secondary btn-icon" style="font-size:11px; padding:3px 8px;" id="btn-show-src-${bubbleId}"><i class="fa-solid fa-list-check"></i> Show retrieved chunks (${sources.length})</button></div>` : ""}
            </div>
        `;

        messagesDiv.appendChild(bubble);
        // Auto-scroll to the newest message
        messagesDiv.scrollTop = messagesDiv.scrollHeight;

        // Wire up the "Show retrieved chunks" button if sources exist
        if (sources.length > 0) {
            // Persist this message's source chunks so restored chats can still open source panels.
            sourcesByBubbleId[bubbleId] = sources;
            document.getElementById(`btn-show-src-${bubbleId}`).addEventListener("click", () => {
                showRetrievedSourcesPanel(sources);
            });
        }

        // Persist structured turns for reliable cross-refresh chat memory.
        if (!isHtml) {
            const role = sender === "user" ? "user" : "assistant";
            chatHistoryTurns.push({ role, content: String(text || "") });
        }

        persistChatHistory();

        // Wire up in-text citation reference links (e.g. [Source 1])
        bubble.querySelectorAll(".citation-ref").forEach(ref => {
            ref.addEventListener("click", (e) => {
                e.preventDefault();
                const sourceIdx = parseInt(ref.getAttribute("data-source-index")) - 1;
                showRetrievedSourcesPanel(sources, sourceIdx);
            });
        });

        return bubbleId;
    }

    /**
     * Display the sliding source chunks panel with retrieved context.
     * Optionally highlights and scrolls to a specific source chunk.
     * @param {Array} sources - Array of retrieved chunk objects.
     * @param {number|null} highlightIdx - Index to highlight (0-based).
     */
    function showRetrievedSourcesPanel(sources, highlightIdx = null) {
        const panel = document.getElementById("retrieved-sources-panel");
        const list = document.getElementById("sources-chunks-list");

        panel.classList.remove("hidden");
        list.innerHTML = "";

        const filteredSources = sources.filter(c => !isLikelyNonEnglishText(c?.text || ""));
        filteredSources.forEach((c, idx) => {
            const item = document.createElement("div");
            item.className = "source-chunk-item";
            // Highlight the clicked source with a blue border + tint
            if (highlightIdx !== null && highlightIdx === idx) {
                item.style.borderColor = "var(--accent-blue)";
                item.style.background = "rgba(0, 242, 254, 0.05)";
            }

            const meta = c.metadata || {};
            const pages = meta.pages ? `Pages: ${meta.pages}` : "Abstract snippet";

            item.innerHTML = `
                <div class="source-chunk-title">${formatChunkCitationLabel(meta, idx)} — "${escapeHTML(meta.title || "Untitled")}"</div>
                <div class="source-chunk-text">"${escapeHTML(c.text)}"</div>
            `;
            list.appendChild(item);

            // Scroll highlighted source into view
            if (highlightIdx !== null && highlightIdx === idx) {
                item.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }
        });

        if (filteredSources.length < sources.length) {
            const note = document.createElement("div");
            note.className = "source-chunk-item";
            note.style.borderStyle = "dashed";
            note.style.opacity = "0.75";
            note.innerHTML = `<div class="source-chunk-title">Filtered non-English chunks</div>
                <div class="source-chunk-text">Some retrieved passages were hidden because they were detected as mostly non-English text.</div>`;
            list.prepend(note);
        }
    }

    /**
     * Heuristic language filter for chunk display:
     * hide chunks dominated by non-Latin scripts to keep the panel English-focused.
     */
    function isLikelyNonEnglishText(text) {
        if (!text) return false;
        const sample = text.slice(0, 1200);
        const latinChars = (sample.match(/[A-Za-z]/g) || []).length;
        const nonLatinChars = (
            sample.match(/[\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u0900-\u097F\u0E00-\u0E7F\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]/g) || []
        ).length;
        // Hide when non-Latin script clearly dominates and there is little Latin evidence.
        return nonLatinChars > 24 && nonLatinChars > (latinChars * 1.1);
    }

    function persistChatHistory() {
        try {
            const messagesDiv = document.getElementById("chat-messages");
            if (!messagesDiv) return;
            const payload = {
                html: messagesDiv.innerHTML,
                turns: chatHistoryTurns,
                sourcesByBubbleId: sourcesByBubbleId,
            };
            localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(payload));
        } catch (_) {}
    }

    function restoreChatHistory() {
        try {
            const raw = localStorage.getItem(CHAT_STORAGE_KEY);
            if (!raw) return;
            const parsed = JSON.parse(raw);
            const messagesDiv = document.getElementById("chat-messages");
            if (!messagesDiv || !parsed?.html) return;
            messagesDiv.innerHTML = parsed.html;
            // Restore structured state used for real multi-turn memory and source lookups.
            chatHistoryTurns = Array.isArray(parsed?.turns) ? parsed.turns : [];
            sourcesByBubbleId = parsed?.sourcesByBubbleId && typeof parsed.sourcesByBubbleId === "object"
                ? parsed.sourcesByBubbleId
                : {};
            // Re-bind source buttons in restored history.
            messagesDiv.querySelectorAll("[id^='btn-show-src-']").forEach(btn => {
                const bubbleId = (btn.id || "").replace("btn-show-src-", "");
                btn.addEventListener("click", () => {
                    const restoredSources = sourcesByBubbleId[bubbleId] || [];
                    if (restoredSources.length > 0) {
                        activeChatSources = restoredSources;
                        showRetrievedSourcesPanel(restoredSources);
                    }
                });
            });
        } catch (_) {}
    }

    /**
     * Escape HTML special characters to prevent XSS injection.
     * Applied to all user-typed text before inserting into the DOM.
     * @param {string} str - Raw string to escape.
     * @returns {string} HTML-safe string.
     */
    function escapeHTML(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    /**
     * Parse a subset of Markdown syntax into HTML for bot response rendering.
     * Handles: bold, italic, headers (h2/h3/h4), unordered lists, code, blockquotes,
     * citation reference links ([Source N]), and paragraph breaks.
     * @param {string} text - Raw markdown string from the LLM response.
     * @returns {string} HTML string wrapped in <p> tags.
     */
    function parseMarkdown(text) {
        // Start by escaping HTML to prevent XSS from LLM-generated content
        let html = escapeHTML(text);

        // Bold: **text** → <strong>text</strong>
        html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

        // Italics: *text* → <em>text</em>
        html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");

        // Headers: ### → h4, ## → h3, # → h2
        html = html.replace(/^### (.*?)$/gm, "<h4>$1</h4>");
        html = html.replace(/^## (.*?)$/gm, "<h3>$1</h3>");
        html = html.replace(/^# (.*?)$/gm, "<h2>$1</h2>");

        // Unordered list items: - text or * text → <li>
        html = html.replace(/^\s*[-*+]\s+(.*?)$/gm, "<li>$1</li>");
        // Wrap consecutive <li> elements in a <ul>
        html = html.replace(/(<li>.*?<\/li>)+/gs, "<ul>$&</ul>");

        // Inline code: `code` → <code>code</code>
        html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

        // Blockquotes: > text → <blockquote>
        html = html.replace(/^&gt;\s+(.*?)$/gm, "<blockquote>$1</blockquote>");

        // Legacy [Source N] links → scroll to chunk; label shows APA-style text when available
        html = html.replace(/\[Source\s+(\d+)\]/gi, (match, num) => {
            const idx = parseInt(num, 10) - 1;
            let label = `Source ${num}`;
            if (activeChatSources[idx]?.metadata) {
                label = formatChunkCitationLabel(activeChatSources[idx].metadata, idx);
            }
            return `<a class="citation-ref" href="#" data-source-index="${num}">${escapeHTML(label)}</a>`;
        });

        // Paragraph breaks: double newlines → </p><p>
        html = html.replace(/\n\n/g, "</p><p>");
        // Single newlines → <br>
        html = html.replace(/\n/g, "<br>");

        return `<p>${html}</p>`;
    }


    /* ==========================================================================
       TAB 3: CITATION ANALYSIS
       ========================================================================== */

    /**
     * Handle the citation analysis form submission.
     * Posts to /api/analyze-citations, receives a run_id, and polls for completion.
     */
    async function handleCitationAnalysis(e) {
        e.preventDefault();
        const paperId = document.getElementById("citation-paper-id").value.trim();
        const limit = document.getElementById("citation-limit").value;
        const jobCard = document.getElementById("citation-job-card");
        const matrixCard = document.getElementById("citation-matrix-card");

        if (!paperId) return;

        // Show the job status card and reset its state
        jobCard.classList.remove("hidden");
        matrixCard.classList.add("hidden");
        document.getElementById("job-bar-fill").style.width = "5%";
        document.getElementById("job-status-label").textContent = "Queued";
        document.getElementById("job-progress-text").textContent = "Triggering background analysis worker...";

        try {
            // Start the citation analysis job on the server
            const resp = await fetch(`${API_BASE}/api/analyze-citations`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    paper_id: paperId,
                    limit: parseInt(limit)
                })
            });

            const data = await resp.json();
            activeCitationRunId = data.run_id;

            // Elapsed time counter (updated every second)
            let seconds = 0;
            const timer = setInterval(() => {
                seconds++;
                document.getElementById("job-time").textContent = `Time: ${seconds}s`;
            }, 1000);

            // Clear any existing polling interval before starting a new one
            if (citationPollInterval) clearInterval(citationPollInterval);

            // Poll the job status endpoint every 1.5 seconds
            citationPollInterval = setInterval(async () => {
                const statusResp = await fetch(`${API_BASE}/api/analyze-citations/${activeCitationRunId}`);
                const status = await statusResp.json();

                // Update the UI progress display
                document.getElementById("job-status-label").textContent = status.status.toUpperCase();
                document.getElementById("job-progress-text").textContent = status.progress;

                if (status.status === "running") {
                    document.getElementById("job-bar-fill").style.width = "50%";
                }

                if (status.status === "completed") {
                    // Job done — stop polling, hide the job card, render results table
                    clearInterval(citationPollInterval);
                    clearInterval(timer);
                    jobCard.classList.add("hidden");
                    renderCitationMatrix(status.result, status.csv_path);
                } else if (status.status === "failed") {
                    // Job failed — stop polling, show error alert
                    clearInterval(citationPollInterval);
                    clearInterval(timer);
                    alert(`Citation analysis job failed:\n${status.error}`);
                    jobCard.classList.add("hidden");
                }
            }, 1500);

        } catch (err) {
            jobCard.classList.add("hidden");
            alert(`Failed to start citation analysis: ${err.message || err}`);
        }
    }

    /**
     * Render the citation classification results as an HTML table.
     * @param {Array} records - Array of classified citation row objects.
     * @param {string} csvFilename - Filename of the generated CSV report for the download link.
     */
    function renderCitationMatrix(records, csvFilename) {
        const card = document.getElementById("citation-matrix-card");
        const tbody = document.getElementById("citation-table-body");
        const dlBtn = document.getElementById("btn-download-report");

        tbody.innerHTML = "";
        // Wire up the CSV download link
        dlBtn.href = `${API_BASE}/api/reports/download/${csvFilename}`;

        if (!records || records.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" class="text-center">No classified citation contexts were recovered. (Verify if open-access papers exist citing this paper).</td></tr>`;
            card.classList.remove("hidden");
            fetchReports(); // Refresh the report history panel
            return;
        }

        records.forEach(row => {
            const tr = document.createElement("tr");

            // The CSS class matches the classification category (supporting, contrasting, etc.)
            const pillColor = row.classification.toLowerCase();

            tr.innerHTML = `
                <td class="citing-paper-cell">${escapeHTML(row.citing_title)}</td>
                <td>${row.year}</td>
                <td class="passage-cell">"${escapeHTML(row.passage)}"</td>
                <td><span class="class-pill ${pillColor}">${row.classification}</span></td>
                <td class="rationale-cell">${escapeHTML(row.rationale)}</td>
            `;
            tbody.appendChild(tr);
        });

        card.classList.remove("hidden");
        fetchReports(); // Refresh the report history panel to include the new report
    }

    /**
     * Fetch the list of saved CSV reports from /api/reports and render them as cards.
     */
    async function fetchReports() {
        const grid = document.getElementById("reports-grid-list");
        try {
            const resp = await fetch(`${API_BASE}/api/reports?ts=${Date.now()}`);
            const reports = await resp.json();

            if (reports.length === 0) {
                grid.innerHTML = `<div class="list-empty">No reports saved yet. Run an analysis above.</div>`;
                return;
            }

            grid.innerHTML = "";
            reports.forEach(r => {
                const card = document.createElement("div");
                card.className = "report-file-card";

                // Format the creation timestamp for human-readable display
                const createdDate = new Date(r.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });

                card.innerHTML = `
                    <div class="report-info">
                        <span class="report-name" title="${r.filename}">${r.filename}</span>
                        <span class="report-meta">${formatBytes(r.size_bytes)} | Created: ${createdDate}</span>
                    </div>
                    <div style="display:flex; gap:6px; align-items:center; flex-shrink:0;">
                        <a class="btn btn-secondary btn-icon" href="${API_BASE}/api/reports/download/${r.filename}" download title="Download CSV">
                            <i class="fa-solid fa-download"></i>
                        </a>
                        <button class="btn-delete-report btn-icon" data-filename="${r.filename}" title="Delete Report" style="padding:6px 8px; border-radius:6px; color:var(--accent-crimson); opacity:0.75; transition:opacity 0.2s, transform 0.2s;">
                            <i class="fa-solid fa-trash-can"></i>
                        </button>
                    </div>
                `;
                grid.appendChild(card);
            });

            // Wire up delete buttons for each report card
            grid.querySelectorAll(".btn-delete-report").forEach(btn => {
                btn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    const filename = btn.getAttribute("data-filename");
                    const reportCard = btn.closest(".report-file-card");
                    btn.disabled = true;
                    btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i>`;
                    try {
                        const response = await fetch(`${API_BASE}/api/reports/${encodeURIComponent(filename)}`, {
                            method: "DELETE",
                        });
                        if (!response.ok) {
                            const errData = await response.json().catch(() => ({}));
                            alert(`Failed to delete report: ${errData.detail || response.statusText || 'Unknown error'}`);
                        } else {
                            // Optimistic UI remove so deletion is immediately visible.
                            if (reportCard) reportCard.remove();
                        }
                    } catch (err) {
                        alert(`Network error deleting report: ${err.message}`);
                    } finally {
                        await fetchReports(); // Refresh the report list
                    }
                });
            });

        } catch (err) {
            grid.innerHTML = `<div class="list-empty text-crimson">Failed to load reports.</div>`;
        }
    }


    /* ==========================================================================
       TAB 4: PROMPT TEMPLATES
       ========================================================================== */

    /**
     * Fetch prompt templates from /api/prompts and render them as info cards.
     * Also populates the RAG template dropdown in the Knowledge Base tab.
     */
    async function fetchPrompts() {
        const container = document.getElementById("prompts-cards-container");
        const select = document.getElementById("rag-template-select");

        try {
            const resp = await fetch(`${API_BASE}/api/prompts`);
            const prompts = await resp.json();

            if (prompts.length === 0) {
                container.innerHTML = `<div class="placeholder-card glass-card"><h3>No Templates Found</h3><p>Ensure prompts/ folder has .txt prompt files.</p></div>`;
                return;
            }

            container.innerHTML = "";
            // Reset the dropdown to just the default "Standard Chat RAG" option
            select.innerHTML = `<option value="">Standard Chat RAG</option>`;

            prompts.forEach(p => {
                const opt = document.createElement("option");
                opt.value = p.name;
                opt.textContent = p.title;
                select.appendChild(opt);

                const card = document.createElement("div");
                card.className = "prompt-card glass-card";
                const deleteBtn = p.protected
                    ? ""
                    : `<button class="btn btn-secondary btn-delete-prompt" data-name="${p.name}" title="Delete template"><i class="fa-solid fa-trash"></i></button>`;

                card.innerHTML = `
                    <div class="prompt-card-top">
                        <span class="prompt-tag">${p.protected ? "Built-in" : "Custom"}</span>
                        <h3>${escapeHTML(p.title)}</h3>
                        <p class="prompt-desc">${escapeHTML(p.description)}</p>
                        <code class="prompt-filename">${p.name}.txt</code>
                    </div>
                    <div class="prompt-actions">
                        <button class="btn btn-secondary btn-view-prompt" data-name="${p.name}"><i class="fa-solid fa-eye"></i> View</button>
                        <button class="btn btn-secondary btn-edit-prompt" data-name="${p.name}"><i class="fa-solid fa-pen"></i> Edit</button>
                        <button class="btn btn-primary btn-use-prompt" data-name="${p.name}"><i class="fa-solid fa-comments"></i> Use</button>
                        ${deleteBtn}
                    </div>
                `;

                card.querySelector(".btn-view-prompt").addEventListener("click", () => showPromptModal(p));
                card.querySelector(".btn-edit-prompt").addEventListener("click", () => loadPromptIntoEditor(p.name));
                card.querySelector(".btn-use-prompt").addEventListener("click", () => {
                    select.value = p.name;
                    updateRagTemplateUi();
                    document.getElementById("nav-rag-btn").click();
                });
                const del = card.querySelector(".btn-delete-prompt");
                if (del) del.addEventListener("click", () => deletePromptTemplate(p.name));

                container.appendChild(card);
            });

            updateRagTemplateUi();

        } catch (err) {
            container.innerHTML = `<div class="placeholder-card glass-card text-crimson"><h3>Failed to load prompts</h3></div>`;
        }
    }

    /**
     * Show a modal dialog displaying the full content of a prompt template.
     * The modal closes when clicking the X button or anywhere outside the dialog.
     * @param {Object} prompt - Prompt object with title and content fields.
     */
    /**
     * Load a template from the API into the in-app editor for viewing or editing.
     */
    async function loadPromptIntoEditor(name) {
        try {
            const resp = await fetch(`${API_BASE}/api/prompts/${encodeURIComponent(name)}`);
            if (!resp.ok) throw new Error("Could not load template");
            const data = await resp.json();
            editingPromptName = data.name;
            document.getElementById("pe-name").value = data.name;
            document.getElementById("pe-name").disabled = true;
            document.getElementById("pe-title").value = data.title;
            document.getElementById("pe-system").value = data.system_body || "";
            document.getElementById("pe-user").value = data.user_template || "{context}";
            document.getElementById("pe-overwrite").checked = true;
            document.getElementById("prompt-editor-panel")?.scrollIntoView({ behavior: "smooth" });
        } catch (err) {
            alert("Failed to load template: " + err.message);
        }
    }

    function clearPromptEditorForm() {
        editingPromptName = null;
        document.getElementById("pe-name").disabled = false;
        document.getElementById("pe-name").value = "";
        document.getElementById("pe-title").value = "";
        document.getElementById("pe-system").value = "";
        document.getElementById("pe-user").value = "{context}\n\nResearcher query: {query}";
        document.getElementById("pe-overwrite").checked = true;
        const status = document.getElementById("prompt-save-status");
        if (status) status.classList.add("hidden");
    }

    /**
     * Save a new or updated prompt template via POST /api/prompts.
     */
    async function handlePromptSave(e) {
        e.preventDefault();
        const statusEl = document.getElementById("prompt-save-status");
        const body = {
            name: document.getElementById("pe-name").value.trim(),
            display_title: document.getElementById("pe-title").value.trim(),
            system_body: document.getElementById("pe-system").value.trim(),
            user_template: document.getElementById("pe-user").value.trim(),
            overwrite: document.getElementById("pe-overwrite").checked,
        };

        try {
            const resp = await fetch(`${API_BASE}/api/prompts`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || "Save failed");

            statusEl.textContent = data.message || "Template saved.";
            statusEl.className = "prompt-save-status success";
            statusEl.classList.remove("hidden");
            await fetchPrompts();
            if (!editingPromptName) clearPromptEditorForm();
        } catch (err) {
            statusEl.textContent = err.message;
            statusEl.className = "prompt-save-status error";
            statusEl.classList.remove("hidden");
        }
    }

    /**
     * Delete a custom prompt template (built-in templates cannot be deleted).
     */
    async function deletePromptTemplate(name) {
        if (!confirm(`Delete prompt template "${name}"? This cannot be undone.`)) return;
        try {
            const resp = await fetch(`${API_BASE}/api/prompts/${encodeURIComponent(name)}`, {
                method: "DELETE",
            });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || "Delete failed");
            await fetchPrompts();
        } catch (err) {
            alert("Delete failed: " + err.message);
        }
    }

    function showPromptModal(prompt) {
        const modal = document.createElement("div");
        modal.className = "prompt-modal";
        modal.innerHTML = `
            <div class="modal-content glass-card">
                <div class="modal-header">
                    <h3>${prompt.title} — System Prompt Template</h3>
                    <button class="btn-icon modal-close-btn"><i class="fa-solid fa-xmark"></i></button>
                </div>
                <div class="modal-body">
                    <div class="prompt-preview-block">${escapeHTML(prompt.content)}</div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Close the modal on X button click or background click
        const close = () => modal.remove();
        modal.querySelector(".modal-close-btn").addEventListener("click", close);
        modal.addEventListener("click", (e) => {
            if (e.target === modal) close();
        });
    }

});