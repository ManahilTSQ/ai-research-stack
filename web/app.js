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

    // Initialize SPA tabs routing
    initTabs();

    // Start health check polling (every 15 seconds)
    checkHealth();
    setInterval(checkHealth, 15000);

    // Initial data loading on page load
    fetchLocalPDFs();
    fetchPrompts();
    fetchReports();

    // Register all event listeners for forms and buttons
    document.getElementById("paper-search-form").addEventListener("submit", handlePaperSearch);
    document.getElementById("rag-query-form").addEventListener("submit", handleRAGQuery);
    document.getElementById("btn-sync-pdfs").addEventListener("click", fetchLocalPDFs);
    document.getElementById("btn-scan-pending").addEventListener("click", handleScanPending);
    document.getElementById("citation-analysis-form").addEventListener("submit", handleCitationAnalysis);
    document.getElementById("close-sources-btn").addEventListener("click", () => {
        document.getElementById("retrieved-sources-panel").classList.add("hidden");
    });

    // Sync the RAG context-limit slider label with the slider value in real time
    const limitSlider = document.getElementById("rag-limit-slider");
    const limitLabel = document.getElementById("lbl-rag-limit");
    limitSlider.addEventListener("input", (e) => {
        limitLabel.textContent = `Context: ${e.target.value} chunks`;
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

        navButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                const targetTab = btn.getAttribute("data-tab");

                // Deactivate all tabs and nav buttons
                navButtons.forEach(b => b.classList.remove("active"));
                panes.forEach(p => p.classList.remove("active"));

                // Activate the selected tab
                btn.classList.add("active");
                document.getElementById(targetTab).classList.add("active");

                // Refresh relevant data when switching tabs
                if (targetTab === "tab-rag") {
                    fetchLocalPDFs();  // Keep manifest fresh when entering KnowledgeBase tab
                } else if (targetTab === "tab-citations") {
                    fetchReports();    // Refresh report history when entering Citation Analysis tab
                }
            });
        });
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
        const limit = document.getElementById("search-limit").value;
        const statusBox = document.getElementById("search-status-message");
        const resultsList = document.getElementById("search-results-list");

        if (!query) return;

        // Show loading spinner while waiting for results
        statusBox.classList.remove("hidden");
        resultsList.innerHTML = "";

        try {
            const resp = await fetch(`${API_BASE}/api/search?q=${encodeURIComponent(query)}&limit=${limit}`);
            const papers = await resp.json();
            statusBox.classList.add("hidden");

            if (papers.length === 0) {
                resultsList.innerHTML = `
                    <div class="placeholder-card glass-card">
                        <i class="fa-solid fa-face-frown placeholder-icon"></i>
                        <h3>No Papers Found</h3>
                        <p>We couldn't find any papers matching your search query. Try broadening your keywords.</p>
                    </div>
                `;
                return;
            }

            // Render a card for each paper result
            papers.forEach(paper => {
                resultsList.appendChild(createPaperCard(paper));
            });

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
                <strong>Abstract summary:</strong> ${paper.abstract ? paper.abstract : "No abstract snippet indexed."}
            </div>

            <div class="paper-footer">
                <span class="report-meta">DOI: ${doiLabel} | arXiv: ${paper.arxiv}</span>
                <button class=\"btn btn-primary btn-download-ingest\" id=\"btn-ingest-${paper.paperId}\">
                    <i class=\"fa-solid fa-cloud-arrow-down\"></i> ${paper.has_pdf ? 'Download & Ingest PDF' : 'Ingest Abstract Only'}
                </button>
            </div>
        `;

        // Wire up the ingestion button click handler
        const btn = card.querySelector(`.btn-download-ingest`);
        btn.addEventListener("click", () => triggerIngestion(paper, btn));

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
        button.disabled = true;
        button.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Downloading PDF...`;

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
                    citationCount: paper.citationCount
                })
            });

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

                item.innerHTML = `
                    <div class="file-item-main" title="${file.title}">
                        <div class="file-name-row">${file.title}</div>
                        <div class="file-meta-row">
                            <span>${formatBytes(file.size_bytes)}</span>
                            ${statusBadge}
                        </div>
                    </div>
                    <button class="btn-delete-file btn-icon" data-filename="${file.filename}" title="Delete Paper">
                        <i class="fa-solid fa-trash-can text-crimson"></i>
                    </button>
                `;
                listDiv.appendChild(item);
            });

            // Wire up deletion for each button
            listDiv.querySelectorAll(".btn-delete-file").forEach(btn => {
                btn.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    const filename = btn.getAttribute("data-filename");
                    if (confirm(`Are you sure you want to delete this paper and remove all its vector chunks from the database?\nFile: ${filename}`)) {
                        try {
                            btn.disabled = true;
                            btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i>`;
                            const deleteResp = await fetch(`${API_BASE}/api/papers/${filename}`, {
                                method: "DELETE"
                            });
                            const delResult = await deleteResp.json();
                            if (delResult.success) {
                                alert("Paper successfully deleted.");
                                fetchLocalPDFs();
                                checkHealth();
                            } else {
                                alert("Failed to fully delete paper chunks. Check backend logs.");
                                fetchLocalPDFs();
                            }
                        } catch (delErr) {
                            alert(`Error deleting paper: ${delErr.message || delErr}`);
                            fetchLocalPDFs();
                        }
                    }
                });
            });

        } catch (err) {
            listDiv.innerHTML = `<div class="list-empty text-crimson">Failed to load manifest.</div>`;
        }
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

            if (result.success) {
                alert("Bulk folder scanning and ingestion started in the background.\nThe document list and database stats will update automatically as PDFs are processed.");
            } else {
                alert("Scan failed to initiate. Please check backend logs.");
            }

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
            alert(`Scan error: ${err.message || err}`);
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
            const resp = await fetch(`${API_BASE}/api/query-rag`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: query,
                    limit: parseInt(limit),
                    prompt_template: template ? template : null
                })
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
            document.getElementById(`btn-show-src-${bubbleId}`).addEventListener("click", () => {
                showRetrievedSourcesPanel(sources);
            });
        }

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

        sources.forEach((c, idx) => {
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
                <div class="source-chunk-title">[Source ${idx + 1}] "${meta.title}" (${pages})</div>
                <div class="source-chunk-text">"${escapeHTML(c.text)}"</div>
            `;
            list.appendChild(item);

            // Scroll highlighted source into view
            if (highlightIdx !== null && highlightIdx === idx) {
                item.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }
        });
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

        // Citation reference links: [Source N] → clickable anchor
        html = html.replace(/\[Source\s+(\d+)\]/gi, (match, num) => {
            return `<a class="citation-ref" href="#" data-source-index="${num}">Source ${num}</a>`;
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
            const resp = await fetch(`${API_BASE}/api/reports`);
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
                    <a class="btn btn-secondary btn-icon" href="${API_BASE}/api/reports/download/${r.filename}" download>
                        <i class="fa-solid fa-download"></i>
                    </a>
                `;
                grid.appendChild(card);
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
                // Add option to the RAG template dropdown
                const opt = document.createElement("option");
                opt.value = p.name;
                opt.textContent = p.title;
                select.appendChild(opt);

                // Build the prompt card in the Prompts tab grid
                const card = document.createElement("div");
                card.className = "prompt-card glass-card";

                card.innerHTML = `
                    <div class="prompt-card-top">
                        <span class="prompt-tag">System Prompt</span>
                        <h3>${p.title}</h3>
                        <p class="prompt-desc">${p.description}</p>
                    </div>
                    <div class="prompt-actions">
                        <button class="btn btn-secondary btn-view-prompt" data-name="${p.name}"><i class="fa-solid fa-eye"></i> View</button>
                        <button class="btn btn-primary btn-use-prompt" data-name="${p.name}"><i class="fa-solid fa-comments"></i> Use Template</button>
                    </div>
                `;

                // "View" opens a modal with the full raw prompt content
                card.querySelector(".btn-view-prompt").addEventListener("click", () => {
                    showPromptModal(p);
                });

                // "Use Template" sets the dropdown and switches to the RAG tab
                card.querySelector(".btn-use-prompt").addEventListener("click", () => {
                    select.value = p.name;
                    document.getElementById("nav-rag-btn").click();
                });

                container.appendChild(card);
            });

        } catch (err) {
            container.innerHTML = `<div class="placeholder-card glass-card text-crimson"><h3>Failed to load prompts</h3></div>`;
        }
    }

    /**
     * Show a modal dialog displaying the full content of a prompt template.
     * The modal closes when clicking the X button or anywhere outside the dialog.
     * @param {Object} prompt - Prompt object with title and content fields.
     */
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