/* ============================================
   GenreAI — Dashboard Logic
   File upload, drag-drop, prediction, results
   ============================================ */

(function () {
  'use strict';

  const CONFIG = {
    MAX_FILE_SIZE: 100 * 1024 * 1024,
    ALLOWED_EXTENSIONS: ['.mp3', '.wav', '.ogg'],
    ALLOWED_MIME_TYPES: ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/mp3']
  };

  const el = {
    fileInput: document.getElementById('audioFileInput'),
    dropZone: document.getElementById('dropZone'),
    browseBtn: document.getElementById('browseBtn'),
    dropZoneTitle: document.getElementById('dropZoneTitle'),
    dropZoneSubtitle: document.getElementById('dropZoneSubtitle'),
    statusMessage: document.getElementById('statusMessage'),
    resultsContainer: document.getElementById('resultsContainer'),
    resultGenre: document.getElementById('resultGenre'),
    resultConfidence: document.getElementById('resultConfidence'),
    confidenceBar: document.getElementById('confidenceBar'),
    additionalGenres: document.getElementById('additionalGenres'),
    additionalGenresList: document.getElementById('additionalGenresList'),
    aiLoader: document.getElementById('aiLoader')
  };

  let selectedFile = null;
  let isProcessing = false;

  function init() {
    if (!el.dropZone || !el.fileInput) return;
    el.browseBtn?.addEventListener('click', e => { e.stopPropagation(); el.fileInput.click(); });
    el.dropZone.addEventListener('click', e => {
      if (e.target === el.browseBtn || el.browseBtn?.contains(e.target)) return;
      el.fileInput.click();
    });
    el.fileInput.addEventListener('change', e => { if (e.target.files[0]) processFile(e.target.files[0]); });

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt =>
      el.dropZone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); }, false)
    );
    el.dropZone.addEventListener('dragenter', () => el.dropZone.classList.add('drag-active'));
    el.dropZone.addEventListener('dragover', () => el.dropZone.classList.add('drag-active'));
    el.dropZone.addEventListener('dragleave', e => { if (e.target === el.dropZone) el.dropZone.classList.remove('drag-active'); });
    el.dropZone.addEventListener('drop', e => {
      el.dropZone.classList.remove('drag-active');
      if (e.dataTransfer.files.length) processFile(e.dataTransfer.files[0]);
    });
  }

  function processFile(file) {
    const v = validateFile(file);
    if (!v.valid) { showToast(v.message, 'error'); return; }
    selectedFile = file;
    el.dropZoneTitle.textContent = file.name;
    el.dropZoneSubtitle.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB — Ready to analyze`;
    el.statusMessage.textContent = 'File loaded, preparing analysis...';
    hideResults();
    predictGenre();
  }

  function validateFile(file) {
    const name = file.name.toLowerCase();
    const hasExt = CONFIG.ALLOWED_EXTENSIONS.some(ext => name.endsWith(ext));
    if (!hasExt && !CONFIG.ALLOWED_MIME_TYPES.includes(file.type))
      return { valid: false, message: 'Invalid file type. Please upload MP3, WAV, or OGG.' };
    if (file.size > CONFIG.MAX_FILE_SIZE)
      return { valid: false, message: `File too large. Max ${CONFIG.MAX_FILE_SIZE / 1024 / 1024}MB.` };
    return { valid: true };
  }

  async function predictGenre() {
    if (!selectedFile || isProcessing) return;
    isProcessing = true;
    showLoader();
    el.statusMessage.textContent = 'Analyzing audio patterns...';

    try {
      const data = await GenreAPI.predictGenre(selectedFile);
      displayResults(data);
      showToast('Analysis complete!', 'success');
    } catch (error) {
      console.error('Prediction error:', error);
      let msg = 'Prediction failed. ';
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        msg += 'Cannot connect to backend. Ensure FastAPI is running on port 8000.';
      } else {
        msg += error.message;
      }
      showToast(msg, 'error', 6000);
      el.statusMessage.textContent = 'Analysis failed';
      setTimeout(resetUI, 5000);
    } finally {
      isProcessing = false;
      hideLoader();
    }
  }

  function displayResults(data) {
    let genre = null, confidence = null, allPredictions = null;

    if (data.analysis) {
      genre = data.analysis.primary_genre || data.analysis.predicted_genre;
      confidence = data.analysis.confidence;
      allPredictions = data.analysis.all_predictions;
    } else {
      genre = data.predicted_genre || data.genre || data.primary_genre;
      confidence = data.confidence || data.probability;
      allPredictions = data.predictions || data.all_predictions;
    }

    genre = genre || 'Unknown';
    confidence = confidence ?? 0;

    el.resultGenre.textContent = capitalize(genre);
    el.resultConfidence.textContent = formatConf(confidence);
    el.statusMessage.textContent = 'Analysis complete!';
    el.dropZoneSubtitle.textContent = `Done: ${selectedFile.name}`;

    // Animate confidence bar
    if (el.confidenceBar) {
      el.confidenceBar.style.width = '0%';
      requestAnimationFrame(() => {
        el.confidenceBar.style.width = (confidence * 100) + '%';
      });
    }

    // Alternative predictions
    if (allPredictions && typeof allPredictions === 'object') {
      const arr = Array.isArray(allPredictions)
        ? allPredictions
        : Object.entries(allPredictions).map(([g, c]) => ({ genre: g, confidence: c }));

      const alts = arr
        .filter(p => (p.genre || '').toLowerCase() !== genre.toLowerCase())
        .sort((a, b) => (b.confidence || 0) - (a.confidence || 0))
        .slice(0, 3);

      if (alts.length && el.additionalGenresList) {
        el.additionalGenresList.innerHTML = '';
        alts.forEach(alt => {
          const badge = document.createElement('span');
          badge.className = 'alt-badge';
          badge.innerHTML = `${capitalize(alt.genre)} <span class="conf">${formatConf(alt.confidence)}</span>`;
          el.additionalGenresList.appendChild(badge);
        });
        el.additionalGenres.style.display = '';
      }
    }

    // Show results with GSAP if available
    el.resultsContainer.classList.add('visible');
    if (typeof gsap !== 'undefined') {
      gsap.fromTo(el.resultsContainer, { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.5, ease: 'power2.out' });
    }
  }

  function showLoader() {
    if (el.aiLoader) el.aiLoader.classList.add('visible');
    if (el.dropZone) el.dropZone.style.display = 'none';
  }
  function hideLoader() {
    if (el.aiLoader) el.aiLoader.classList.remove('visible');
    if (el.dropZone) el.dropZone.style.display = '';
  }
  function hideResults() {
    el.resultsContainer?.classList.remove('visible');
    if (el.additionalGenres) el.additionalGenres.style.display = 'none';
    if (el.confidenceBar) el.confidenceBar.style.width = '0%';
  }
  function resetUI() {
    el.dropZoneTitle.textContent = 'Drag to Analyze';
    el.dropZoneSubtitle.textContent = 'Upload lossless WAV or MP3 for highest classification precision';
    el.statusMessage.textContent = 'Listening for input...';
    selectedFile = null;
    el.fileInput.value = '';
    hideResults();
  }

  function capitalize(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1).toLowerCase() : 'Unknown'; }
  function formatConf(c) { return typeof c === 'number' ? (c * 100).toFixed(1) + '%' : c || 'N/A'; }

  init();
})();
