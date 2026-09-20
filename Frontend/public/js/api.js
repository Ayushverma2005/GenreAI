/* ============================================
   GenreAI — Centralized API Service
   Toast notifications, error handling
   ============================================ */

const GenreAPI = {
  BASE_URL: 'http://127.0.0.1:8000',

  async predictGenre(file) {
    const formData = new FormData();
    formData.append('file', file);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000); // 60s timeout

    try {
      const response = await fetch(`${this.BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });

      clearTimeout(timeout);

      if (!response.ok) {
        const errText = await response.text().catch(() => 'Unknown error');
        throw new Error(`Server error (${response.status}): ${errText}`);
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeout);
      if (error.name === 'AbortError') {
        throw new Error('Request timed out. Please try a shorter audio file.');
      }
      throw error;
    }
  },

  async healthCheck() {
    try {
      const res = await fetch(`${this.BASE_URL}/health`);
      return await res.json();
    } catch {
      return { status: 'unreachable', model_loaded: false };
    }
  }
};

// ---------- Toast Notifications ----------
function showToast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const icons = {
    error: 'error',
    success: 'check_circle',
    warning: 'warning',
    info: 'info'
  };

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span class="material-symbols-outlined">${icons[type] || 'info'}</span>
    <span>${message}</span>
  `;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(40px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}
