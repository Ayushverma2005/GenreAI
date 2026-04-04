/* ============================================
   GenreAI — Main JavaScript
   Theme, GSAP animations, navbar, cursor glow
   ============================================ */

// ---------- Theme Toggle ----------
function toggleTheme() {
  const html = document.documentElement;
  html.classList.toggle('light');
  localStorage.setItem('theme', html.classList.contains('light') ? 'light' : 'dark');
}

// ---------- Mobile Menu ----------
function toggleMobileMenu() {
  document.getElementById('mobileMenu')?.classList.toggle('open');
  document.getElementById('mobileOverlay')?.classList.toggle('open');
}

// ---------- Cursor Glow ----------
(function () {
  const glow = document.getElementById('cursorGlow');
  if (!glow || window.innerWidth < 768) return;
  document.addEventListener('mousemove', e => {
    glow.style.left = e.clientX + 'px';
    glow.style.top = e.clientY + 'px';
  });
})();

// ---------- Navbar Scroll ----------
(function () {
  const nav = document.getElementById('navbar');
  if (!nav) return;
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 40);
  }, { passive: true });
})();

// ---------- GSAP ScrollTrigger Animations ----------
document.addEventListener('DOMContentLoaded', () => {
  if (typeof gsap === 'undefined' || typeof ScrollTrigger === 'undefined') {
    // Fallback: reveal without animation
    document.querySelectorAll('.reveal').forEach(el => {
      el.style.opacity = '1';
      el.style.transform = 'none';
    });
    return;
  }

  gsap.registerPlugin(ScrollTrigger);

  // Reveal animations
  gsap.utils.toArray('.reveal').forEach((el, i) => {
    gsap.fromTo(el,
      { opacity: 0, y: 50 },
      {
        opacity: 1, y: 0,
        duration: 0.8,
        delay: i % 3 * 0.15,
        ease: 'power3.out',
        scrollTrigger: {
          trigger: el,
          start: 'top 88%',
          toggleActions: 'play none none none'
        }
      }
    );
  });

  // Hero content stagger
  const heroText = document.querySelector('.hero-text');
  if (heroText) {
    gsap.fromTo(heroText.children,
      { opacity: 0, y: 40 },
      { opacity: 1, y: 0, duration: 0.9, stagger: 0.15, ease: 'power3.out', delay: 0.2 }
    );
  }

  // Hero visual card
  const heroCard = document.querySelector('.hero-visual-card');
  if (heroCard) {
    gsap.fromTo(heroCard,
      { opacity: 0, y: 60, scale: 0.95 },
      { opacity: 1, y: 0, scale: 1, duration: 1, ease: 'power3.out', delay: 0.5 }
    );
  }

  // Dashboard upload panel
  const uploadPanel = document.querySelector('.upload-panel');
  if (uploadPanel) {
    gsap.fromTo(uploadPanel,
      { opacity: 0, y: 50, scale: 0.96 },
      { opacity: 1, y: 0, scale: 1, duration: 0.9, ease: 'power3.out', delay: 0.3 }
    );
  }

  // Stat cards count-up
  gsap.utils.toArray('[data-count]').forEach(el => {
    const target = parseFloat(el.dataset.count);
    const obj = { val: 0 };
    gsap.to(obj, {
      val: target,
      duration: 2,
      ease: 'power2.out',
      scrollTrigger: { trigger: el, start: 'top 90%' },
      onUpdate: () => {
        el.textContent = obj.val.toFixed(1) + '%';
      }
    });
  });

  // Glass cards hover parallax
  document.querySelectorAll('.glass-card').forEach(card => {
    card.addEventListener('mousemove', e => {
      const rect = card.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width - 0.5;
      const y = (e.clientY - rect.top) / rect.height - 0.5;
      card.style.transform = `translateY(-4px) perspective(600px) rotateX(${-y * 4}deg) rotateY(${x * 4}deg)`;
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
    });
  });

  // Footer stagger
  const footerLinks = document.querySelectorAll('.footer-link');
  if (footerLinks.length) {
    gsap.fromTo(footerLinks,
      { opacity: 0, y: 15 },
      {
        opacity: 1, y: 0, duration: 0.5, stagger: 0.05, ease: 'power2.out',
        scrollTrigger: { trigger: '.footer', start: 'top 95%' }
      }
    );
  }

  // Gradient orbs gentle parallax on scroll
  gsap.utils.toArray('.hero-gradient-orb').forEach((orb, i) => {
    gsap.to(orb, {
      y: -80 * (i + 1),
      scrollTrigger: {
        trigger: '.hero, .features-hero',
        start: 'top top',
        end: 'bottom top',
        scrub: 1
      }
    });
  });

  // Features page — Metric count-up animation
  gsap.utils.toArray('[data-count-to]').forEach(el => {
    const target = parseFloat(el.dataset.countTo);
    const prefix = el.dataset.prefix || '';
    const suffix = el.dataset.suffix || '';
    const isDecimal = String(target).includes('.');
    const obj = { val: 0 };
    gsap.to(obj, {
      val: target,
      duration: 2.2,
      ease: 'power2.out',
      scrollTrigger: { trigger: el, start: 'top 90%' },
      onUpdate: () => {
        el.textContent = prefix + (isDecimal ? obj.val.toFixed(1) : Math.round(obj.val)) + suffix;
      }
    });
  });
});

