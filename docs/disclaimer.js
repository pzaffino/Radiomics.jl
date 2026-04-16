/**
 * disclaimer.js
 * Inject and manage the disclaimer banner on all pages.
 * Uses sessionStorage: the user sees the banner only once per session.
 * Includes: "Learn more" modal, animation, accessibility (ESC to close).
 */

(function () {
    'use strict';

    const STORAGE_KEY = 'disclaimer-accepted';

    // Already accepted in this session, exit immediately
    if (sessionStorage.getItem(STORAGE_KEY) === 'true') return;

    const style = document.createElement('style');
    style.textContent = `
    #disclaimer-banner {
      position: fixed;
      bottom: 0;
      left: 0;
      right: 0;
      background: #0f172a;
      border-top: 3px solid var(--primary, #6366f1);
      padding: 18px 28px;
      display: flex;
      align-items: center;
      gap: 20px;
      flex-wrap: wrap;
      z-index: 9999;
      animation: disc-slideUp 0.4s ease;
      box-shadow: 0 -4px 24px rgba(0,0,0,0.4);
    }

    @keyframes disc-slideUp {
      from { transform: translateY(100%); opacity: 0; }
      to   { transform: translateY(0);    opacity: 1; }
    }

    #disclaimer-banner .disc-text {
      flex: 1;
      min-width: 200px;
    }

    #disclaimer-banner .disc-text strong.disc-title {
      display: block;
      color: #ffffff;
      font-size: 0.88rem;
      font-weight: 600;
      margin-bottom: 4px;
    }

    #disclaimer-banner .disc-text p {
      color: rgba(255,255,255,0.65);
      font-size: 0.78rem;
      line-height: 1.5;
      margin: 0;
    }

    #disclaimer-banner .disc-actions {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-shrink: 0;
    }

    #btn-disc-accept {
      background: var(--primary, #6366f1);
      color: white;
      border: none;
      padding: 9px 20px;
      border-radius: 8px;
      font-family: inherit;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s, transform 0.1s;
      white-space: nowrap;
    }

    #btn-disc-accept:hover  { background: var(--primary-dark, #4f46e5); }
    #btn-disc-accept:active { transform: scale(0.97); }

    #btn-disc-more {
      color: rgba(255,255,255,0.5);
      font-size: 0.78rem;
      text-decoration: underline;
      background: none;
      border: none;
      cursor: pointer;
      font-family: inherit;
      white-space: nowrap;
      transition: color 0.2s;
    }

    #btn-disc-more:hover { color: white; }

    /* ── Modal ────────────────────────────────────────────────────────────── */
    #disc-modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.7);
      z-index: 10000;
      display: flex;
      align-items: center;
      justify-content: center;
      animation: disc-fadeIn 0.2s ease;
    }

    @keyframes disc-fadeIn {
      from { opacity: 0; }
      to   { opacity: 1; }
    }

    #disc-modal {
      background: #1e293b;
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 16px;
      max-width: 560px;
      width: 90%;
      padding: 36px;
      color: white;
      font-family: inherit;
      box-shadow: 0 24px 64px rgba(0,0,0,0.6);
      position: relative;
    }

    #disc-modal h2 {
      margin: 0 0 16px;
      font-size: 1.1rem;
      font-weight: 700;
      color: #fff;
      line-height: 1.4;
    }

    #disc-modal p, #disc-modal li {
      font-size: 0.85rem;
      color: rgba(255,255,255,0.7);
      line-height: 1.7;
      margin: 0 0 12px;
    }

    #disc-modal ul {
      padding-left: 20px;
      margin: 0 0 16px;
    }

    #disc-modal li { margin-bottom: 6px; }

    #disc-modal .modal-tag {
      display: inline-block;
      background: rgba(99,102,241,0.2);
      color: #a5b4fc;
      border: 1px solid rgba(99,102,241,0.4);
      border-radius: 6px;
      padding: 2px 8px;
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      margin-bottom: 16px;
    }

    #btn-disc-close-modal {
      position: absolute;
      top: 16px;
      right: 16px;
      background: none;
      border: none;
      color: rgba(255,255,255,0.4);
      font-size: 1.4rem;
      cursor: pointer;
      line-height: 1;
      transition: color 0.2s;
    }

    #btn-disc-close-modal:hover { color: white; }

    #btn-disc-modal-accept {
      display: block;
      width: 100%;
      margin-top: 20px;
      background: var(--primary, #6366f1);
      color: white;
      border: none;
      padding: 11px;
      border-radius: 10px;
      font-family: inherit;
      font-size: 0.88rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }

    #btn-disc-modal-accept:hover { background: var(--primary-dark, #4f46e5); }
  `;
    document.head.appendChild(style);

    // Banner HTML 
    const banner = document.createElement('div');
    banner.id = 'disclaimer-banner';
    banner.setAttribute('role', 'alertdialog');
    banner.setAttribute('aria-labelledby', 'disc-title');
    banner.innerHTML = `
    <div style="font-size: 22px; flex-shrink: 0;" aria-hidden="true">⚠️</div>
    <div class="disc-text">
      <strong class="disc-title" id="disc-title">Research use only — not a medical device</strong>
      <p>
        Radiomics.jl is a research software package and is <strong style="color:white;">not a medical device</strong>.
        Not intended for diagnostic, therapeutic, or clinical use. Not approved under EU MDR 2017/745 or equivalent.
        Results must not be used to inform clinical decisions.
      </p>
    </div>
    <div class="disc-actions">
      <button id="btn-disc-accept">I understand</button>
      <button id="btn-disc-more">Learn more</button>
    </div>
  `;

    // Modal HTML
    const modalOverlay = document.createElement('div');
    modalOverlay.id = 'disc-modal-overlay';
    modalOverlay.setAttribute('role', 'dialog');
    modalOverlay.setAttribute('aria-modal', 'true');
    modalOverlay.setAttribute('aria-labelledby', 'disc-modal-title');
    modalOverlay.innerHTML = `
    <div id="disc-modal">
      <button id="btn-disc-close-modal" aria-label="Close">×</button>
      <span class="modal-tag">REGULATORY NOTICE</span>
      <h2 id="disc-modal-title">Radiomics.jl — Research Use Only</h2>
      <p>
        This software is provided exclusively for <strong style="color:white;">academic and research purposes</strong>.
        It has not been evaluated, cleared, or approved as a medical device by any regulatory authority.
      </p>
      <ul>
        <li>Not approved or certified under <strong style="color:white;">EU MDR 2017/745</strong></li>
        <li>Not cleared by the <strong style="color:white;">FDA</strong> (21 CFR Part 820 or 510(k))</li>
        <li>Not compliant with <strong style="color:white;">IVDR 2017/746</strong></li>
        <li>Must not be used to support or replace clinical decision-making</li>
        <li>Results are for investigational and exploratory analysis only</li>
      </ul>
      <p>
        By continuing, you confirm that you are using this software solely for research,
        that you understand its limitations, and that you take full responsibility
        for any use of its outputs.
      </p>
      <button id="btn-disc-modal-accept">I understand — continue to the site</button>
    </div>
  `;

    // Inject in the DOM after loading
    function inject() {
        document.body.prepend(banner);

        // Button "I understand" (banner)
        document.getElementById('btn-disc-accept').addEventListener('click', accept);

        // Button "Learn more" → apre modal
        document.getElementById('btn-disc-more').addEventListener('click', openModal);
    }

    // Functions
    function accept() {
        sessionStorage.setItem(STORAGE_KEY, 'true');
        banner.style.animation = 'disc-slideUp 0.3s ease reverse';
        setTimeout(() => banner.remove(), 280);
        closeModal();
    }

    function openModal() {
        document.body.appendChild(modalOverlay);

        document.getElementById('btn-disc-close-modal').addEventListener('click', closeModal);
        document.getElementById('btn-disc-modal-accept').addEventListener('click', accept);

        // Close by clicking outside the modal
        modalOverlay.addEventListener('click', (e) => {
            if (e.target === modalOverlay) closeModal();
        });

        // Close with ESC
        document.addEventListener('keydown', onEsc);
    }

    function closeModal() {
        const overlay = document.getElementById('disc-modal-overlay');
        if (overlay) overlay.remove();
        document.removeEventListener('keydown', onEsc);
    }

    function onEsc(e) {
        if (e.key === 'Escape') closeModal();
    }

    // Start
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', inject);
    } else {
        inject();
    }

})();