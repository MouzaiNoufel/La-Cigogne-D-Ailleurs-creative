/* =========================================================
   La Cigogne D'Ailleurs — Creative
   Catalogue de meubles (vues de dessus, style "plan")
   Chaque item : nom, dimensions réelles (m), couleur, SVG.
   ========================================================= */

function shade(hex, pct) {
  const n = parseInt(hex.slice(1), 16);
  const amt = Math.round(2.55 * pct);
  const r = Math.min(255, Math.max(0, (n >> 16) + amt));
  const g = Math.min(255, Math.max(0, ((n >> 8) & 0xff) + amt));
  const b = Math.min(255, Math.max(0, (n & 0xff) + amt));
  return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

function svgUrl(svg) {
  return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
}

const MAKERS = {
  sofa(c) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 95">
      <rect x="2" y="2" width="216" height="91" rx="14" fill="${c}"/>
      <rect x="2" y="2" width="26" height="91" rx="12" fill="${shade(c, -18)}"/>
      <rect x="192" y="2" width="26" height="91" rx="12" fill="${shade(c, -18)}"/>
      <rect x="32" y="10" width="80" height="75" rx="10" fill="${shade(c, 14)}"/>
      <rect x="116" y="10" width="80" height="75" rx="10" fill="${shade(c, 14)}"/>
    </svg>`;
  },
  armchair(c) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 110 100">
      <rect x="2" y="2" width="106" height="96" rx="16" fill="${c}"/>
      <rect x="2" y="2" width="24" height="96" rx="12" fill="${shade(c, -18)}"/>
      <rect x="84" y="2" width="24" height="96" rx="12" fill="${shade(c, -18)}"/>
      <rect x="28" y="12" width="54" height="76" rx="10" fill="${shade(c, 14)}"/>
    </svg>`;
  },
  bed(c) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 200">
      <rect x="3" y="3" width="154" height="194" rx="10" fill="${shade(c, -10)}"/>
      <rect x="12" y="12" width="136" height="40" rx="9" fill="#f4f1ea"/>
      <rect x="12" y="58" width="136" height="130" rx="9" fill="${c}"/>
      <line x1="80" y1="66" x2="80" y2="182" stroke="${shade(c, -22)}" stroke-width="4"/>
    </svg>`;
  },
  table(c) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 70">
      <rect x="3" y="3" width="114" height="64" rx="10" fill="${shade(c, -12)}"/>
      <rect x="10" y="10" width="100" height="50" rx="7" fill="${c}"/>
    </svg>`;
  },
  chair(c) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60">
      <rect x="4" y="4" width="52" height="52" rx="8" fill="${shade(c, -14)}"/>
      <rect x="10" y="10" width="40" height="40" rx="6" fill="${c}"/>
    </svg>`;
  },
  lamp(c) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60">
      <circle cx="30" cy="30" r="26" fill="${shade(c, -25)}" opacity="0.35"/>
      <circle cx="30" cy="30" r="17" fill="${c}"/>
      <circle cx="30" cy="30" r="8" fill="${shade(c, 25)}"/>
    </svg>`;
  },
  plant(c) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 70 70">
      <circle cx="35" cy="32" r="24" fill="${c}"/>
      <circle cx="24" cy="24" r="12" fill="${shade(c, 14)}"/>
      <circle cx="46" cy="26" r="11" fill="${shade(c, 10)}"/>
      <circle cx="35" cy="62" r="6" fill="#8a5a33"/>
    </svg>`;
  },
  rug(c) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 120">
      <rect x="3" y="3" width="174" height="114" rx="14" fill="${c}"/>
      <rect x="14" y="14" width="152" height="92" rx="10" fill="none" stroke="${shade(c, 22)}" stroke-width="5"/>
      <rect x="30" y="30" width="120" height="60" rx="8" fill="none" stroke="${shade(c, -16)}" stroke-width="4"/>
    </svg>`;
  },
  tvstand(c) {
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 55">
      <rect x="3" y="3" width="194" height="49" rx="8" fill="${c}"/>
      <line x1="100" y1="6" x2="100" y2="49" stroke="${shade(c, -20)}" stroke-width="4"/>
      <rect x="12" y="10" width="76" height="35" rx="4" fill="${shade(c, 10)}"/>
      <rect x="112" y="10" width="76" height="35" rx="4" fill="${shade(c, 10)}"/>
    </svg>`;
  }
};

const CATALOG = [
  // h is the real-world display height used to normalize imported GLB models.
  // It deliberately lives with the catalog data rather than in the renderer.
  { id: "sofa",      name: "Canapé 3 places", w: 2.20, h: 0.86, d: 0.95, color: "#7f9bb3", make: MAKERS.sofa },
  { id: "armchair",  name: "Fauteuil",        w: 0.95, h: 0.88, d: 0.90, color: "#a98fbf", make: MAKERS.armchair },
  { id: "bed",       name: "Lit 160×200",     w: 1.60, h: 0.62, d: 2.00, color: "#c2b8a3", make: MAKERS.bed },
  { id: "table",     name: "Table basse",     w: 1.10, h: 0.48, d: 0.60, color: "#b08968", make: MAKERS.table },
  { id: "chair",     name: "Chaise",          w: 0.45, h: 0.82, d: 0.45, color: "#8fa58f", make: MAKERS.chair },
  { id: "lamp",      name: "Lampadaire",      w: 0.35, h: 1.58, d: 0.35, color: "#e0c36b", make: MAKERS.lamp },
  { id: "plant",     name: "Plante",          w: 0.45, h: 0.85, d: 0.45, color: "#5d8a5d", make: MAKERS.plant },
  { id: "rug",       name: "Tapis 160×110",   w: 1.60, h: 0.04, d: 1.10, color: "#c17f5e", make: MAKERS.rug },
  { id: "tvstand",   name: "Meuble TV",       w: 1.80, h: 0.55, d: 0.40, color: "#9c8469", make: MAKERS.tvstand },
];

// Pré-chargement des images
const CATALOG_IMGS = {};
CATALOG.forEach(item => {
  const img = new Image();
  img.src = svgUrl(item.make(item.color));
  CATALOG_IMGS[item.id] = img;
});

// Expose the catalog to ES modules (Phase 3).
window.CATALOG = CATALOG;
