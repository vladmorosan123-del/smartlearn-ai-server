// ============================================================
// ai-rag.js — server AI LOCAL cu RAG peste materialele din Supabase
// MOD LOCAL (implicit): citeste materialele prin RPC public (anon),
//   descarca fisierele, face embeddings si tine indexul intr-un JSON local.
//   La intrebare, cauta in index (cosine) si raspunde din documente.
// Fara DB admin, fara login profesor, fara acces la server.
//
// Endpoints:
//   GET  /api/health
//   POST /api/ai/index-local   -> (re)construieste indexul local din materiale
//   POST /api/ai/ask           -> { question, subject? } -> { answer, sources }
//
// Porneste:  node ai-rag.js   (din folderul server/)
// .env are nevoie de: GEMINI_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY
// Doar pentru DEZVOLTARE LOCALA (fara auth).
// ============================================================

require('dotenv').config();

const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { GoogleGenAI } = require('@google/genai');
const mammoth = require('mammoth');
const { PDFParse } = require('pdf-parse');
let officeParser = null;
try { officeParser = require('officeparser'); } catch (e) { /* optional: pptx */ }

// Rezilienta: o eroare neprinsa doar se logheaza, NU opreste serverul
process.on('unhandledRejection', (e) => console.error('unhandledRejection:', (e && e.message) || e));
process.on('uncaughtException', (e) => console.error('uncaughtException:', (e && e.message) || e));

const app = express();
const PORT = process.env.PORT || process.env.AI_DEV_PORT || 3030;
const CHAT_MODEL = 'gemini-2.5-flash';
const EMBED_MODEL = 'gemini-embedding-001';
const EMBED_DIM = 768;

const SUPABASE_URL = (process.env.SUPABASE_URL || '').replace(/\/+$/, '');
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY || '';
const INDEX_FILE = path.join(__dirname, 'rag-index.json');

// CORS: in productie seteaza CORS_ORIGIN cu domeniul site-ului (ex. https://site.ro);
// local (nesetat) = permite orice origine.
const corsOrigins = (process.env.CORS_ORIGIN || '').split(',').map((s) => s.trim()).filter(Boolean);
app.use(cors(corsOrigins.length ? { origin: corsOrigins } : {}));
app.use(express.json({ limit: '2mb' }));

// Rate limiting simplu per-IP pentru /ask (anti-abuz al cotei Gemini)
const rlMap = new Map();
function rateLimit(req, res, next) {
  const ip = (req.headers['x-forwarded-for'] || '').split(',')[0].trim() || req.socket.remoteAddress || 'unknown';
  const now = Date.now();
  const WINDOW = 60000, MAX = Number(process.env.RATE_MAX || 30);
  let e = rlMap.get(ip);
  if (!e || now > e.reset) { e = { count: 0, reset: now + WINDOW }; rlMap.set(ip, e); }
  e.count++;
  if (e.count > MAX) {
    return res.json({ answer: 'Prea multe întrebări într-un minut. Te rog așteaptă puțin și încearcă din nou.', sources: [] });
  }
  next();
}

// Protectie pentru operatiile administrative (indexare/debug): daca ADMIN_TOKEN e setat, cere-l.
function requireAdmin(req, res, next) {
  const token = process.env.ADMIN_TOKEN;
  if (token && req.headers['x-admin-token'] !== token) return res.status(403).json({ error: 'Interzis' });
  next();
}

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// index local in memorie: [{ material_id, title, subject, category, content, embedding:[...] }]
let localIndex = [];
function loadIndex() {
  try {
    if (fs.existsSync(INDEX_FILE)) {
      localIndex = JSON.parse(fs.readFileSync(INDEX_FILE, 'utf8'));
      console.log(`Index local incarcat: ${localIndex.length} bucati`);
    }
  } catch (e) { console.error('nu pot citi indexul:', e.message); localIndex = []; }
}
loadIndex();

// ─── Helpers ───────────────────────────────────────────────
async function withRetry(fn, label) {
  let lastErr;
  for (let attempt = 1; attempt <= 4; attempt++) {
    try { return await fn(); }
    catch (err) {
      lastErr = err;
      const msg = String((err && err.message) || err);
      if (!/503|UNAVAILABLE|overloaded|high demand|429|RESOURCE_EXHAUSTED|quota/i.test(msg)) throw err;
      await new Promise((r) => setTimeout(r, 1200 * attempt));
    }
  }
  throw lastErr;
}

async function embedOne(text, taskType) {
  return withRetry(async () => {
    const res = await ai.models.embedContent({
      model: EMBED_MODEL, contents: text, config: { taskType, outputDimensionality: EMBED_DIM },
    });
    return res.embeddings[0].values;
  }, 'embed');
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

function chunkText(text, size = 1000, overlap = 150) {
  const clean = text.replace(/\s+\n/g, '\n').replace(/[ \t]{2,}/g, ' ').trim();
  const chunks = [];
  let i = 0;
  while (i < clean.length) {
    let end = Math.min(i + size, clean.length);
    if (end < clean.length) {
      const slice = clean.slice(i, end);
      const lastBreak = Math.max(slice.lastIndexOf('. '), slice.lastIndexOf('\n'));
      if (lastBreak > size * 0.5) end = i + lastBreak + 1;
    }
    const piece = clean.slice(i, end).trim();
    if (piece.length > 30) chunks.push(piece);
    const next = end - overlap;
    i = next > i ? next : end;
  }
  return chunks;
}

async function extractText(buf, hint) {
  const h = (hint || '').toLowerCase();
  if (h.includes('pdf')) {
    const parser = new PDFParse({ data: buf });
    const data = await parser.getText();
    return data.text || '';
  }
  if (h.includes('word') || h.includes('wordprocessingml') || (h.includes('docx'))) {
    const { value } = await mammoth.extractRawText({ buffer: buf });
    return value || '';
  }
  if (officeParser && (h.includes('powerpoint') || h.includes('presentation') || h.includes('pptx') || h.includes('.ppt'))) {
    try { return (await officeParser.parseOfficeAsync(buf)) || ''; } catch (e) { return ''; }
  }
  return '';
}

// Tipul MIME pentru OCR (imagini / pdf). null = OCR nu se aplica.
function mimeFor(hint) {
  const h = (hint || '').toLowerCase();
  if (h.includes('png')) return 'image/png';
  if (h.includes('jpg') || h.includes('jpeg')) return 'image/jpeg';
  if (h.includes('webp')) return 'image/webp';
  if (h.includes('pdf')) return 'application/pdf';
  return null;
}

// OCR pentru fisiere scanate/imagini: trimite fisierul la Gemini (multimodal) sa extraga textul.
async function geminiOcr(buf, mimeType) {
  const b64 = buf.toString('base64');
  return withRetry(async () => {
    const r = await ai.models.generateContent({
      model: CHAT_MODEL,
      contents: [
        { inlineData: { mimeType: mimeType || 'application/pdf', data: b64 } },
        { text: 'Extrage si returneaza TOT textul din acest document, inclusiv formulele scrise in text simplu. Returneaza doar textul, fara alte comentarii.' },
      ],
      config: { maxOutputTokens: 8192 },
    });
    return r.text || '';
  }, 'ocr');
}

async function generateWithRetry(contents, systemInstruction) {
  // gemini-2.5-flash-lite are cota gratuita separata -> fallback util cand flash e limitat
  const models = [CHAT_MODEL, 'gemini-2.5-flash-lite'];
  let lastErr;
  for (const model of models) {
    try {
      return await withRetry(async () => {
        const result = await ai.models.generateContent({ model, contents, config: { systemInstruction, maxOutputTokens: 8192 } });
        if (!result.text) throw new Error('raspuns gol');
        return result.text;
      }, 'gen');
    } catch (err) { lastErr = err; }
  }
  throw lastErr;
}

// Plasa de siguranta: converteste orice LaTeX scapat in text lizibil (Unicode).
// Astfel utilizatorul nu vede NICIODATA $, \frac, ^{...} etc.
const SUP = { '0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','+':'⁺','-':'⁻','(':'⁽',')':'⁾','n':'ⁿ','x':'ˣ','i':'ⁱ' };
const SUB = { '0':'₀','1':'₁','2':'₂','3':'₃','4':'₄','5':'₅','6':'₆','7':'₇','8':'₈','9':'₉','+':'₊','-':'₋','(':'₍',')':'₎',',':',' };
const toSuper = (s) => s.split('').map((c) => SUP[c] || c).join('');
const toSub = (s) => s.split('').map((c) => SUB[c] || c).join('');

function sanitizeMath(text) {
  if (!text) return text;
  // Protejeaza codul (```...``` si `inline`) de curatarea de "math",
  // altfel s-ar strica acoladele, \n-urile si indicii din cod (informatica).
  const codeBlocks = [];
  let t = text.replace(/```[\s\S]*?```/g, (m) => `@@C${codeBlocks.push(m) - 1}@@`);
  t = t.replace(/`[^`\n]*`/g, (m) => `@@C${codeBlocks.push(m) - 1}@@`);
  // scoate delimitatorii de math
  t = t.replace(/\$\$([\s\S]*?)\$\$/g, '$1').replace(/\$([^$\n]*?)\$/g, '$1');
  t = t.replace(/\\\[([\s\S]*?)\\\]/g, '$1').replace(/\\\(([\s\S]*?)\\\)/g, '$1');
  // \frac{a}{b} -> (a)/(b)  (repetat pentru cazuri simple imbricate)
  for (let i = 0; i < 4; i++) t = t.replace(/\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, '($1)/($2)');
  t = t.replace(/\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}/g, '$1√($2)');
  t = t.replace(/\\sqrt\s*\{([^{}]*)\}/g, '√($1)');
  // operatori si litere grecesti
  const MAP = { '\\times':'×','\\cdot':'·','\\div':'÷','\\pm':'±','\\mp':'∓','\\leq':'≤','\\le':'≤','\\geq':'≥','\\ge':'≥','\\neq':'≠','\\ne':'≠','\\approx':'≈','\\equiv':'≡','\\infty':'∞','\\Rightarrow':'⇒','\\rightarrow':'→','\\to':'→','\\Leftarrow':'⇐','\\Delta':'Δ','\\delta':'δ','\\pi':'π','\\alpha':'α','\\beta':'β','\\gamma':'γ','\\theta':'θ','\\lambda':'λ','\\mu':'μ','\\varphi':'φ','\\phi':'φ','\\omega':'ω','\\Omega':'Ω','\\sum':'Σ','\\prod':'Π','\\int':'∫','\\in':'∈','\\notin':'∉','\\forall':'∀','\\exists':'∃','\\cdots':'⋯','\\ldots':'…','\\dots':'…','\\cup':'∪','\\cap':'∩','\\subset':'⊂','\\emptyset':'∅' };
  for (const k in MAP) t = t.split(k).join(MAP[k]);
  // functii: pastreaza numele
  t = t.replace(/\\(lim|ln|log|sin|cos|tan|tg|ctg|cot|arcsin|arccos|arctan|max|min|exp|deg|mod)\b/g, '$1');
  // comenzi de spatiere/format -> sterge
  t = t.replace(/\\(left|right|big|Big|bigg|displaystyle|text|mathrm|mathbf|operatorname|quad|qquad)\b/g, '');
  t = t.replace(/\\[,;:! ]/g, ' ').replace(/\\\\/g, ' ');
  // exponenti si indici
  t = t.replace(/\^\{([^{}]+)\}/g, (m, g) => /^[-+0-9nxi()]+$/.test(g) ? toSuper(g) : '^(' + g + ')');
  t = t.replace(/\^(-?[0-9nxi])/g, (m, g) => toSuper(g));
  t = t.replace(/_\{([^{}]+)\}/g, (m, g) => /^[-+0-9,()]+$/.test(g) ? toSub(g) : '_(' + g + ')');
  t = t.replace(/_([0-9])/g, (m, g) => toSub(g));
  // ce a mai ramas cu backslash -> scoate backslash-ul
  t = t.replace(/\\([a-zA-Z]+)/g, '$1');
  // acolade ramase
  t = t.replace(/[{}]/g, '');
  // restaureaza blocurile de cod protejate
  t = t.replace(/@@C(\d+)@@/g, (m, i) => codeBlocks[Number(i)]);
  return t;
}

const BASE_RULES =
  'Esti un profesor rabdator care explica pe intelesul elevilor de liceu, la TOATE materiile de BAC: matematica, informatica, fizica, romana, chimie, biologie, istorie etc.\n' +
  'REGULI (obligatorii):\n' +
  '- Scrie in limba romana, clar si prietenos, ca pentru un elev care invata.\n' +
  '- EXPLICA MEREU: nu da doar rezultatul, ci si rationamentul — de ce faci fiecare pas, ce formula folosesti si de ce se aplica.\n' +
  '- La probleme complexe, imparte in pasi mici numerotati si comenteaza fiecare pas; daca ajuta, da o scurta intuitie.\n' +
  '- Defineste pe scurt termenii importanti cand apar prima data.\n' +
  '- Formatare: Markdown curat (**ingrosat** pentru termeni cheie, liste numerotate pentru pasi).\n' +
  '- MATEMATICA in TEXT SIMPLU cu simboluri Unicode normale. ESTE INTERZIS LaTeX: fara $, fara \\frac, \\sqrt, \\times, \\pm, fara acolade { } sau ^{ } _{ }.\n' +
  '  Scrie asa: fractii ca (a)/(b) sau a/b, puteri ca x² sau x^2, radical ca √(x), indici ca x₁, x₂, x₃. Foloseste simboluri: × ÷ ± √ π ² ³ ≤ ≥ ≠ ≈ → ∞ ∫ Σ Δ ∈.\n' +
  '  Exemple CORECTE: "x₁,₂ = (-b ± √Δ) / (2a)", "x₃ = -1/2 - i", "∫ x² dx = x³/3 + C", "lim (x→0) sin(x)/x = 1".\n' +
  '- FIZICA: la fiecare problema include unitatile de masura (m, s, kg, N, J, W, m/s²) si formula folosita, apoi calculul.\n' +
  '- INFORMATICA: scrie codul in blocuri Markdown cu limbajul specificat (```cpp, ```python sau ```pseudocod). Pentru BAC foloseste C++ sau pseudocod. Explica algoritmul pas cu pas si complexitatea unde e cazul. In blocurile de cod scrie cod normal (acolade, indentare, operatori) — acolo regulile de matematica NU se aplica.\n' +
  '- ROMANA: pentru eseuri/comentarii structureaza clar (introducere, cuprins cu argumente sustinute prin citate din text, incheiere) si foloseste termeni de teorie literara. Pentru gramatica: da regula + exemple.\n' +
  '- Verifica-ti rezultatul la final daca se poate (ex: inlocuind solutia).\n' +
  '- Termina cu o linie clara: "**Raspuns final:** ...".\n' +
  'Fii complet si didactic, fara text de umplutura.';

// ─── Routes ────────────────────────────────────────────────
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok', mode: 'ai-rag-local', model: CHAT_MODEL,
    hasKey: !!process.env.GEMINI_API_KEY,
    supabase: !!(SUPABASE_URL && SUPABASE_ANON_KEY),
    indexedChunks: localIndex.length,
  });
});

// Descarca un fisier; daca URL-ul brut da eroare (diacritice/spatii neencodate),
// reincearca cu encodeURI. Astfel recuperam fisierele care dadeau HTTP 400.
async function fetchFile(url) {
  let fr = await fetch(url);
  if (!fr.ok) {
    const enc = encodeURI(url);
    if (enc !== url) fr = await fetch(enc);
  }
  return fr;
}

app.post('/api/ai/index-local', requireAdmin, async (req, res) => {
  if (!SUPABASE_URL || !SUPABASE_ANON_KEY) return res.status(400).json({ error: 'SUPABASE_URL / SUPABASE_ANON_KEY lipsesc in .env' });
  const reindex = !!(req.body && req.body.reindex);
  try {
    const r = await fetch(`${SUPABASE_URL}/rest/v1/rpc/get_materials_for_students`, {
      method: 'POST',
      headers: { apikey: SUPABASE_ANON_KEY, Authorization: `Bearer ${SUPABASE_ANON_KEY}`, 'Content-Type': 'application/json' },
      body: '{}',
    });
    if (!r.ok) return res.status(502).json({ error: 'RPC materiale a esuat: HTTP ' + r.status });
    const materials = await r.json();

    // incremental: pastreaza ce e deja indexat, adauga doar ce lipseste (reindex=true reface tot)
    const baseIndex = reindex ? [] : [...localIndex];
    const haveIds = new Set(baseIndex.map((c) => c.material_id));
    const added = [];
    const skipped = [];
    let done = 0;
    for (const m of materials) {
      if (haveIds.has(m.id)) continue; // deja indexat, sarim
      if (!m.file_url) { skipped.push({ title: m.title, reason: 'fara file_url' }); continue; }
      const ft = ((m.file_type || '') + ' ' + (m.file_url || '')).toLowerCase();
      if (ft.includes('link') || /youtube|youtu\.be|\/watch/.test(ft)) { skipped.push({ title: m.title, reason: 'link extern (nu e fisier)' }); continue; }
      if (/\.mp4|\.webm|\.mov|\.avi|\.mkv|video/.test(ft)) { skipped.push({ title: m.title, reason: 'video (fara text)' }); continue; }
      try {
        const fr = await fetchFile(m.file_url);
        if (!fr.ok) { skipped.push({ title: m.title, reason: 'download HTTP ' + fr.status }); continue; }
        const buf = Buffer.from(await fr.arrayBuffer());
        let text = await extractText(buf, m.file_type || m.file_url);
        // fallback OCR cu Gemini pentru imagini / pdf-uri scanate (mime corect)
        const mime = mimeFor(m.file_type || m.file_url);
        if ((!text || text.trim().length < 30) && mime && buf.length < 15 * 1024 * 1024) {
          try { text = await geminiOcr(buf, mime); if (text && text.trim().length >= 30) console.log('  OCR (Gemini) aplicat: ' + m.title); } catch (e) { /* ramane gol */ }
        }
        if (!text || text.trim().length < 30) { skipped.push({ title: m.title, reason: 'fara text extractabil' }); continue; }
        const chunks = chunkText(text);
        for (let j = 0; j < chunks.length; j++) {
          const emb = await embedOne(chunks[j], 'RETRIEVAL_DOCUMENT');
          added.push({ material_id: m.id, title: m.title, subject: m.subject, category: m.category, content: chunks[j], embedding: emb });
        }
        done++;
        console.log(`nou indexat: ${m.title} (${chunks.length} bucati)`);
      } catch (e) {
        skipped.push({ title: m.title, reason: (e.message || 'eroare').slice(0, 80) });
      }
    }

    const merged = baseIndex.concat(added);
    localIndex = merged;
    fs.writeFileSync(INDEX_FILE, JSON.stringify(merged), 'utf8');
    res.json({ materiale_total: materials.length, nou_indexate: done, bucati_noi: added.length, total_bucati: merged.length, skipped });
  } catch (err) {
    console.error('index-local error:', err);
    res.status(500).json({ error: 'Eroare la indexare: ' + err.message });
  }
});

app.post('/api/ai/ask', rateLimit, async (req, res) => {
  try {
    const { question, subject = null, history = [] } = req.body || {};
    if (!question || question.trim().length < 2) return res.status(400).json({ error: 'Intrebare lipsa' });

    let currentText = question;
    let systemInstruction = 'Esti un tutor profesionist pentru elevi de liceu din Romania (BAC). ' + BASE_RULES;
    let sources = [];

    if (localIndex.length > 0) {
      try {
        const qVec = await embedOne(question, 'RETRIEVAL_QUERY');
        let pool = localIndex;
        if (subject && subject !== 'all') {
          const filtered = localIndex.filter((c) => (c.subject || '').toLowerCase() === String(subject).toLowerCase());
          if (filtered.length > 0) pool = filtered;
        }
        const scored = pool.map((c) => ({ c, sim: cosine(qVec, c.embedding) })).sort((a, b) => b.sim - a.sim);
        const best = scored.length ? scored[0].sim : 0;
        // Folosim materialele DOAR daca cel mai bun match e clar relevant (>=0.70),
        // ca sa evitam surse false la intrebari generale. Altfel -> cunostinte generale.
        const top = best >= 0.70 ? scored.slice(0, 5).filter((x) => x.sim >= 0.66) : [];
        if (top.length > 0) {
          const context = top.map((x, i) => `[Material ${i + 1}: ${x.c.title}] (${x.c.subject || '?'})\n${x.c.content}`).join('\n\n---\n\n');
          currentText = `Materiale din platforma (sursa principala):\n\n${context}\n\nIntrebare: ${question}`;
          systemInstruction =
            'Esti un tutor profesionist pentru elevi de liceu din Romania (BAC). ' +
            'Raspunzi FOLOSIND in primul rand materialele platformei de mai jos. ' +
            'Daca materialele acopera intrebarea, bazeaza-te pe ele; daca nu sunt suficiente, completeaza din cunostintele tale, dar nu inventa. ' +
            BASE_RULES;
          const seen = new Set();
          for (const x of top) {
            if (sources.length >= 3) break;
            if (!seen.has(x.c.material_id)) { seen.add(x.c.material_id); sources.push({ material_id: x.c.material_id, title: x.c.title, subject: x.c.subject, category: x.c.category }); }
          }
        }
      } catch (e) {
        console.error('RAG local retrieval error (fallback general):', e.message);
      }
    }

    // conversatie multi-tura: istoric recent + tura curenta (care contine si contextul RAG)
    const contents = [];
    const recent = Array.isArray(history) ? history.slice(-8) : [];
    for (const h of recent) {
      if (!h || !h.content) continue;
      contents.push({ role: h.role === 'assistant' ? 'model' : 'user', parts: [{ text: String(h.content).slice(0, 4000) }] });
    }
    contents.push({ role: 'user', parts: [{ text: currentText }] });

    const answer = sanitizeMath(await generateWithRetry(contents, systemInstruction));
    res.json({ answer, sources });
  } catch (err) {
    console.error('ask error:', err);
    const msg = String((err && err.message) || err);
    // cota Gemini epuizata -> mesaj prietenos (200), nu eroare urata
    if (/RESOURCE_EXHAUSTED|quota|429/i.test(msg)) {
      return res.json({
        answer: 'Am atins pentru moment limita gratuită de întrebări (planul gratuit Gemini). Te rog încearcă din nou peste un minut. Dacă apare des, e nevoie de o cheie Gemini cu facturare activată.',
        sources: [],
      });
    }
    res.status(500).json({ error: 'Eroare la generarea raspunsului: ' + (msg || 'necunoscuta') });
  }
});

app.post('/api/ai/debug', requireAdmin, async (req, res) => {
  try {
    const { question } = req.body || {};
    const qVec = await embedOne(question, 'RETRIEVAL_QUERY');
    const scored = localIndex
      .map((c) => ({ title: c.title, sim: Number(cosine(qVec, c.embedding).toFixed(3)) }))
      .sort((a, b) => b.sim - a.sim)
      .slice(0, 8);
    res.json({ question, top: scored });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// ─── Mentor AI ──────────────────────────────────────────────
// Genereaza raspuns structurat (JSON) pentru mentorul personal.
async function generateJSON(contents, systemInstruction) {
  // 'gemini-flash-latest' are cota separata (modelele fixe se epuizeaza des pe planul gratuit)
  const models = ['gemini-flash-latest', CHAT_MODEL, 'gemini-2.5-flash-lite'];
  let lastErr;
  for (const model of models) {
    try {
      return await withRetry(async () => {
        const result = await ai.models.generateContent({
          model, contents,
          config: { systemInstruction, maxOutputTokens: 8192, responseMimeType: 'application/json' },
        });
        return JSON.parse(result.text || '{}');
      }, 'json');
    } catch (err) { lastErr = err; }
  }
  throw lastErr;
}

// Materiale unice disponibile (pentru recomandari), filtrate pe materie.
function materialsForSubject(subject) {
  const seen = new Set();
  const list = [];
  for (const c of localIndex) {
    if (seen.has(c.material_id)) continue;
    if (subject && subject !== 'universal' && (c.subject || '').toLowerCase() !== String(subject).toLowerCase()) continue;
    seen.add(c.material_id);
    list.push({ id: c.material_id, title: c.title, subject: c.subject, category: c.category });
  }
  return list.slice(0, 40);
}

app.post('/api/ai/mentor', rateLimit, async (req, res) => {
  try {
    const { message, history = [], subject = 'universal', state = {} } = req.body || {};
    if (!message || String(message).trim().length < 1) return res.status(400).json({ error: 'Mesaj lipsa' });

    const chapters = Array.isArray(state.chapters) ? state.chapters : [];
    const todos = Array.isArray(state.todos) ? state.todos : [];
    const materials = materialsForSubject(subject);

    const systemInstruction =
      'Esti "Mentor AI", profesorul personal al unui elev de liceu din Romania (BAC: matematica, informatica, fizica, romana etc). ' +
      'In functie de mesaj faci DOUA lucruri: ' +
      '(A) Daca elevul te INTREABA ceva sau cere ajutor la o problema/tema — RASPUNZI COMPLET in campul "reply": rezolvi pas cu pas si explici, exact ca un tutor bun. ' +
      '(B) Daca elevul RAPORTEAZA ce a lucrat / cum s-a descurcat — raspunzi scurt si incurajator si ii actualizezi progresul. ' +
      'IMPORTANT: cand elevul spune ca NU STIE, NU INTELEGE sau SE BLOCHEAZA la un subiect, pe langa ca il ajuti in reply, adauga OBLIGATORIU acel subiect in actions.todosAdd (lista "de lucrat"). ' +
      'Mereu poti recomanda lectii/teste/probleme din materialele platformei. ' +
      'INFORMATICA: scrie codul in blocuri ```cpp sau ```pseudocod. MATEMATICA/FIZICA: simboluri Unicode (× ÷ √ π ² ³), fara LaTeX; la fizica pune unitatile. ' +
      'Raspunde DOAR cu JSON valid, in limba romana.';

    const prompt =
      `MESAJUL ELEVULUI: "${String(message).slice(0, 1500)}"\n\n` +
      `MATERIA ACTIVA: ${subject}\n\n` +
      `CAPITOLE (id | eticheta | facut):\n${chapters.map((c) => `${c.id} | ${c.label} | ${c.done ? 'da' : 'nu'}`).join('\n') || '(niciunul)'}\n\n` +
      `DE LUCRAT ACUM:\n${todos.filter((t) => !t.done).map((t) => `- ${t.title}`).join('\n') || '(gol)'}\n\n` +
      `MATERIALE DISPONIBILE (PRIORITATE: recomanda mai intai DIN ASTEA, folosind id-ul si titlul EXACT; recomanda o tema generala fara id doar daca nimic din lista nu se potriveste):\n${materials.map((m) => `- [${m.id}] ${m.title} (${m.subject || '?'})`).join('\n') || '(niciunul)'}\n\n` +
      `Intoarce EXACT acest JSON:\n` +
      `{\n` +
      `  "reply": "raspunsul tau: daca elevul intreaba ceva sau cere ajutor la o problema, REZOLVI complet pas cu pas si explici (ca un tutor); daca doar raporteaza ce a lucrat, raspunzi scurt si incurajator",\n` +
      `  "actions": {\n` +
      `    "chaptersDone": ["id-uri de capitole de bifat ca FACUTE daca elevul zice ca le-a terminat"],\n` +
      `    "chaptersUndone": ["id-uri de debifat"],\n` +
      `    "todosAdd": [{"title":"subiectul de exersat","hint":"ce sa faca concret"}] (pune AICI orice subiect la care elevul zice ca nu stie / nu intelege / se blocheaza),\n` +
      `    "todosDone": ["titlul exact al unui todo de marcat completat"],\n` +
      `    "strengthsAdd": [{"title":"scurt","note":"ce i-a iesit bine"}],\n` +
      `    "workLogAdd": {"what":"ce a lucrat","duration":"ex: 45m"},\n` +
      `    "testsDone": <numar INTREG de teste/simulari pe care le-a FACUT/rezolvat elevul, daca a mentionat (ex: "am lucrat 3 teste" -> 3); altfel omite>,\n` +
      `    "examsAdd": [{"type":"Test","title":"scurt","scope":"din ce capitole","when":"ex: 18 iul","daysLeft":9,"plan":["zi 1-3: ...","zi 4-6: ..."]}]\n` +
      `  },\n` +
      `  "recommendations": [{"type":"Lectie","materialId":"id-ul EXACT din lista de materiale daca recomanzi unul din ele, altfel lasa gol","title":"titlul material real din lista sau o tema","reason":"de ce ii ajuta"}]\n` +
      `}\n` +
      `Toate campurile din "actions" sunt optionale — pune [] sau omite daca nu se aplica. Nu bifa capitole daca elevul nu a spus clar ca le-a terminat. ` +
      `Daca elevul mentioneaza un test/simulare/examen VIITOR (si din ce da), adauga-l in "examsAdd" cu un mini-plan. ` +
      `Daca elevul zice ca A FACUT / a lucrat / a rezolvat teste (deja), pune numarul lor in "testsDone" (NU in examsAdd).`;

    const contents = [];
    for (const h of (Array.isArray(history) ? history.slice(-6) : [])) {
      if (h && h.content) contents.push({ role: h.role === 'assistant' ? 'model' : 'user', parts: [{ text: String(h.content).slice(0, 2000) }] });
    }
    contents.push({ role: 'user', parts: [{ text: prompt }] });

    const out = await generateJSON(contents, systemInstruction);
    if (out && out.reply) out.reply = sanitizeMath(String(out.reply));
    out.actions = out.actions || {};
    out.recommendations = Array.isArray(out.recommendations) ? out.recommendations : [];
    // ataseaza un materialId VERIFICAT (setat doar daca materialul chiar exista in lista)
    const normT = (s) => String(s || '').toLowerCase().replace(/\s+/g, ' ').trim();
    const byId = new Map(materials.map((m) => [String(m.id), m]));
    const byTitle = new Map(materials.map((m) => [normT(m.title), m]));
    out.recommendations = out.recommendations.map((r) => {
      const mat = (r && r.materialId && byId.get(String(r.materialId))) || (r && byTitle.get(normT(r.title)));
      return { ...r, materialId: mat ? String(mat.id) : null };
    });
    res.json(out);
  } catch (err) {
    console.error('mentor error:', err);
    const msg = String((err && err.message) || err);
    if (/RESOURCE_EXHAUSTED|quota|429/i.test(msg)) {
      return res.json({ reply: 'Am atins limita gratuita de raspunsuri pentru moment. Te rog incearca din nou peste un minut.', actions: {}, recommendations: [] });
    }
    res.status(500).json({ error: 'Eroare mentor: ' + msg });
  }
});

app.listen(PORT, () => {
  console.log(`AI-RAG server pe portul ${PORT}  (key: ${process.env.GEMINI_API_KEY ? 'da' : 'LIPSA'}, supabase: ${SUPABASE_URL && SUPABASE_ANON_KEY ? 'da' : 'LIPSA'}, index: ${localIndex.length} bucati)`);
});

// Keep-alive: pe Render gratuit serverul adoarme dupa ~15 min inactivitate,
// iar trezirea dureaza 1-3 min. Se auto-pinge la 10 min ca sa ramana treaz.
const KEEP_ALIVE_URL = process.env.RENDER_EXTERNAL_URL || 'https://smartlearn-ai-server.onrender.com';
setInterval(() => {
  fetch(`${KEEP_ALIVE_URL}/api/health`).catch(() => {});
}, 10 * 60 * 1000);
console.log('Keep-alive activat pentru', KEEP_ALIVE_URL);
