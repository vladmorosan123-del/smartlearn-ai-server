# SmartLearning — Server AI

Server AI (RAG + Gemini) care răspunde din materialele platformei. Node + Express.
Se pune pe un host cu HTTPS automat (ex. **Render**), iar frontend-ul îl apelează prin `VITE_AI_URL`.

## Rulare locală
```bash
npm install
cp .env.example .env   # completează GEMINI_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY
npm start              # pornește pe portul 3030
```

## Deploy pe Render (HTTPS automat, gratis)

1. Pune acest folder într-un **repo GitHub separat** (ex. `smartlearn-ai-server`).
2. Pe [render.com](https://render.com) → **New → Web Service** → conectează repo-ul.
3. Setări:
   - **Build Command:** `npm install`
   - **Start Command:** `npm start`
   - **Instance Type:** Free
4. La **Environment** adaugă variabilele (din `.env.example`):
   - `GEMINI_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`
   - `CORS_ORIGIN` = domeniul site-ului tău (ex. `https://site-ul-tau.ro`)
   - `ADMIN_TOKEN` = o parolă aleasă de tine (pentru reindexare)
5. Deploy. Render îți dă un URL `https://smartlearn-ai-server.onrender.com`.
6. Testează: `https://...onrender.com/api/health` → trebuie să arate `"status":"ok"`.
7. În frontend, setează `VITE_AI_URL` = acel URL `https://...` și fă rebuild/deploy.

## Endpoints
- `GET  /api/health` — stare
- `POST /api/ai/ask` — `{ question, subject?, history? }` → `{ answer, sources }` (limitat pe IP)
- `POST /api/ai/index-local` — reindexează materialele (necesită header `x-admin-token`)

## Reindexare (când adaugi materiale noi)
```bash
curl -X POST https://...onrender.com/api/ai/index-local -H "x-admin-token: PAROLA_TA"
```

## Note
- `rag-index.json` (indexul materialelor) e inclus în repo → serverul pornește gata cu el.
- Planul gratuit Render „adoarme" după inactivitate → prima întrebare după pauză durează ~30s.
- Planul gratuit Gemini are limită zilnică → pentru mulți utilizatori, folosește o cheie cu facturare.
