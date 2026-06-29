# 🛡️ Iron Dome AI — Cameroon Digital Trust Platform

**AI-powered platform for detecting scams, misinformation, phishing, and verifying factual claims in Cameroonian cyberspace.**

---

## 🚀 Quick Start (Local)

### 1. Backend Setup

```bash
cd backend
cp .env.example .env
# Edit .env and paste your Gemini API key
nano .env
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the backend:
```bash
uvicorn main:app --reload --port 8000
```

Backend runs at: `http://localhost:8000`
Health check: `http://localhost:8000/health`

---

### 2. Frontend Setup

No build step needed. Just open the file:

```bash
# Option 1: Open directly
open frontend/index.html

# Option 2: Serve locally (recommended for fetch to work without CORS issues)
cd frontend
python3 -m http.server 3000
# Then open http://localhost:3000
```

The frontend auto-detects `localhost` and points to `http://localhost:8000`.

---

## 🌐 Deploy to Render (Backend)

1. Push this entire repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Root Directory**: `backend`
   - **Environment**: Docker
   - **Dockerfile path**: `./Dockerfile`
5. Add environment variable:
   - `GEMINI_API_KEY` = your key
6. Deploy

After deploying, copy your Render URL (e.g. `https://iron-dome-ai.onrender.com`) and update this line in `frontend/index.html`:

```javascript
: 'YOUR_RENDER_BACKEND_URL'; // Replace this
```

---

## 🌐 Deploy Frontend

**Option A — GitHub Pages:**
- Put `frontend/index.html` in the root or `docs/` folder
- Enable GitHub Pages in repo settings

**Option B — Netlify:**
- Drag and drop the `frontend/` folder to [netlify.com/drop](https://app.netlify.com/drop)

---

## 📁 Project Structure

```
iron-dome-ai/
├── backend/
│   ├── main.py          # FastAPI app — all analysis logic
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example     # Copy to .env and add your key
├── frontend/
│   └── index.html       # Complete frontend (single file)
├── render.yaml          # Render deployment config
└── README.md
```

---

## 🔑 Required API Key

Get your **Gemini API key** at: https://aistudio.google.com/app/apikey

Paste it in `backend/.env`:
```
GEMINI_API_KEY=AIza...your_key_here
```

---

## ✅ Supported Input Types

| Type | Description |
|------|-------------|
| **Text** | WhatsApp messages, news claims, social media posts |
| **URL** | News articles, websites — content is fully scraped |
| **YouTube** | Full transcript extracted, then analyzed |
| **Image** | Screenshots, images analyzed with Gemini Vision |

---

## 🌍 Language Support

- French → responds in French
- English → responds in English  
- Pidgin → responds in Pidgin
- Franglais → responds naturally in Franglais

---

## 🎯 Scope

Iron Dome AI only analyzes content related to **Cameroon** — Cameroonian politics, society, economy, media, security, public figures, and institutions. Out-of-scope content is cleanly rejected.
