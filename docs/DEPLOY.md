# Deploy the API (non-local)

So Modal (or any client) can call your API from the internet.

---

## Option 1: Fly.io (recommended)

**1. Install Fly CLI**  
https://fly.io/docs/hands-on/install-flyctl/

**2. Log in and create app (first time)**  
```bash
cd "/Users/fgarcia/AI Personal Project/treehacks-2026"
fly auth login
fly launch --no-deploy
```
When asked for app name, use e.g. `treehacks-vector-api`. Say no to PostgreSQL if offered.

**3. Set secrets (your MongoDB and API key)**  
```bash
fly secrets set MONGODB_URI="mongodb+srv://USER:PASSWORD@treehacks2026.afbqken.mongodb.net/"
fly secrets set MONGODB_DB="TreeHacks2026"
fly secrets set API_KEY="your-modal-api-key"
```
Use your real values. No quotes inside the URI if it has special chars, or use single quotes.

**4. Deploy**  
```bash
fly deploy
```

**5. Get your URL**  
```bash
fly status
# or open the app
fly open
```
Your API URL is: **`https://treehacks-vector-api.fly.dev`** (or whatever app name you chose). Use that as **VECTOR_API_URL** in Modal.

**Health check:** `https://your-app.fly.dev/health`

---

## Option 2: Railway

**1. Install Railway CLI** (optional) or use the dashboard: https://railway.app

**2. New project → Deploy from GitHub** (connect repo) or **Empty project → Deploy from Dockerfile**.

**3. Root directory:** project root. Build: Dockerfile. Start command is in the Dockerfile.

**4. Variables (in Railway dashboard):**  
- `MONGODB_URI`  
- `MONGODB_DB`  
- `API_KEY` (optional)

**5. Deploy.** Railway gives you a URL like `https://your-app.up.railway.app`. Use that as **VECTOR_API_URL**.

---

## Option 3: Render

**1. Go to https://render.com** → New → Web Service.

**2. Connect repo** or use “Docker” and paste Dockerfile.

**3. Environment:** Add `MONGODB_URI`, `MONGODB_DB`, `API_KEY`.

**4. Deploy.** URL like `https://your-service.onrender.com` → use as **VECTOR_API_URL**.

---

## After deploy

- **API URL** = the base URL (e.g. `https://treehacks-vector-api.fly.dev`). No path.
- In **Modal**: set secret **VECTOR_API_URL** = that URL, **VECTOR_API_KEY** = same as **API_KEY**.
- Test: `curl https://your-url/health`
