# SupportAI Final Deployment Guide

This guide describes how to configure, run, and deploy SupportAI in a 3-stage modular deployment flow.

---

## ⚙️ 1. Deployment Stages Configuration

SupportAI can be dynamically toggled into three different deployment architectures using simple environment variables. You do **not** need to modify the codebase or rebuild your Docker containers to switch stages.

| Stage | Mode Description | RAM Footprint | CPU Usage | Environment Settings |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | **Classifier Only** (intent prediction only) | **~100 MB** | Very Low | `RETRIEVAL_ENABLED=false`<br>`LLM_ENABLED=false` |
| **Stage 2** | **Classifier + Retrieval** (intent + similar tickets) | **~250 MB** | Low | `RETRIEVAL_ENABLED=true`<br>`LLM_ENABLED=false` |
| **Stage 3** | **Full RAG** (intent + retrieval + LLM drafts) | **~8 GB+** (Phi-3) | High | `RETRIEVAL_ENABLED=true`<br>`LLM_ENABLED=true` |

---

## 🐳 2. Local Container Deployment

To build and run the production container locally:

```bash
# Build the Docker image
docker build -t supportai:latest .

# Run Stage 1 (Classifier-only) - Fast, lightweight
docker run -p 8000:8000 -e RETRIEVAL_ENABLED=false -e LLM_ENABLED=false supportai:latest

# Run Stage 2 (Classifier + Retrieval)
docker run -p 8000:8000 -e RETRIEVAL_ENABLED=true -e LLM_ENABLED=false supportai:latest

# Run Stage 3 (Full RAG)
docker run -p 8000:8000 -e RETRIEVAL_ENABLED=true -e LLM_ENABLED=true supportai:latest
```

---

## 🤗 3. Hugging Face Spaces Deployment (Recommended)

Hugging Face Spaces is an ideal showcase platform. To deploy SupportAI on Spaces using the Docker SDK:

### Step 1: Create a Space
1. Log in to [Hugging Face](https://huggingface.co/).
2. Click **New Space**.
3. Name your space (e.g., `SupportAI`).
4. Select **Docker** as the SDK.
5. Choose **Blank** template.
6. Set the Space to **Public** or **Private** as desired.

### Step 2: Push the Repository
Hugging Face Spaces runs using a Git remote. Add it and push your code:

```bash
git remote add hf https://huggingface.co/spaces/<your-username>/<your-space-name>
git push hf master --force
```

### Step 3: Configure Environment Toggles
Because the free Hugging Face Spaces tier operates on 16GB CPU RAM, we recommend running **Stage 2** (Classifier + Retrieval) to prevent out-of-memory crashes due to loading Phi-3:
1. Go to your Hugging Face Space page.
2. Under **Settings** $\rightarrow$ **Variables and Secrets**.
3. Create the following Variables:
   - `RETRIEVAL_ENABLED = true`
   - `LLM_ENABLED = false`

The Space will automatically build the `Dockerfile` in the repository, detect port `7860` dynamically from the `PORT` env var, and launch the web demo!

---

## 🚀 4. Deployment on Render, Railway, or Fly.io

Because SupportAI reads the listening port from the `PORT` environment variable, it works out-of-the-box on web hosting platforms:

### Render
1. Create a new **Web Service** pointing to your GitHub repository.
2. Select **Docker** as the runtime.
3. Under Environment variables, configure `RETRIEVAL_ENABLED` and `LLM_ENABLED` according to your target stage.
4. Render automatically configures `PORT` and routes public traffic to it.

### Railway
1. Click **New Project** $\rightarrow$ **Deploy from GitHub repository**.
2. Select the SupportAI repository.
3. In variables, add `RETRIEVAL_ENABLED` and `LLM_ENABLED`.
4. Railway will build the Docker container and expose it publicly.
