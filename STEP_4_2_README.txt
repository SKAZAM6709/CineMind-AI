STEP 4.2 — CINEMATIC HOMEPAGE PREVIEW
===================================
Merge the apps/ directory in this ZIP with the apps/ directory in your existing project.
No existing source files are replaced. Your original apps/frontend/app.py continues to work.

From your project root (virtual environment activated):
    python -m streamlit run apps/frontend/cinematic_preview.py

What this prototype demonstrates:
- Original dark visual identity and responsive cinematic hero
- CSS entrance and hover animation (with reduced motion support)
- Real MovieLens movie catalog browsing, searching, and filtering
- No third-party images or API tokens required

What it does NOT include yet:
- Actual film posters and plot descriptions
- RAG-based retrieval, LLM responses, user login or watchlist
- Finished navigation or multiple pages

Once you approve the design we can move these elements into reusable components and merge with app.py.
