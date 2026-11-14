# Frontend

React + TypeScript (Vite) chat interface for the Smart Knowledge Assistant.

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # type check + production build into dist/
```

The dev server proxies `/api` to the FastAPI backend at `http://localhost:8000`
(override with `VITE_API_BASE_URL`). The UI calls `POST /api/v1/chat`, keeps the returned
`conversation_id` for follow-up questions and shows the cited sources under each answer.

Open `/?demo` to render a sample conversation without the backend. The README screenshot
was taken in this mode.

| File | Role |
| --- | --- |
| `src/App.tsx` | Chat layout, message list, citations, composer |
| `src/api.ts` | Typed client for the chat endpoint |
| `src/demo.ts` | Sample conversation used by `?demo` |
| `src/styles.css` | Dark theme |
