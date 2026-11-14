import { useEffect, useRef, useState, type FormEvent } from "react";
import { sendMessage, type Citation } from "./api";
import { DEMO_MESSAGES } from "./demo";

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

const isDemo = new URLSearchParams(window.location.search).has("demo");

const EXAMPLES = [
  "How do I declare a path parameter in FastAPI?",
  "What is dependency injection in FastAPI?",
  "How do I install FastAPI?",
];

// Renders ``` fenced code blocks and `inline code`; everything else is text.
function MessageText({ text }: { text: string }) {
  const parts = text.split(/```(?:\w+)?\n?([\s\S]*?)```/g);
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <pre key={i}>
            <code>{part.trimEnd()}</code>
          </pre>
        ) : (
          <p key={i}>
            {part
              .trim()
              .split(/`([^`]+)`/g)
              .map((chunk, j) => (j % 2 === 1 ? <code key={j}>{chunk}</code> : chunk))}
          </p>
        ),
      )}
    </>
  );
}

function Citations({ citations }: { citations: Citation[] }) {
  return (
    <div className="citations">
      <span className="citations-label">Sources</span>
      {citations.map((c, i) => (
        <a key={i} href={c.source_url} target="_blank" rel="noreferrer" className="citation">
          <span className="citation-index">{i + 1}</span>
          <span className="citation-title">{c.title || c.source_url}</span>
          <span className="citation-meta">chunk {c.chunk_index}</span>
        </a>
      ))}
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>(isDemo ? DEMO_MESSAGES : []);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function ask(question: string) {
    const text = question.trim();
    if (!text || loading) return;
    setInput("");
    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);
    try {
      const res = await sendMessage(text, conversationId);
      setConversationId(res.conversation_id);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.answer, citations: res.citations },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    void ask(input);
  }

  function newConversation() {
    setMessages([]);
    setConversationId(null);
    setError(null);
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="logo">SK</div>
          <div>
            <h1>Smart Knowledge Assistant</h1>
            <p>Answers grounded in official developer documentation</p>
          </div>
        </div>
        <button className="ghost" onClick={newConversation}>
          New chat
        </button>
      </header>

      <main className="chat">
        {messages.length === 0 && (
          <div className="empty">
            <h2>Ask a question about the docs</h2>
            <p>Hybrid search (Qdrant + BM25), cross-encoder reranking and cited answers.</p>
            <div className="examples">
              {EXAMPLES.map((q) => (
                <button key={q} onClick={() => void ask(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            <div className="avatar">{m.role === "user" ? "You" : "AI"}</div>
            <div className="bubble">
              <MessageText text={m.content} />
              {m.citations && m.citations.length > 0 && <Citations citations={m.citations} />}
            </div>
          </div>
        ))}

        {loading && (
          <div className="message assistant">
            <div className="avatar">AI</div>
            <div className="bubble typing">
              <span />
              <span />
              <span />
            </div>
          </div>
        )}

        {error && <div className="error">Request failed: {error}</div>}
        <div ref={bottomRef} />
      </main>

      <form className="composer" onSubmit={onSubmit}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about FastAPI, Qdrant, Docker Compose..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
