// Client for the FastAPI chat endpoint (backend/app/api/chat_router.py).

export interface Citation {
  title: string;
  source_url: string;
  chunk_index: string | number;
}

export interface ChatResponse {
  conversation_id: string;
  answer: string;
  citations: Citation[];
}

export async function sendMessage(
  message: string,
  conversationId: string | null,
): Promise<ChatResponse> {
  const response = await fetch("/api/v1/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, conversation_id: conversationId }),
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // keep the status text
    }
    throw new Error(detail);
  }
  return response.json();
}
