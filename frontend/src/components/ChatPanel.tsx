import { useEffect, useRef, useState } from "react";
import { api, type ChatMessage } from "../api";

interface Turn extends ChatMessage {
  tools?: string[];
}

const TOOL_LABELS: Record<string, string> = {
  list_seasons: "seasons",
  get_season_winners: "season winners",
  get_top_predicted_styles: "top predictions",
  get_style_details: "style details",
  search_styles: "style search",
};

const SUGGESTIONS = [
  "Why is the black skirt the #1 style for Autumn 2020?",
  "What were the winning styles in Summer 2019?",
  "Which Menswear styles look strongest next season?",
  "What is the concept for the black dress?",
];

/** The AI assistant: a floating button that opens a chat panel. Answers come from the API's data tools. */
export default function ChatPanel() {
  const [open, setOpen] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });   // braces: an effect may only return a clean-up function
  }, [turns, busy]);

  async function send(text: string) {
    const question = text.trim();
    if (!question || busy) return;
    const history: Turn[] = [...turns, { role: "user", content: question }];
    setTurns(history);
    setInput("");
    setBusy(true);
    setError(null);
    try {
      const response = await api.chat(history.map(({ role, content }) => ({ role, content })));
      const tools = [...new Set(response.tools_used.map((t) => TOOL_LABELS[t.name] ?? t.name))];
      setTurns([...history, { role: "assistant", content: response.reply, tools }]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-5 right-5 z-40 flex items-center gap-2 rounded-full bg-accent px-5 py-3 text-sm font-semibold text-white shadow-lg hover:brightness-110"
      >
        <span aria-hidden>✦</span> Ask the assistant
      </button>
    );
  }

  return (
    <aside
      aria-label="AI assistant"
      className="fixed bottom-5 right-5 z-40 flex h-[min(640px,calc(100vh-2.5rem))] w-[min(420px,calc(100vw-2.5rem))] flex-col overflow-hidden rounded-3xl bg-white shadow-2xl ring-1 ring-neutral-200"
    >
      <header className="flex items-center justify-between border-b border-neutral-100 px-5 py-3">
        <div>
          <p className="font-semibold">Style assistant</p>
          <p className="text-xs text-neutral-500">Answers from the model's predictions and sales data</p>
        </div>
        <button onClick={() => setOpen(false)} aria-label="Close assistant" className="rounded-full p-2 text-xl leading-none hover:bg-neutral-100">
          ×
        </button>
      </header>

      <div className="flex-1 space-y-3 overflow-y-auto px-5 py-4 text-sm">
        {turns.length === 0 && (
          <div className="space-y-2">
            <p className="text-neutral-500">Ask about winning styles, predictions, their reasons, or the concepts. For example:</p>
            {SUGGESTIONS.map((s) => (
              <button key={s} onClick={() => send(s)} className="block w-full rounded-2xl bg-neutral-50 px-3 py-2 text-left hover:bg-neutral-100">
                {s}
              </button>
            ))}
          </div>
        )}
        {turns.map((t, i) => (
          <div key={i} className={t.role === "user" ? "flex justify-end" : ""}>
            <div className={`max-w-[85%] whitespace-pre-wrap rounded-2xl px-3 py-2 leading-relaxed ${
              t.role === "user" ? "bg-neutral-900 text-white" : "bg-neutral-100 text-neutral-900"}`}>
              {t.content}
            </div>
            {t.tools && t.tools.length > 0 && (
              <p className="mt-1 text-[11px] text-neutral-400">used: {t.tools.join(", ")}</p>
            )}
          </div>
        ))}
        {busy && <p className="text-neutral-400">Looking it up… (this can take 10–30 seconds)</p>}
        {error && <p className="rounded-2xl bg-red-50 px-3 py-2 text-red-700">{error}</p>}
        <div ref={endRef} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        className="flex gap-2 border-t border-neutral-100 p-3"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question…"
          aria-label="Your question"
          className="flex-1 rounded-full bg-neutral-100 px-4 py-2 text-sm outline-none focus:ring-2 focus:ring-accent"
        />
        <button type="submit" disabled={busy || !input.trim()}
                className="rounded-full bg-accent px-4 py-2 text-sm font-semibold text-white disabled:opacity-40">
          Send
        </button>
      </form>
    </aside>
  );
}
