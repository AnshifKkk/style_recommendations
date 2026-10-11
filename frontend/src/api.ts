// Typed client for the backend API (see backend/schemas.py).
export const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

export type SeasonStatus = "actual" | "predicted" | "coming_soon";

export interface SeasonInfo {
  quarter_id: string;
  label: string;
  status: SeasonStatus;
  start: string;
  end: string;
}

export interface SeasonArticle {
  article_id: string;
  name: string;
  description: string | null;
  units: number;
  image_url: string | null;
}

export interface Concept {
  image_url: string | null;
  description: string | null;
}

export interface SeasonStyle {
  rank: number;
  style_id: string;
  name: string;
  category: string;
  success_score: number;
  score_explanation: string;
  articles: SeasonArticle[];
  concept: Concept | null;
}

export interface SeasonDetail extends SeasonInfo {
  styles: SeasonStyle[];
  concepts_image_url: string | null;
}

export interface QuarterSales {
  quarter_id: string;
  season: string;
  success_score: number;
  revenue: number;
  units: number;
  articles: number;
}

export interface Contribution {
  feature: string;
  label: string;
  value: string;
  contribution: number;
  effect: "raises" | "lowers";
}

export interface StyleDetail {
  style_id: string;
  name: string;
  category: string;
  quarter: string;
  rank: number | null;
  commercially_relevant: boolean;
  prediction_score: number;
  relative_score: number;
  percentile: number | null;
  attributes: { segment: string; product_type: string; pattern: string; colour: string; tone: string };
  articles: { article_id: string; name: string; description: string | null; units_last_quarter: number; image_url: string | null }[];
  explanation: { summary: string; contributions: Contribution[] };
  sales_history: QuarterSales[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  reply: string;
  tools_used: { name: string; args: Record<string, unknown> }[];
  model: string;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  seasons: () => get<SeasonInfo[]>("/seasons"),
  season: (quarterId: string) => get<SeasonDetail>(`/seasons/${quarterId}`),
  style: (styleId: string) => get<StyleDetail>(`/styles/${encodeURIComponent(styleId)}`),
  chat: (messages: ChatMessage[]) => post<ChatResponse>("/chat", { messages }),
};

/** Image paths from the API are relative to the API server. */
export const assetUrl = (path: string | null) => (path ? `${API_URL}${path}` : null);

/** "2020_autumn" → "Autumn 2020" */
export const quarterLabel = (quarterId: string) => {
  const [year, season] = quarterId.split("_");
  return `${season[0].toUpperCase()}${season.slice(1)} ${year}`;
};
