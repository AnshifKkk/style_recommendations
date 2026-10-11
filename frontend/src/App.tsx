import { useCallback, useEffect, useState } from "react";
import { api, type SeasonDetail, type SeasonInfo } from "./api";
import SeasonTimeline from "./components/SeasonTimeline";
import ChatPanel from "./components/ChatPanel";
import ConceptsOverview from "./components/ConceptsOverview";
import StyleBoard from "./components/StyleBoard";
import StyleDetail from "./components/StyleDetail";

export default function App() {
  const [seasons, setSeasons] = useState<SeasonInfo[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [season, setSeason] = useState<SeasonDetail | null>(null);
  const [openStyle, setOpenStyle] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.seasons()
      .then((list) => {
        setSeasons(list);
        setSelected(list.find((s) => s.status === "predicted")?.quarter_id ?? list[list.length - 1]?.quarter_id ?? null);
      })
      .catch((e: Error) => setError(`Can't reach the API: ${e.message}. Is the backend running?`));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setSeason(null);
    api.season(selected).then(setSeason).catch((e: Error) => setError(e.message));
  }, [selected]);

  const closeDetail = useCallback(() => setOpenStyle(null), []);
  const predicted = season?.status === "predicted";

  return (
    <div className="mx-auto max-w-7xl px-4 pb-16 sm:px-8">
      <header className="sticky top-0 z-10 bg-white/90 py-4 backdrop-blur">
        <div className="mb-3 flex items-baseline gap-3">
          <h1 className="text-2xl font-bold tracking-tight">
            Style<span className="text-accent">.</span>Intelligence
          </h1>
          <p className="hidden text-sm text-neutral-500 sm:block">The three winning styles of every season</p>
        </div>
        <SeasonTimeline seasons={seasons} selected={selected} onSelect={setSelected} />
      </header>

      {error && <p className="mt-6 rounded-xl bg-red-50 p-4 text-sm text-red-700">{error}</p>}
      {selected && !season && !error && <p className="mt-10 text-sm text-neutral-500">Loading season…</p>}

      {season && (
        <main className="mt-6">
          <div className="mb-8">
            <h2 className="text-3xl font-semibold">{season.label}</h2>
            <p className="text-sm text-neutral-500">
              {season.status === "actual" && "Actual winners: the top 3 commercially relevant styles of this season and their best-selling articles."}
              {season.status === "predicted" && "Predicted winners for next season, with an AI-generated concept for each style."}
              {season.status === "coming_soon" && "Beyond the prediction horizon."}
            </p>
          </div>

          {season.status === "coming_soon" ? (
            <div className="flex flex-col items-center rounded-3xl bg-neutral-50 px-6 py-24 text-center">
              <p className="text-4xl font-semibold">Coming soon</p>
              <p className="mt-3 max-w-md text-sm text-neutral-500">
                Predictions are made one quarter ahead. {season.label} will be predicted once the season before it has ended.
              </p>
            </div>
          ) : (
            <>
              {season.styles.map((s) => <StyleBoard key={s.style_id} style={s} predicted={predicted} onOpen={setOpenStyle} />)}
              {season.concepts_image_url && <ConceptsOverview imageUrl={season.concepts_image_url} styles={season.styles} />}
            </>
          )}
        </main>
      )}

      {openStyle && <StyleDetail styleId={openStyle} onClose={closeDetail} />}
      <ChatPanel />
    </div>
  );
}
