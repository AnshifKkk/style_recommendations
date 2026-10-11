import { useEffect, useState } from "react";
import { api, quarterLabel, type StyleDetail as Detail } from "../api";
import PinCard from "./PinCard";
import SalesChart from "./SalesChart";

interface Props {
  styleId: string;
  onClose: () => void;
}

/** Detail view: images, attributes, sales history, prediction score and the reasoning behind it. */
export default function StyleDetail({ styleId, onClose }: Props) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.style(styleId).then(setDetail).catch((e: Error) => setError(e.message));
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [styleId, onClose]);

  const maxContribution = detail ? Math.max(...detail.explanation.contributions.map((c) => Math.abs(c.contribution)), 1e-9) : 1;

  return (
    <div className="fixed inset-0 z-30 flex justify-center overflow-y-auto bg-black/40 p-4 sm:p-8" onClick={onClose}>
      <article
        role="dialog"
        aria-modal="true"
        aria-label="Style details"
        className="h-fit w-full max-w-5xl rounded-3xl bg-white p-6 shadow-2xl sm:p-8"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-wide text-neutral-500">{detail?.category ?? "Style"}</p>
            <h2 className="text-2xl font-semibold">{detail?.name ?? styleId}</h2>
          </div>
          <button onClick={onClose} aria-label="Close" className="rounded-full p-2 text-xl leading-none hover:bg-neutral-100">
            ×
          </button>
        </div>

        {error && <p className="text-sm text-red-600">{error}</p>}
        {!detail && !error && <p className="text-sm text-neutral-500">Loading…</p>}

        {detail && (
          <div className="grid gap-8 lg:grid-cols-[1fr_1.3fr]">
            <div>
              <div className="columns-2 gap-3">
                {detail.articles.map((a) => (
                  <PinCard key={a.article_id} imageUrl={a.image_url} title={a.name}
                           subtitle={`${a.units_last_quarter.toLocaleString()} sold last quarter`} />
                ))}
              </div>
            </div>

            <div className="space-y-6">
              <section>
                <h3 className="mb-2 text-sm font-semibold">Attributes</h3>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(detail.attributes).map(([k, v]) => (
                    <span key={k} className="rounded-full bg-neutral-100 px-3 py-1 text-xs">
                      <span className="text-neutral-500">{k.replace("_", " ")}:</span> {v}
                    </span>
                  ))}
                </div>
              </section>

              <section className="rounded-2xl bg-accent-soft p-4">
                <h3 className="mb-1 text-sm font-semibold">Prediction for {quarterLabel(detail.quarter)}</h3>
                <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm">
                  <span>Score <b>{detail.prediction_score.toFixed(2)}</b></span>
                  <span>Rank <b>{detail.rank ?? "not ranked"}</b></span>
                  {detail.percentile !== null && <span>Ahead of <b>{(detail.percentile * 100).toFixed(1)}%</b> of styles</span>}
                </div>
                <p className="mt-2 text-sm leading-relaxed">{detail.explanation.summary}</p>
              </section>

              <section>
                <h3 className="mb-2 text-sm font-semibold">Why: what drives the prediction</h3>
                <ul className="space-y-2">
                  {detail.explanation.contributions.map((c) => (
                    <li key={c.feature} className="text-xs">
                      <div className="flex justify-between gap-2">
                        <span>{c.label} <span className="text-neutral-500">({c.value})</span></span>
                        <span className={c.effect === "raises" ? "text-emerald-700" : "text-red-600"}>
                          {c.effect === "raises" ? "+" : "−"}{Math.abs(c.contribution).toFixed(2)}
                        </span>
                      </div>
                      <div className="mt-1 h-1.5 rounded-full bg-neutral-100">
                        <div
                          className={`h-1.5 rounded-full ${c.effect === "raises" ? "bg-emerald-600" : "bg-red-500"}`}
                          style={{ width: `${(Math.abs(c.contribution) / maxContribution) * 100}%` }}
                        />
                      </div>
                    </li>
                  ))}
                </ul>
              </section>

              <section>
                <h3 className="mb-2 text-sm font-semibold">Historical performance (success score per quarter)</h3>
                <SalesChart history={detail.sales_history} />
              </section>
            </div>
          </div>
        )}
      </article>
    </div>
  );
}
