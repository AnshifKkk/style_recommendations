import type { SeasonStyle } from "../api";
import PinCard from "./PinCard";
import ScoreBadge from "./ScoreBadge";

interface Props {
  style: SeasonStyle;
  predicted: boolean;
  onOpen: (styleId: string) => void;
}

/** One winning style: header with rank and score, then a masonry board of its best-selling articles. */
export default function StyleBoard({ style, predicted, onOpen }: Props) {
  return (
    <section className="mb-12">
      <header className="mb-4 flex flex-wrap items-center gap-3">
        <span className="flex h-9 w-9 items-center justify-center rounded-full bg-accent-soft text-base font-bold text-accent">
          {style.rank}
        </span>
        <div className="mr-auto">
          <h2 className="text-lg font-semibold leading-tight">{style.name}</h2>
          <p className="text-xs text-neutral-500">{style.category}</p>
        </div>
        <ScoreBadge score={style.success_score} explanation={style.score_explanation} predicted={predicted} />
        <button
          onClick={() => onOpen(style.style_id)}
          className="rounded-full border border-neutral-300 px-4 py-1.5 text-sm font-medium hover:bg-neutral-100"
        >
          Details
        </button>
      </header>

      <div className="columns-2 gap-4 sm:columns-3 lg:columns-6">
        {predicted && style.concept && (
          <PinCard
            imageUrl={style.concept.image_url}
            title="Next-season concept"
            badge="AI concept"
            description={style.concept.description ?? "Concept generation pending."}
          />
        )}
        {style.articles.map((a) => (
          <PinCard
            key={a.article_id}
            imageUrl={a.image_url}
            title={a.name}
            subtitle={`${a.units.toLocaleString()} sold${predicted ? " last quarter" : ""}`}
          />
        ))}
      </div>
    </section>
  );
}
