import type { SeasonInfo } from "../api";

interface Props {
  seasons: SeasonInfo[];
  selected: string | null;
  onSelect: (quarterId: string) => void;
}

const statusTag: Record<SeasonInfo["status"], string | null> = {
  actual: null,
  predicted: "Predicted",
  coming_soon: "Soon",
};

export default function SeasonTimeline({ seasons, selected, onSelect }: Props) {
  return (
    <nav aria-label="Seasons" className="no-scrollbar flex gap-2 overflow-x-auto pb-1">
      {seasons.map((s) => {
        const active = s.quarter_id === selected;
        const tag = statusTag[s.status];
        return (
          <button
            key={s.quarter_id}
            onClick={() => onSelect(s.quarter_id)}
            aria-current={active ? "page" : undefined}
            className={`flex shrink-0 items-center gap-2 rounded-full px-4 py-2 text-sm font-medium transition-colors ${
              active
                ? "bg-neutral-900 text-white"
                : s.status === "coming_soon"
                  ? "bg-neutral-50 text-neutral-400 hover:bg-neutral-100"
                  : "bg-neutral-100 text-neutral-800 hover:bg-neutral-200"
            }`}
          >
            {s.label}
            {tag && (
              <span
                className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
                  s.status === "predicted" ? "bg-accent text-white" : "bg-neutral-200 text-neutral-500"
                }`}
              >
                {tag}
              </span>
            )}
          </button>
        );
      })}
    </nav>
  );
}
