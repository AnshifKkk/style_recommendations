interface Props {
  score: number;
  explanation: string;
  predicted?: boolean;
}

/** The raw success score; hovering (or focusing) shows what it means in plain words. */
export default function ScoreBadge({ score, explanation, predicted = false }: Props) {
  return (
    <span className="group relative inline-flex">
      <span
        tabIndex={0}
        className={`cursor-help rounded-full px-3 py-1 text-sm font-semibold outline-none ring-offset-2 focus-visible:ring-2 focus-visible:ring-accent ${
          predicted ? "bg-accent text-white" : "bg-neutral-900 text-white"
        }`}
        aria-describedby="score-tooltip"
      >
        {score.toFixed(2)}
        {predicted && <span className="ml-1 font-normal opacity-80">predicted</span>}
      </span>
      <span
        role="tooltip"
        className="pointer-events-none absolute right-0 top-full z-20 mt-2 w-72 max-w-[calc(100vw-2rem)] rounded-xl bg-neutral-900 p-3 text-xs leading-relaxed text-white opacity-0 shadow-xl transition-opacity group-hover:opacity-100 group-focus-within:opacity-100"
      >
        {explanation}
      </span>
    </span>
  );
}
