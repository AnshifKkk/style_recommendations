import { useState } from "react";
import { assetUrl, type SeasonStyle } from "../api";
import Lightbox from "./Lightbox";

interface Props {
  imageUrl: string;
  styles: SeasonStyle[];
}

/** The single image with all three next-season concepts, and how each concept links back to its predicted style. */
export default function ConceptsOverview({ imageUrl, styles }: Props) {
  const [enlarged, setEnlarged] = useState(false);
  const src = assetUrl(imageUrl)!;
  return (
    <section className="mb-12 rounded-3xl bg-neutral-50 p-4 sm:p-6">
      <h3 className="text-lg font-semibold">Next-season concepts</h3>
      <p className="mb-4 text-sm text-neutral-500">
        All three concepts in one image. Click to enlarge.
      </p>
      <button onClick={() => setEnlarged(true)} aria-label="Enlarge the concepts image" className="block w-full cursor-zoom-in">
        <img src={src} alt="All three next-season concepts" className="w-full rounded-2xl bg-white shadow-sm" />
      </button>
      <ol className="mt-4 grid gap-3 sm:grid-cols-3">
        {styles.map((s) => (
          <li key={s.style_id} className="rounded-2xl bg-white p-3 text-xs leading-relaxed shadow-sm">
            <p className="font-semibold text-neutral-900">#{s.rank} {s.name}</p>
            <p className="mt-1 text-neutral-600">{s.concept?.description ?? "Concept pending."}</p>
          </li>
        ))}
      </ol>
      {enlarged && (
        <Lightbox src={src} title="Next-season concepts" caption={styles.map((s) => `#${s.rank} ${s.concept?.description ?? s.name}`).join("  ·  ")}
                  onClose={() => setEnlarged(false)} />
      )}
    </section>
  );
}
