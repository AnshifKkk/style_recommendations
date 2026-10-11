import { useState } from "react";
import { assetUrl } from "../api";
import Lightbox from "./Lightbox";

interface Props {
  imageUrl: string | null;
  title: string;
  subtitle?: string;
  badge?: string;
  description?: string | null;
}

/** A Pinterest-style card: image with rounded corners, a short caption below. Click the image to enlarge it. */
export default function PinCard({ imageUrl, title, subtitle, badge, description }: Props) {
  const src = assetUrl(imageUrl);
  const [failed, setFailed] = useState(false);   // e.g. the image couldn't be fetched from the image source
  const [enlarged, setEnlarged] = useState(false);
  const showImage = src && !failed;
  return (
    <figure className="mb-4 break-inside-avoid">
      <div className="relative overflow-hidden rounded-2xl bg-neutral-100">
        {showImage ? (
          <button onClick={() => setEnlarged(true)} aria-label={`Enlarge ${title}`} className="block w-full cursor-zoom-in">
            <img
              src={src}
              alt={title}
              loading="lazy"
              onError={() => setFailed(true)}
              className="w-full object-cover transition-transform duration-300 hover:scale-[1.03]"
            />
          </button>
        ) : (
          <div className="flex aspect-[2/3] items-center justify-center px-4 text-center text-xs text-neutral-400">
            Image not available
          </div>
        )}
        {badge && (
          <span className="pointer-events-none absolute left-2 top-2 rounded-full bg-white/90 px-2 py-0.5 text-[11px] font-semibold text-accent shadow">
            {badge}
          </span>
        )}
      </div>
      <figcaption className="px-1 pt-2">
        <p className="text-sm font-medium leading-snug">{title}</p>
        {subtitle && <p className="text-xs text-neutral-500">{subtitle}</p>}
        {description && <p className="mt-1 text-xs leading-snug text-neutral-600">{description}</p>}
      </figcaption>
      {enlarged && src && (
        <Lightbox src={src} title={title} caption={description ?? subtitle} onClose={() => setEnlarged(false)} />
      )}
    </figure>
  );
}
