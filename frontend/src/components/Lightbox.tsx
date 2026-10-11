import { useEffect } from "react";
import { createPortal } from "react-dom";

interface Props {
  src: string;
  title: string;
  caption?: string | null;
  onClose: () => void;
}

/** Full-screen view of one image. Click outside the image or press Esc to close. */
export default function Lightbox({ src, title, caption, onClose }: Props) {
  useEffect(() => {
    // capture phase + stopImmediatePropagation: Esc closes only the lightbox, not a dialog underneath it
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.stopImmediatePropagation();
      onClose();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onClose]);

  // rendered at the page root so the masonry columns or a dialog can't clip it
  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-label={title}
      className="fixed inset-0 z-50 flex cursor-zoom-out flex-col items-center justify-center bg-black/85 p-4"
      onClick={onClose}
    >
      <button onClick={onClose} aria-label="Close" className="absolute right-4 top-4 rounded-full p-2 text-2xl leading-none text-white hover:bg-white/10">
        ×
      </button>
      <img
        src={src}
        alt={title}
        className="max-h-[80vh] max-w-full cursor-default rounded-2xl bg-white object-contain shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      />
      <div className="mt-4 max-w-2xl cursor-default text-center text-white" onClick={(e) => e.stopPropagation()}>
        <p className="font-medium">{title}</p>
        {caption && <p className="mt-1 text-sm text-white/70">{caption}</p>}
      </div>
    </div>,
    document.body,
  );
}
