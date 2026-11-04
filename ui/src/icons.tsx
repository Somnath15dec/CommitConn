type P = { size?: number; className?: string };

export const Logo = ({ size = 32 }: P) => (
  <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
    <circle cx="16" cy="16" r="16" fill="var(--green)" />
    <rect x="8" y="9" width="16" height="3" rx="1.5" fill="#000" />
    <rect x="8" y="14.5" width="12" height="3" rx="1.5" fill="#000" />
    <rect x="8" y="20" width="8" height="3" rx="1.5" fill="#000" />
  </svg>
);

export const Search = ({ size = 22, className }: P) => (
  <svg width={size} height={size} viewBox="0 0 24 24" className={className} aria-hidden>
    <path fill="currentColor" d="M10.53 3a7.53 7.53 0 1 0 4.7 13.41l4.18 4.18a1 1 0 0 0 1.42-1.42l-4.18-4.18A7.53 7.53 0 0 0 10.53 3Zm-5.53 7.53a5.53 5.53 0 1 1 11.06 0 5.53 5.53 0 0 1-11.06 0Z" />
  </svg>
);

export const Arrow = ({ size = 20 }: P) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden>
    <path fill="currentColor" d="M13.3 5.3a1 1 0 0 1 1.4 0l6 6a1 1 0 0 1 0 1.4l-6 6a1 1 0 0 1-1.4-1.4l4.3-4.3H4a1 1 0 1 1 0-2h13.6l-4.3-4.3a1 1 0 0 1 0-1.4Z" />
  </svg>
);

export const Heart = ({ size = 18, filled }: P & { filled?: boolean }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden>
    {filled ? (
      <path fill="currentColor" d="M12 20.7 4.6 13.5a4.9 4.9 0 0 1 6.9-7l.5.5.5-.5a4.9 4.9 0 0 1 6.9 7Z" />
    ) : (
      <path fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" d="M12 20.7 4.6 13.5a4.9 4.9 0 0 1 6.9-7l.5.5.5-.5a4.9 4.9 0 0 1 6.9 7Z" />
    )}
  </svg>
);

export const Skip = ({ size = 18 }: P) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden>
    <circle cx="12" cy="12" r="8.6" fill="none" stroke="currentColor" strokeWidth="1.8" />
    <path stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" d="M8.5 12h7" />
  </svg>
);

export const Sparkle = ({ size = 12 }: P) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden>
    <path fill="currentColor" d="M12 2c.4 4.9 2.6 7.6 8 8.5v1c-5.4.9-7.6 3.6-8 8.5h-1c-.4-4.9-2.6-7.6-8-8.5v-1c5.4-.9 7.6-3.6 8-8.5h1Z" />
  </svg>
);

export const User = ({ size = 20 }: P) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden>
    <path fill="currentColor" d="M12 3a4.5 4.5 0 1 0 0 9 4.5 4.5 0 0 0 0-9Zm-7.5 17.2C4.9 16.4 8 14 12 14s7.1 2.4 7.5 6.2a.8.8 0 0 1-.8.8H5.3a.8.8 0 0 1-.8-.8Z" />
  </svg>
);
