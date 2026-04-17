export default function AppLogo({ className = "", alt = "Izlek" }) {
  return (
    <span className={`inline-flex items-center leading-none select-none ${className}`.trim()} data-testid="app-logo-wrap">
      <img
        src="/logo-light.svg"
        alt={alt}
        className="block h-9 w-auto shrink-0 object-contain dark:hidden"
        draggable="false"
      />
      <img
        src="/logo-dark.svg"
        alt={alt}
        className="hidden h-9 w-auto shrink-0 object-contain dark:block dark:opacity-95"
        draggable="false"
      />
    </span>
  );
}
