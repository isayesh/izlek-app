export default function AppLogo({ className = "", alt = "Izlek" }) {
  return (
    <span className={`inline-flex items-center ${className}`.trim()} data-testid="app-logo-wrap">
      <img
        src="/logo-light.svg"
        alt={alt}
        className="h-12 w-auto shrink-0 object-contain dark:hidden"
        draggable="false"
      />
      <img
        src="/logo-dark.svg"
        alt={alt}
        className="hidden h-12 w-auto shrink-0 object-contain dark:block"
        draggable="false"
      />
    </span>
  );
}
