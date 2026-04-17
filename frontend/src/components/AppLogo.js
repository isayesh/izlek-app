export default function AppLogo({ className = "", alt = "Izlek" }) {
  return (
    <span className={`inline-flex items-center ${className}`.trim()} data-testid="app-logo-wrap">
      <img
        src="/logo-light.png"
        alt={alt}
        className="h-10 w-auto object-contain dark:hidden"
        draggable="false"
      />
      <img
        src="/logo-dark.png"
        alt={alt}
        className="hidden h-10 w-auto object-contain dark:block"
        draggable="false"
      />
    </span>
  );
}
