export default function AppLogo({ className = "", alt = "Izlek" }) {
  return (
    <span className={`flex items-center leading-none select-none ${className}`.trim()} data-testid="app-logo-wrap">
      <img
        src="/logo-light.svg"
        alt={alt}
        className="block w-[140px] h-auto shrink-0 object-contain dark:hidden"
        draggable="false"
      />
      <img
        src="/logo-dark.svg"
        alt={alt}
        className="hidden w-[140px] h-auto shrink-0 object-contain dark:block"
        draggable="false"
      />
    </span>
  );
}
