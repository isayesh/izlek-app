{
  "design_system_name": "İzlek Dashboard — Wide Premium SaaS Refresh",
  "brand_attributes": [
    "ferah (breathing room)",
    "premium SaaS",
    "odaklı (öğrenci/sınav)",
    "güven veren",
    "hızlı aksiyon"
  ],
  "hard_rules": {
    "layout": [
      "NO max-w-6xl / max-w-7xl",
      "NO mx-auto centering wrappers for the dashboard shell",
      "NO single boxed admin-panel container wrapping everything",
      "Use full-width feel with controlled padding: px-6 (mobile) → px-10/px-12 (lg)"
    ],
    "gradients": {
      "restriction": [
        "NEVER use dark/saturated gradient combos (e.g., purple/pink) on any UI element.",
        "Prohibited gradients: blue-500 to purple-600, purple-500 to pink-500, green-500 to blue-500, red to pink etc",
        "NEVER let gradients cover more than 20% of the viewport.",
        "NEVER apply gradients to text-heavy content or reading areas.",
        "NEVER use gradients on small UI elements (<100px width).",
        "NEVER stack multiple gradient layers in the same viewport."
      ],
      "enforcement": "IF gradient area exceeds 20% of viewport OR affects readability THEN use solid colors",
      "allowed_usage": [
        "Hero section background accents only (subtle)",
        "Decorative overlays (noise/grain)"
      ]
    },
    "testing": "All interactive and key informational elements MUST include data-testid (kebab-case, role-based).",
    "tech": "Project uses .js (not .tsx). Guidelines below assume React + Tailwind + shadcn/ui .jsx components."
  },
  "visual_personality": {
    "style_fusion": [
      "Swiss-style hierarchy (clear typographic grid)",
      "Bento-card layout (premium SaaS)",
      "Soft-border minimalism + subtle grain texture",
      "Navy dark mode (not pure black)"
    ],
    "do_not": [
      "Avoid ‘admin panel boxed’ look",
      "Avoid heavy shadows",
      "Avoid centered page container feel",
      "Avoid badge/box behind logo"
    ]
  },
  "typography": {
    "font_pairing": {
      "heading": {
        "name": "Space Grotesk",
        "google_fonts": "https://fonts.google.com/specimen/Space+Grotesk",
        "usage": "Hero greeting, section titles, card headings"
      },
      "body": {
        "name": "Inter",
        "google_fonts": "https://fonts.google.com/specimen/Inter",
        "usage": "Body, labels, helper text"
      }
    },
    "tailwind_usage_notes": [
      "Add fonts via index.html <link> or CSS @import; then set in index.css body and headings utility classes.",
      "Keep Turkish diacritics readability: prefer font-weight 500/600 for headings, 400/500 for body."
    ],
    "type_scale": {
      "h1": "text-4xl sm:text-5xl lg:text-6xl",
      "h2": "text-base md:text-lg",
      "body": "text-sm sm:text-base",
      "small": "text-xs sm:text-sm"
    },
    "copy_tone": {
      "language": "TR",
      "hero_example": {
        "title": "Merhaba, {ad}",
        "subtitle": "Bugün küçük bir adım bile seriyi korur. Hadi planı netleştirelim."
      }
    }
  },
  "color_system": {
    "intent": "Light mode: clean paper + cool neutrals. Dark mode: deep navy surfaces (no pure black). Accent: ocean/teal + soft amber for highlights.",
    "tokens_css_custom_properties": {
      "note": "Implement by updating /app/frontend/src/index.css :root and .dark HSL tokens (shadcn style).",
      "light": {
        "--background": "210 40% 98%",
        "--foreground": "222 47% 11%",
        "--card": "0 0% 100%",
        "--card-foreground": "222 47% 11%",
        "--popover": "0 0% 100%",
        "--popover-foreground": "222 47% 11%",
        "--primary": "222 47% 11%",
        "--primary-foreground": "210 40% 98%",
        "--secondary": "210 30% 96%",
        "--secondary-foreground": "222 47% 11%",
        "--muted": "210 25% 95%",
        "--muted-foreground": "215 16% 40%",
        "--accent": "188 72% 40%",
        "--accent-foreground": "210 40% 98%",
        "--border": "214 20% 90%",
        "--input": "214 20% 90%",
        "--ring": "188 72% 40%",
        "--destructive": "0 72% 52%",
        "--destructive-foreground": "210 40% 98%",
        "--radius": "1rem"
      },
      "dark": {
        "--background": "222 47% 7%",
        "--foreground": "210 40% 98%",
        "--card": "222 47% 10%",
        "--card-foreground": "210 40% 98%",
        "--popover": "222 47% 10%",
        "--popover-foreground": "210 40% 98%",
        "--primary": "210 40% 98%",
        "--primary-foreground": "222 47% 11%",
        "--secondary": "222 35% 14%",
        "--secondary-foreground": "210 40% 98%",
        "--muted": "222 35% 14%",
        "--muted-foreground": "215 20% 70%",
        "--accent": "188 72% 45%",
        "--accent-foreground": "222 47% 7%",
        "--border": "222 28% 18%",
        "--input": "222 28% 18%",
        "--ring": "188 72% 45%",
        "--destructive": "0 62% 35%",
        "--destructive-foreground": "210 40% 98%"
      }
    },
    "semantic_usage": {
      "primary": "Text + primary buttons (solid, no gradients)",
      "accent": "Key highlights: streak, progress, active day in weekly schedule",
      "muted": "Secondary text + subtle surfaces",
      "border": "Soft separators; avoid heavy outlines"
    }
  },
  "layout": {
    "dashboard_shell": {
      "goal": "Full-width feel; content breathes; no centered max-width container.",
      "outer_wrapper_classes": "min-h-screen bg-background text-foreground",
      "page_padding_classes": "px-4 sm:px-6 lg:px-10 xl:px-12",
      "vertical_rhythm": "py-8 sm:py-10",
      "section_spacing": "space-y-8 sm:space-y-10",
      "top_hero_spacing": "pt-6 sm:pt-8 lg:pt-10"
    },
    "grid": {
      "desktop": "grid grid-cols-1 lg:grid-cols-[minmax(0,7fr)_minmax(0,3fr)] gap-6 lg:gap-8",
      "mobile": "Single column stack; weekly panel moves below tasks",
      "rule": "Left column is primary (Today). Right column is secondary (Weekly)."
    },
    "header": {
      "structure": "Left: brand text ‘izlek’ (no box). Right: existing navigation + theme toggle.",
      "classes": "flex items-center justify-between gap-4 py-5",
      "brand": {
        "text": "izlek",
        "classes": "font-[Space_Grotesk] tracking-tight text-xl sm:text-2xl font-semibold",
        "no_badge": true
      }
    }
  },
  "components": {
    "card_style": {
      "base_classes": "rounded-2xl border bg-card text-card-foreground shadow-sm",
      "padding": "p-6 sm:p-7",
      "shadow": "shadow-sm (light) / dark: shadow-none + border contrast",
      "header_row": "flex items-start justify-between gap-4",
      "title": "text-base sm:text-lg font-semibold",
      "description": "text-sm text-muted-foreground"
    },
    "hero_intro": {
      "layout": "A wide hero band (not boxed) with subtle background accent + greeting + quick stats.",
      "classes": "relative overflow-hidden rounded-2xl border bg-card p-6 sm:p-8",
      "background_accent": {
        "rule": "Keep accent subtle; no large gradients. Use a small radial highlight + noise overlay.",
        "example_classes": "before:absolute before:inset-0 before:bg-[radial-gradient(600px_circle_at_20%_10%,hsl(var(--accent)/0.12),transparent_55%)] before:pointer-events-none"
      },
      "content": [
        "H1 greeting",
        "H2 subtitle",
        "Inline summary chips (streak, today count) as plain text rows (avoid badges)",
        "Quick actions row below"
      ]
    },
    "quick_actions": {
      "pattern": "Action strip under hero: 2 primary actions (Oda Bul / Oda Oluştur) + 1 secondary (Görev Ekle).",
      "layout_classes": "grid grid-cols-1 sm:grid-cols-3 gap-3",
      "button_variants": {
        "primary": "Button (shadcn) variant=default; add subtle hover lift",
        "secondary": "variant=secondary",
        "ghost": "variant=ghost"
      },
      "micro_interaction": "hover: translate-y-[-1px] + shadow-sm; active: scale-[0.98]",
      "data_testids": [
        "dashboard-quick-action-find-room-button",
        "dashboard-quick-action-create-room-button",
        "dashboard-quick-action-add-task-button"
      ]
    },
    "today_tasks_card": {
      "purpose": "Primary work area: Today’s tasks list + add/edit/delete/toggle.",
      "header": "Title left; right side: small controls (filter, add) using Button size=sm",
      "list": {
        "use": "ScrollArea for long lists; keep card height stable on desktop",
        "classes": "max-h-[520px]"
      },
      "empty_state": {
        "tone": "Motivating, not childish",
        "cta": "Add task button",
        "classes": "py-10 text-center text-muted-foreground"
      },
      "data_testids": [
        "today-tasks-card",
        "today-tasks-add-button",
        "today-tasks-list",
        "today-tasks-empty-state"
      ]
    },
    "weekly_program_panel": {
      "purpose": "Secondary: weekly schedule overview; quick glance.",
      "layout": "Card with segmented days (Tabs) + compact list.",
      "recommended_components": [
        "Tabs",
        "ScrollArea",
        "Separator"
      ],
      "data_testids": [
        "weekly-program-card",
        "weekly-program-tabs",
        "weekly-program-day-panel"
      ]
    },
    "progress_streak_card": {
      "purpose": "Small KPI card(s) near hero: streak + completion progress.",
      "use_components": [
        "Progress",
        "Tooltip"
      ],
      "visual": "Use accent color for progress indicator; keep background muted.",
      "data_testids": [
        "dashboard-streak-summary",
        "dashboard-progress-summary"
      ]
    },
    "dialogs": {
      "use": "Dialog / AlertDialog from shadcn",
      "rules": [
        "Dialog content max width should be comfortable but not huge: w-full sm:max-w-lg",
        "Inputs full width; labels above",
        "Primary action right aligned"
      ],
      "data_testids": [
        "task-create-dialog",
        "task-edit-dialog",
        "task-delete-alert-dialog",
        "task-dialog-submit-button"
      ]
    },
    "network_error_banner": {
      "pattern": "Top-of-content inline Alert (not toast) with destructive tone.",
      "use_components": [
        "Alert"
      ],
      "classes": "rounded-2xl border border-destructive/30 bg-destructive/10 text-foreground",
      "data_testids": [
        "dashboard-network-error-banner"
      ]
    }
  },
  "component_path": {
    "shadcn_ui": {
      "button": "/app/frontend/src/components/ui/button.jsx",
      "card": "/app/frontend/src/components/ui/card.jsx",
      "dialog": "/app/frontend/src/components/ui/dialog.jsx",
      "alert_dialog": "/app/frontend/src/components/ui/alert-dialog.jsx",
      "tabs": "/app/frontend/src/components/ui/tabs.jsx",
      "scroll_area": "/app/frontend/src/components/ui/scroll-area.jsx",
      "separator": "/app/frontend/src/components/ui/separator.jsx",
      "progress": "/app/frontend/src/components/ui/progress.jsx",
      "tooltip": "/app/frontend/src/components/ui/tooltip.jsx",
      "switch": "/app/frontend/src/components/ui/switch.jsx",
      "sonner": "/app/frontend/src/components/ui/sonner.jsx",
      "calendar": "/app/frontend/src/components/ui/calendar.jsx",
      "skeleton": "/app/frontend/src/components/ui/skeleton.jsx",
      "dropdown_menu": "/app/frontend/src/components/ui/dropdown-menu.jsx"
    },
    "notes": [
      "Prefer shadcn components over raw HTML for dropdown/dialog/calendar/toast.",
      "Use sonner for toasts if already wired; otherwise keep existing behavior and only restyle containers."
    ]
  },
  "motion_micro_interactions": {
    "principles": [
      "No universal transition: avoid transition-all",
      "Use short, premium motion: 150–220ms",
      "Hover lift for cards/buttons: translateY -1px",
      "Respect prefers-reduced-motion"
    ],
    "tailwind_snippets": {
      "card_hover": "transition-shadow duration-200 hover:shadow-md",
      "button_hover": "transition-colors duration-200 hover:shadow-sm active:scale-[0.98]",
      "focus": "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
    },
    "optional_library": {
      "name": "framer-motion",
      "why": "Entrance animations for hero + cards (subtle).",
      "install": "npm i framer-motion",
      "usage_js": "import { motion } from 'framer-motion'; // wrap hero/card with motion.div and use initial/animate with small y/opacity"
    }
  },
  "responsive_behavior": {
    "mobile_first": [
      "Header wraps: brand left, nav becomes horizontal scroll or collapses into Sheet if already present",
      "Hero becomes stacked; quick actions become 1-column then 3-column at sm",
      "Main grid stacks: Today first, Weekly second"
    ],
    "touch_targets": "Minimum 44px height for primary buttons; use Button size=lg for hero CTAs"
  },
  "accessibility": {
    "requirements": [
      "WCAG AA contrast for text on card/background",
      "Visible focus ring using --ring token",
      "Use aria-label for icon-only buttons",
      "Respect prefers-reduced-motion"
    ],
    "keyboard": [
      "Dialogs trap focus (shadcn handles)",
      "Tabs navigable via keyboard"
    ]
  },
  "images": {
    "usage_rule": "Dashboard should be mostly UI-driven; images only as subtle decorative optional hero background or empty state illustration (keep minimal).",
    "image_urls": [
      {
        "category": "texture",
        "description": "Subtle grain/noise background image for hero overlay (optional). Use very low opacity (0.04–0.08).",
        "url": "https://images.unsplash.com/photo-1704470842373-d93a1eaaa8bc?crop=entropy&cs=srgb&fm=jpg&ixlib=rb-4.1.0&q=85"
      },
      {
        "category": "texture",
        "description": "Alternative light grain texture.",
        "url": "https://images.unsplash.com/photo-1570649160447-ffe37dc6f2f0?crop=entropy&cs=srgb&fm=jpg&ixlib=rb-4.1.0&q=85"
      },
      {
        "category": "hero_optional",
        "description": "Soft navy/teal blur background for a small decorative corner (use as background-image in a pseudo-element, not full viewport).",
        "url": "https://images.unsplash.com/photo-1708305729900-906f34a7d49d?crop=entropy&cs=srgb&fm=jpg&ixlib=rb-4.1.0&q=85"
      },
      {
        "category": "empty_state_optional",
        "description": "Minimal study desk photo for empty state illustration (use small, cropped, and optional).",
        "url": "https://images.unsplash.com/photo-1495465798138-718f86d1a4bc?crop=entropy&cs=srgb&fm=jpg&ixlib=rb-4.1.0&q=85"
      }
    ]
  },
  "implementation_notes_for_main_agent": {
    "files_to_touch_preferred": [
      "/app/frontend/src/index.css (update tokens + radius)",
      "Dashboard page component (.js) and its direct widgets/components",
      "Header component (.js) if separate"
    ],
    "layout_instructions": [
      "Remove any mx-auto + max-w wrappers from dashboard shell.",
      "Use full-width page padding: px-4 sm:px-6 lg:px-10 xl:px-12.",
      "Increase vertical spacing: section wrappers use space-y-8 sm:space-y-10.",
      "Implement 70/30 grid on lg+: grid-cols-[minmax(0,7fr)_minmax(0,3fr)] with gap-6 lg:gap-8.",
      "Hero greeting becomes a prominent Card-like band (rounded-2xl) but NOT the entire page boxed—only the hero section is a card.",
      "Cards: rounded-2xl, p-6/p-7, border soft, shadow-sm.",
      "Dark mode: ensure background is deep navy (HSL tokens above), not pure black.",
      "Logo: plain text 'izlek' top-left; no badge, no background, no border.",
      "Keep existing navigation actions; only restyle layout/spacing.",
      "Add data-testid to: header brand, nav links/buttons, theme toggle, quick actions, task list items, dialog submit buttons, error banner."
    ],
    "data_testid_convention": {
      "rule": "kebab-case, role-based",
      "examples": [
        "dashboard-header-brand",
        "dashboard-theme-toggle",
        "dashboard-nav-settings-link",
        "today-task-item-toggle",
        "today-task-item-edit-button",
        "today-task-item-delete-button"
      ]
    }
  },
  "General UI UX Design Guidelines": [
    "- You must **not** apply universal transition. Eg: `transition: all`. This results in breaking transforms. Always add transitions for specific interactive elements like button, input excluding transforms",
    "- You must **not** center align the app container, ie do not add `.App { text-align: center; }` in the css file. This disrupts the human natural reading flow of text",
    "- NEVER: use AI assistant Emoji characters like`🤖🧠💭💡🔮🎯📚🎭🎬🎪🎉🎊🎁🎀🎂🍰🎈🎨🎰💰💵💳🏦💎🪙💸🤑📊📈📉💹🔢🏆🥇 etc for icons. Always use **FontAwesome cdn** or **lucid-react** library already installed in the package.json",
    "\n **GRADIENT RESTRICTION RULE**\nNEVER use dark/saturated gradient combos (e.g., purple/pink) on any UI element.  Prohibited gradients: blue-500 to purple 600, purple 500 to pink-500, green-500 to blue-500, red to pink etc\nNEVER use dark gradients for logo, testimonial, footer etc\nNEVER let gradients cover more than 20% of the viewport.\nNEVER apply gradients to text-heavy content or reading areas.\nNEVER use gradients on small UI elements (<100px width).\nNEVER stack multiple gradient layers in the same viewport.\n\n**ENFORCEMENT RULE:**\n    • Id gradient area exceeds 20% of viewport OR affects readability, **THEN** use solid colors\n\n**How and where to use:**\n   • Section backgrounds (not content backgrounds)\n   • Hero section header content. Eg: dark to light to dark color\n   • Decorative overlays and accent elements only\n   • Hero section with 2-3 mild color\n   • Gradients creation can be done for any angle say horizontal, vertical or diagonal\n\n- For AI chat, voice application, **do not use purple color. Use color like light green, ocean blue, peach orange etc\n",
    "\n</Font Guidelines>\n\n- Every interaction needs micro-animations - hover states, transitions, parallax effects, and entrance animations. Static = dead. \n   \n- Use 2-3x more spacing than feels comfortable. Cramped designs look cheap.\n\n- Subtle grain textures, noise overlays, custom cursors, selection states, and loading animations: separates good from extraordinary.\n   \n- Before generating UI, infer the visual style from the problem statement (palette, contrast, mood, motion) and immediately instantiate it by setting global design tokens (primary, secondary/accent, background, foreground, ring, state colors), rather than relying on any library defaults. Don't make the background dark as a default step, always understand problem first and define colors accordingly\n    Eg: - if it implies playful/energetic, choose a colorful scheme\n           - if it implies monochrome/minimal, choose a black–white/neutral scheme\n\n**Component Reuse:**\n\t- Prioritize using pre-existing components from src/components/ui when applicable\n\t- Create new components that match the style and conventions of existing components when needed\n\t- Examine existing components to understand the project's component patterns before creating new ones\n\n**IMPORTANT**: Do not use HTML based component like dropdown, calendar, toast etc. You **MUST** always use `/app/frontend/src/components/ui/ ` only as a primary components as these are modern and stylish component\n\n**Best Practices:**\n\t- Use Shadcn/UI as the primary component library for consistency and accessibility\n\t- Import path: ./components/[component-name]\n\n**Export Conventions:**\n\t- Components MUST use named exports (export const ComponentName = ...)\n\t- Pages MUST use default exports (export default function PageName() {...})\n\n**Toasts:**\n  - Use `sonner` for toasts\"\n  - Sonner component are located in `/app/src/components/ui/sonner.tsx`\n\nUse 2–4 color gradients, subtle textures/noise overlays, or CSS-based noise to avoid flat visuals."
  ]
}
