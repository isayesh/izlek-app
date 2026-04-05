import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import ThemeToggle from "@/components/ThemeToggle";
import {
  ArrowRight,
  BarChart3,
  BookOpen,
  Calendar,
  CheckCircle2,
  ChevronDown,
  Clock3,
  LogIn,
  MessageSquare,
  Sparkles,
  Target,
  Timer,
  UserPlus,
  Users,
} from "lucide-react";

const HERO_PREVIEW_IMAGE =
  "https://customer-assets.emergentagent.com/job_izlek-showcase/artifacts/e7n3smcn_hero_crop.png";

const ROOM_PREVIEW_IMAGE =
  "https://customer-assets.emergentagent.com/job_izlek-showcase/artifacts/dyw9havo_room_crop_final.png";

const heroMetrics = [
  {
    icon: CheckCircle2,
    label: "Günlük görevler",
    value: "6 görev",
    detail: "öncelik sırasıyla hazır",
    accentClassName: "text-blue-700 dark:text-blue-200",
  },
  {
    icon: Users,
    label: "Aktif oda",
    value: "124 çevrimiçi",
    detail: "eş zamanlı oturumlarda",
    accentClassName: "text-emerald-700 dark:text-emerald-200",
  },
  {
    icon: Calendar,
    label: "Haftalık plan",
    value: "7 gün",
    detail: "dengeli bir çalışma ritmi",
    accentClassName: "text-violet-700 dark:text-violet-200",
  },
];

const steps = [
  {
    number: "01",
    icon: UserPlus,
    title: "Kaydol ve seviyeni tanımla",
    description:
      "Kısa bir başlangıç akışıyla hedefini ve mevcut durumunu paylaş. İlk yapı hemen oluşsun.",
  },
  {
    number: "02",
    icon: Target,
    title: "Planını netleştir",
    description:
      "Günlük görevler, haftalık tempo ve konu dağılımı tek bir ekran üzerinde toparlansın.",
  },
  {
    number: "03",
    icon: BarChart3,
    title: "Ritim kazan ve takip et",
    description:
      "Odalar, ilerleme görünümü ve düzenli tekrarlarla çalışma düzenini sürdürülebilir kıl.",
  },
];

const roomHighlights = [
  "Senkron sayaç ile aynı oturuma birlikte gir.",
  "Anlık chat sayesinde soru sor, kopmadan ilerle.",
  "Oda akışı motive eder; yalnız hissettirmez.",
  "Canlı odalara katıl veya kendi odanı aç.",
];

const roomSignals = [
  { label: "Senkron sayaç", value: "24:10" },
  { label: "Aktif sohbet", value: "3 yeni mesaj" },
  { label: "Katılımcı", value: "18 kişi" },
];

const features = [
  {
    icon: Target,
    title: "Kişisel çalışma akışı",
    description: "Hedefinle uyumlu günlük görevler tek bakışta görünür.",
    cardClassName:
      "border-blue-100/80 bg-blue-50/35 dark:border-blue-500/10 dark:bg-blue-500/[0.04] hover:border-blue-200/80",
    iconClassName:
      "bg-blue-100/90 text-blue-900 dark:bg-blue-500/12 dark:text-blue-100",
  },
  {
    icon: Users,
    title: "Ortak çalışma odaları",
    description:
      "Birlikte çalışma temposu ile motivasyon daha sürdürülebilir hale gelir.",
    cardClassName:
      "border-violet-100/80 bg-violet-50/30 dark:border-violet-500/10 dark:bg-violet-500/[0.04] hover:border-violet-200/80",
    iconClassName:
      "bg-violet-100/90 text-violet-900 dark:bg-violet-500/12 dark:text-violet-100",
  },
  {
    icon: Timer,
    title: "Odaklı seans yapısı",
    description:
      "Süreyi takip etmek kolaylaşır; çalışma anı daha net hissedilir.",
    cardClassName:
      "border-emerald-100/80 bg-emerald-50/30 dark:border-emerald-500/10 dark:bg-emerald-500/[0.04] hover:border-emerald-200/80",
    iconClassName:
      "bg-emerald-100/90 text-emerald-900 dark:bg-emerald-500/12 dark:text-emerald-100",
  },
  {
    icon: MessageSquare,
    title: "Anlık iletişim",
    description:
      "Sorular, kısa notlar ve motivasyon mesajları aynı yerde kalır.",
    cardClassName:
      "border-slate-200/90 bg-slate-50/70 dark:border-slate-700 dark:bg-slate-900/60 hover:border-slate-300/90",
    iconClassName:
      "bg-slate-200/80 text-slate-900 dark:bg-slate-800 dark:text-slate-100",
  },
  {
    icon: BookOpen,
    title: "Haftalık görünüm",
    description:
      "Konuların, görevlerin ve tempo dağılımın dağılmadan yönetilir.",
    cardClassName:
      "border-blue-100/80 bg-blue-50/30 dark:border-blue-500/10 dark:bg-blue-500/[0.04] hover:border-blue-200/80",
    iconClassName:
      "bg-blue-100/90 text-blue-900 dark:bg-blue-500/12 dark:text-blue-100",
  },
  {
    icon: BarChart3,
    title: "İlerleme takibi",
    description:
      "Tamamlanan işler ve kalan yük dengeli bir görünümle izlenir.",
    cardClassName:
      "border-violet-100/80 bg-violet-50/25 dark:border-violet-500/10 dark:bg-violet-500/[0.04] hover:border-violet-200/80",
    iconClassName:
      "bg-violet-100/90 text-violet-900 dark:bg-violet-500/12 dark:text-violet-100",
  },
];

const faqs = [
  {
    question: "İzlek ile nasıl başlıyorum?",
    answer:
      "Kayıt olduktan sonra kısa bir başlangıç akışıyla hedefini ve mevcut durumunu paylaşıyorsun. Ardından plan görünümü senin için hazırlanıyor.",
  },
  {
    question: "Online çalışma odaları nasıl işliyor?",
    answer:
      "Bir odaya katıldığında senkron zamanlayıcı ve anlık sohbet akışına dahil oluyorsun. Böylece hem birlikte çalışma hissi hem de odak korunuyor.",
  },
  {
    question: "Programımı sonra düzenleyebilir miyim?",
    answer:
      "Evet. Haftalık planını, görevlerini ve çalışma yoğunluğunu ihtiyaçlarına göre güncelleyebilirsin.",
  },
  {
    question: "Mobilde de kullanılabiliyor mu?",
    answer:
      "Evet. Landing ve uygulama akışı farklı ekran genişliklerinde rahat kullanılacak şekilde tasarlandı.",
  },
];

function SectionHeading({ eyebrow, title, description, align = "left", testIdPrefix }) {
  const alignment =
    align === "center"
      ? "mx-auto max-w-3xl text-center"
      : "max-w-2xl text-left";

  return (
    <div className={alignment} data-testid={`${testIdPrefix}-heading-block`}>
      <div
        className="inline-flex items-center gap-2 rounded-full border border-border/70 bg-card px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-slate-600 dark:bg-background/70 dark:text-slate-300"
        data-testid={`${testIdPrefix}-eyebrow`}
      >
        <Sparkles className="h-3.5 w-3.5" />
        {eyebrow}
      </div>
      <h2
        className="mt-5 font-display text-4xl font-semibold tracking-tight text-slate-900 dark:text-white sm:text-5xl"
        data-testid={`${testIdPrefix}-title`}
      >
        {title}
      </h2>
      <p
        className="mt-4 text-base leading-7 text-slate-600 dark:text-slate-300 sm:text-lg"
        data-testid={`${testIdPrefix}-description`}
      >
        {description}
      </p>
    </div>
  );
}

function JourneyMark({ idSuffix }) {
  return (
    <span className="relative inline-flex h-5 w-5 items-center justify-center" aria-hidden="true">
      <svg viewBox="0 0 32 32" className="h-5 w-5 dark:hidden" fill="none">
        <defs>
          <linearGradient id={`journey-gradient-${idSuffix}`} x1="4" y1="25" x2="28" y2="6" gradientUnits="userSpaceOnUse">
            <stop stopColor="#0F172A" />
            <stop offset="1" stopColor="#2563EB" />
          </linearGradient>
        </defs>
        <path
          d="M5 24.5C8.4 18.8 11.8 15.2 15.8 12.8C19.3 10.7 22.4 10.4 25.7 7.5"
          stroke={`url(#journey-gradient-${idSuffix})`}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="26.8" cy="6.4" r="2.2" fill={`url(#journey-gradient-${idSuffix})`} />
      </svg>

      <svg viewBox="0 0 32 32" className="hidden h-5 w-5 dark:block" fill="none">
        <path
          d="M5 24.5C8.4 18.8 11.8 15.2 15.8 12.8C19.3 10.7 22.4 10.4 25.7 7.5"
          stroke="white"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="26.8" cy="6.4" r="2.2" fill="white" />
      </svg>
    </span>
  );
}

export default function LandingPage() {
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  const [openFaq, setOpenFaq] = useState(0);

  useEffect(() => {
    if (currentUser) {
      navigate("/dashboard");
    }
  }, [currentUser, navigate]);

  return (
    <div
      className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-white"
      data-testid="landing-page"
    >
      <nav
        className="sticky top-0 z-50 border-b border-border/70 bg-white/80 backdrop-blur-xl dark:bg-slate-950/80"
        data-testid="landing-nav"
      >
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
          <button
            type="button"
            onClick={() => navigate("/")}
            className="flex items-center gap-3 text-left"
            data-testid="landing-brand-button"
          >
            <div className="flex h-7 w-7 items-center justify-center text-slate-900 dark:text-white">
              <JourneyMark idSuffix="nav" />
            </div>
            <div>
              <div
                className="font-display text-[1.4rem] font-semibold tracking-[0.08em]"
                data-testid="landing-brand-name"
              >
                izlek
              </div>
              <div
                className="text-xs text-slate-500 dark:text-slate-400"
                data-testid="landing-brand-tagline"
              >
                odaklı çalışma alanın
              </div>
            </div>
          </button>

          <div className="flex items-center gap-2 sm:gap-3" data-testid="landing-nav-actions">
            <ThemeToggle
              className="border border-border bg-background/90 shadow-sm hover:bg-secondary"
              dataTestId="landing-theme-toggle"
            />
            <Button
              variant="ghost"
              onClick={() => navigate("/login")}
              data-testid="landing-nav-login-button"
            >
              <LogIn className="h-4 w-4" />
              <span className="hidden sm:inline">Giriş Yap</span>
            </Button>
            <Button
              className="bg-primary text-primary-foreground shadow-sm hover:bg-slate-800 hover:shadow-md"
              onClick={() => navigate("/register")}
              data-testid="landing-nav-register-button"
            >
              Hemen Başla
            </Button>
          </div>
        </div>
      </nav>

      <main>
        <section
          className="relative overflow-hidden border-b border-border/60"
          data-testid="landing-hero-section"
        >
          <div className="absolute inset-x-0 top-0 h-72 bg-gradient-to-b from-slate-100/85 via-white/0 to-transparent dark:from-slate-900/40" />
          <div className="absolute inset-y-0 right-0 hidden w-1/2 bg-[radial-gradient(circle_at_top,rgba(15,23,42,0.06),transparent_58%)] dark:bg-[radial-gradient(circle_at_top,rgba(37,99,235,0.08),transparent_58%)] lg:block" />
          <div className="pointer-events-none absolute right-[10%] top-[24%] hidden h-56 w-56 rounded-full bg-[radial-gradient(circle,rgba(59,130,246,0.1),transparent_72%)] opacity-50 blur-[110px] lg:block dark:opacity-30" />
          <img
            src={HERO_PREVIEW_IMAGE}
            alt=""
            aria-hidden="true"
            className="pointer-events-none absolute right-[2%] top-1/2 hidden w-[44rem] max-w-[52vw] -translate-y-1/2 opacity-[0.18] blur-[52px] dark:block"
          />

          <div className="mx-auto max-w-7xl px-4 pb-20 pt-14 sm:px-6 sm:pt-20 lg:px-8 lg:pb-24">
            <div className="grid items-center gap-14 lg:grid-cols-[minmax(0,1.02fr)_minmax(0,0.98fr)]">
              <div className="max-w-2xl" data-testid="hero-copy-column">
                <div
                  className="inline-flex items-center gap-2 rounded-full border border-border/70 bg-card px-4 py-2 text-sm font-medium text-slate-600 shadow-sm dark:bg-background/80 dark:text-slate-300"
                  data-testid="hero-badge"
                >
                  <Sparkles className="h-4 w-4 text-slate-700 dark:text-slate-200" />
                  Premium bir çalışma akışı için sakin ve net bir başlangıç
                </div>

                <h1
                  className="mt-6 font-display text-4xl font-semibold tracking-tight text-slate-900 dark:text-white sm:text-5xl lg:text-6xl"
                  data-testid="hero-title"
                >
                  Planın hazır olsun,
                  <span className="block bg-gradient-to-r from-slate-900 via-slate-700 to-blue-500 bg-clip-text text-transparent dark:from-white dark:via-slate-100 dark:to-blue-300">
                    sen çalışmaya odaklan.
                  </span>
                </h1>

                <p
                  className="mt-6 max-w-xl text-base leading-7 text-slate-600 dark:text-slate-300 sm:text-lg"
                  data-testid="hero-description"
                >
                  İzlek; günlük görevlerini, haftalık ritmini ve online çalışma odalarını tek bir ürün deneyiminde bir araya getirir.
                  Böylece ne yapacağını düşünmek yerine düzenli çalışmaya geçebilirsin.
                </p>

                <div className="mt-8 flex flex-col gap-3 sm:flex-row" data-testid="hero-cta-group">
                  <Button
                    size="lg"
                    className="h-12 rounded-xl bg-primary px-7 text-sm text-primary-foreground shadow-sm hover:-translate-y-0.5 hover:bg-slate-800 hover:shadow-lg"
                    onClick={() => navigate("/register")}
                    data-testid="hero-start-button"
                  >
                    Hemen Başla
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                  <Button
                    size="lg"
                    variant="outline"
                    className="h-12 rounded-xl border-border/80 bg-white/90 px-7 text-sm dark:bg-slate-900/90"
                    onClick={() => navigate("/login")}
                    data-testid="hero-login-button"
                  >
                    <LogIn className="h-4 w-4" />
                    Giriş Yap
                  </Button>
                </div>

                <div className="mt-10 grid gap-3 sm:grid-cols-3" data-testid="hero-metrics-grid">
                  {heroMetrics.map((item, index) => {
                    const Icon = item.icon;

                    return (
                      <div
                        key={item.label}
                        className="rounded-2xl border border-border/70 bg-white/85 p-4 shadow-sm dark:bg-slate-900/85"
                        data-testid={`hero-metric-card-${index}`}
                      >
                        <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-[0.16em] text-slate-500 dark:text-slate-400">
                          <Icon className={`h-4 w-4 ${item.accentClassName}`} />
                          <span data-testid={`hero-metric-label-${index}`}>{item.label}</span>
                        </div>
                        <p
                          className="mt-3 text-lg font-semibold text-slate-900 dark:text-white"
                          data-testid={`hero-metric-value-${index}`}
                        >
                          {item.value}
                        </p>
                        <p
                          className="mt-1 text-sm text-slate-500 dark:text-slate-400"
                          data-testid={`hero-metric-detail-${index}`}
                        >
                          {item.detail}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="relative mx-auto w-full" data-testid="hero-preview-column">
                <Card
                  className="flex w-full max-w-[560px] items-center justify-end overflow-visible border-0 bg-transparent p-0 shadow-none dark:hidden"
                  data-testid="hero-preview-card"
                >
                  <CardContent className="flex w-full items-center justify-end p-0">
                    <div className="relative flex w-full max-w-[560px] items-center justify-end bg-transparent p-0 shadow-none" data-testid="hero-preview-image-frame">
                        <div className="pointer-events-none absolute right-[10%] top-1/2 h-[72%] w-[72%] -translate-y-1/2 rounded-full bg-[radial-gradient(circle,rgba(59,130,246,0.1),transparent_72%)] blur-[72px] dark:hidden" />
                        <img
                          src={HERO_PREVIEW_IMAGE}
                          alt="İzlek ürün önizlemesi"
                          className="m-0 block h-auto w-full max-w-[560px] rounded-[24px] p-0 object-contain shadow-[0_26px_48px_-34px_rgba(15,23,42,0.24)]"
                          data-testid="hero-preview-image"
                        />
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        </section>

        <section className="bg-[linear-gradient(145deg,rgba(239,246,255,0.52)_0%,rgba(248,250,252,0.92)_42%,rgba(255,255,255,1)_100%)] py-20 dark:bg-[linear-gradient(145deg,rgba(30,41,59,0.82)_0%,rgba(15,23,42,0.96)_54%,rgba(15,23,42,1)_100%)]" data-testid="how-it-works-section">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <SectionHeading
              eyebrow="Nasıl çalışır"
              title="Üç net adımda ritmini kur"
              description="Başlangıç akışı kısa kalır; asıl odak planın, görünürlüğün ve çalışmayı sürdürebilmen üzerinedir."
              align="center"
              testIdPrefix="how-it-works"
            />

            <div className="mt-12 grid gap-6 lg:grid-cols-3" data-testid="how-it-works-grid">
              {steps.map((step, index) => {
                const Icon = step.icon;

                return (
                  <Card
                    key={step.number}
                    className="border-border/70 bg-card/95 shadow-sm"
                    data-testid={`step-card-${index}`}
                  >
                    <CardContent className="p-6 sm:p-7">
                      <div className="flex items-start justify-between gap-4">
                        <div>
                          <div
                            className="inline-flex h-11 w-11 items-center justify-center rounded-2xl bg-secondary text-sm font-semibold text-foreground"
                            data-testid={`step-number-${index}`}
                          >
                            {step.number}
                          </div>
                          <h3
                            className="mt-5 font-display text-2xl font-semibold text-slate-900 dark:text-white"
                            data-testid={`step-title-${index}`}
                          >
                            {step.title}
                          </h3>
                        </div>
                        <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-border/70 bg-slate-50 text-slate-700 dark:bg-slate-950 dark:text-slate-200">
                          <Icon className="h-5 w-5" />
                        </div>
                      </div>
                      <p
                        className="mt-4 text-base leading-7 text-slate-600 dark:text-slate-300"
                        data-testid={`step-description-${index}`}
                      >
                        {step.description}
                      </p>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </div>
        </section>

        <section
          className="border-y border-border/60 bg-slate-50 py-20 dark:bg-slate-950"
          data-testid="study-rooms-section"
        >
          <div className="mx-auto grid max-w-7xl items-center gap-10 px-4 sm:px-6 lg:grid-cols-[minmax(0,0.92fr)_minmax(0,1.08fr)] lg:px-8">
            <div data-testid="study-rooms-copy-column">
              <SectionHeading
                eyebrow="Online çalışma odaları"
                title="Yalnız değil, birlikte odakta kal"
                description="İzlek'in en güçlü taraflarından biri ortak çalışma hissini ürün deneyiminin merkezinde tutmasıdır. Senkron sayaç, anlık sohbet ve aynı hedefe yürüyen öğrenciler tek akışta buluşur."
                testIdPrefix="study-rooms"
              />

              <div className="mt-8 space-y-4" data-testid="study-rooms-highlight-list">
                {roomHighlights.map((item, index) => (
                  <div
                    key={item}
                    className="flex items-start gap-3 rounded-2xl border border-border/70 bg-white/90 px-4 py-4 shadow-sm dark:bg-slate-900/90"
                    data-testid={`study-rooms-highlight-${index}`}
                  >
                    <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-secondary text-foreground">
                      <CheckCircle2 className="h-4 w-4" />
                    </div>
                    <p className="text-sm leading-6 text-slate-600 dark:text-slate-300">{item}</p>
                  </div>
                ))}
              </div>

              <div className="mt-8 grid gap-3 sm:grid-cols-3" data-testid="study-rooms-signal-grid">
                {roomSignals.map((signal, index) => (
                  <div
                    key={signal.label}
                    className="rounded-2xl border border-border/70 bg-white/90 p-4 shadow-sm dark:bg-slate-900/90"
                    data-testid={`study-rooms-signal-${index}`}
                  >
                    <p className="text-xs uppercase tracking-[0.16em] text-slate-500 dark:text-slate-400">
                      {signal.label}
                    </p>
                    <p
                      className="mt-2 text-lg font-semibold text-slate-900 dark:text-white"
                      data-testid={`study-rooms-signal-value-${index}`}
                    >
                      {signal.value}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            <Card
              className="flex w-full max-w-[560px] items-center justify-end overflow-visible border-0 bg-transparent p-0 shadow-none"
              data-testid="study-rooms-preview-card"
            >
              <CardContent className="flex w-full items-center justify-end p-0">
                <div className="flex w-full max-w-[560px] items-center justify-end bg-transparent p-0 shadow-none" data-testid="study-rooms-image-frame">
                    <img
                      src={ROOM_PREVIEW_IMAGE}
                      alt="İzlek online çalışma odası önizlemesi"
                      className="m-0 block h-auto w-full max-w-[560px] rounded-[24px] p-0 object-contain"
                      data-testid="study-rooms-image"
                    />
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        <section className="bg-white py-20 dark:bg-slate-900" data-testid="features-section">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <SectionHeading
              eyebrow="Öne çıkanlar"
              title="Planlama ve kullanım tek ürün dilinde buluşuyor"
              description="Landing tarafı daha sakin ve daha pazarlama odaklı kalırken; ürünün asıl gücünü gösteren temel özellikler kısa ve net biçimde öne çıkarılır."
              align="center"
              testIdPrefix="features"
            />

            <div className="mt-12 grid gap-6 md:grid-cols-2 xl:grid-cols-3" data-testid="features-grid">
              {features.map((feature, index) => {
                const Icon = feature.icon;

                return (
                  <Card
                    key={feature.title}
                    className={`border shadow-[0_14px_28px_-24px_rgba(15,23,42,0.14)] transition-[border-color,box-shadow,transform] duration-200 hover:-translate-y-0.5 hover:shadow-[0_22px_38px_-24px_rgba(15,23,42,0.18)] ${feature.cardClassName}`}
                    data-testid={`feature-card-${index}`}
                  >
                    <CardContent className="p-6 sm:p-7">
                      <div className="flex items-start gap-4">
                        <div
                          className={`flex h-11 w-11 items-center justify-center rounded-2xl ${feature.iconClassName}`}
                          data-testid={`feature-icon-${index}`}
                        >
                          <Icon className="h-5 w-5" />
                        </div>
                      </div>
                      <h3
                        className="mt-5 font-display text-2xl font-semibold text-slate-900 dark:text-white"
                        data-testid={`feature-title-${index}`}
                      >
                        {feature.title}
                      </h3>
                      <p
                        className="mt-3 text-base leading-7 text-slate-600 dark:text-slate-300"
                        data-testid={`feature-description-${index}`}
                      >
                        {feature.description}
                      </p>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </div>
        </section>

        <section className="bg-slate-50 py-20 dark:bg-slate-950" data-testid="faq-section">
          <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
            <SectionHeading
              eyebrow="Kısa sorular"
              title="Merak edilenleri sade biçimde toparladık"
              description="Landing akışını uzatmadan, başlarken en çok sorulan noktaları burada bırakıyoruz."
              align="center"
              testIdPrefix="faq"
            />

            <div className="mt-10 space-y-4" data-testid="faq-list">
              {faqs.map((faq, index) => {
                const isOpen = openFaq === index;

                return (
                  <Card
                    key={faq.question}
                    className="border-border/70 bg-white/95 shadow-sm dark:bg-slate-900/95"
                    data-testid={`faq-card-${index}`}
                  >
                    <CardContent className="p-0">
                      <button
                        type="button"
                        onClick={() => setOpenFaq(isOpen ? null : index)}
                        className="flex w-full items-center justify-between gap-4 px-6 py-5 text-left sm:px-7"
                        aria-expanded={isOpen}
                        data-testid={`faq-toggle-${index}`}
                      >
                        <span
                          className="pr-4 text-base font-semibold text-slate-900 dark:text-white"
                          data-testid={`faq-question-${index}`}
                        >
                          {faq.question}
                        </span>
                        <span
                          className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-border/70 bg-slate-50 text-slate-500 transition-transform dark:bg-slate-950 dark:text-slate-300 ${
                            isOpen ? "rotate-180" : "rotate-0"
                          }`}
                        >
                          <ChevronDown className="h-4 w-4" />
                        </span>
                      </button>
                      {isOpen && (
                        <div
                          className="border-t border-border/70 px-6 py-5 sm:px-7"
                          data-testid={`faq-answer-${index}`}
                        >
                          <p className="text-sm leading-7 text-slate-600 dark:text-slate-300">{faq.answer}</p>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </div>
        </section>

        <section className="bg-slate-50 pb-20 dark:bg-slate-950" data-testid="final-cta-section">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="overflow-hidden rounded-[2rem] border border-border/70 bg-slate-900 px-6 py-12 text-white shadow-[0_24px_60px_-42px_rgba(15,23,42,0.55)] sm:px-10 sm:py-14 lg:px-14">
              <div className="flex flex-col gap-10 lg:flex-row lg:items-end lg:justify-between">
                <div className="max-w-3xl" data-testid="final-cta-copy">
                  <div
                    className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.18em] text-white/90"
                    data-testid="final-cta-badge"
                  >
                    <Sparkles className="h-3.5 w-3.5" />
                    İzlek ile daha düzenli bir başlangıç
                  </div>
                  <h2
                    className="mt-6 font-display text-4xl font-semibold tracking-tight sm:text-5xl"
                    data-testid="final-cta-title"
                  >
                    Sakin, rafine ve ürün odaklı bir çalışma alanıyla başla.
                  </h2>
                  <p
                    className="mt-5 max-w-2xl text-base leading-7 text-white/75 sm:text-lg"
                    data-testid="final-cta-description"
                  >
                    Planını kur, oda ritmine dahil ol ve ne çalışacağına karar vermek yerine gerçekten çalışmaya geç.
                  </p>
                </div>

                <div className="flex w-full flex-col gap-3 sm:w-auto sm:flex-row" data-testid="final-cta-actions">
                  <Button
                    size="lg"
                    className="h-12 rounded-xl bg-white px-7 text-sm text-slate-950 hover:bg-slate-100"
                    onClick={() => navigate("/register")}
                    data-testid="final-cta-start-button"
                  >
                    Hemen Başla
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                  <Button
                    size="lg"
                    variant="outline"
                    className="h-12 rounded-xl border-white/20 bg-white/5 px-7 text-sm text-white hover:bg-white/10 hover:text-white"
                    onClick={() => navigate("/login")}
                    data-testid="final-cta-login-button"
                  >
                    Giriş Yap
                  </Button>
                </div>
              </div>

              <div className="mt-10 grid gap-3 sm:grid-cols-3" data-testid="final-cta-proof-grid">
                <div
                  className="rounded-2xl border border-white/10 bg-white/5 px-4 py-4 text-sm text-white/80"
                  data-testid="final-cta-proof-0"
                >
                  Kurulum akışı kısa, ürün deneyimi net.
                </div>
                <div
                  className="rounded-2xl border border-white/10 bg-white/5 px-4 py-4 text-sm text-white/80"
                  data-testid="final-cta-proof-1"
                >
                  Çalışma odaları motivasyonu görünür kılar.
                </div>
                <div
                  className="rounded-2xl border border-white/10 bg-white/5 px-4 py-4 text-sm text-white/80"
                  data-testid="final-cta-proof-2"
                >
                  Dashboard ile aynı ürün ailesi hissi korunur.
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer
        className="border-t border-border/60 bg-white py-8 dark:bg-slate-950"
        data-testid="landing-footer"
      >
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 text-sm text-slate-500 sm:px-6 md:flex-row md:items-center md:justify-between lg:px-8 dark:text-slate-400">
          <div className="flex items-center gap-3" data-testid="landing-footer-brand">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-border/70 bg-card text-slate-900 dark:bg-background/70 dark:text-white">
              <JourneyMark idSuffix="footer" />
            </div>
            <div>
              <div
                className="font-display text-base font-semibold text-slate-900 dark:text-white"
                data-testid="landing-footer-brand-name"
              >
                izlek
              </div>
              <div data-testid="landing-footer-brand-copy">çalışma ritmini netleştiren ürün alanı</div>
            </div>
          </div>
          <div className="text-left md:text-right" data-testid="landing-footer-meta">
            <p data-testid="landing-footer-copyright">© İzlek. Tüm hakları saklıdır.</p>
            <p className="mt-1" data-testid="landing-footer-note">
              Planını gör, ritmini koru, odağını dağıtma.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
