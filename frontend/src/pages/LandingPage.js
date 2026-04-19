import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import {
  ArrowRight,
  BarChart3,
  Flame,
  ListChecks,
  MessageSquare,
  PlayCircle,
  Timer,
  Trophy,
  Users,
} from "lucide-react";

const HOW_IT_WORKS_STEPS = [
  {
    icon: Users,
    title: "Oda oluştur veya katıl",
    description: "Tek tıkla bir odaya gir, çalışma ritmine hemen dahil ol.",
  },
  {
    icon: Timer,
    title: "Timer başlat",
    description: "Ortak sayaçla herkes aynı odak akışında kalır.",
  },
  {
    icon: PlayCircle,
    title: "Birlikte odaklan",
    description: "Sessiz oda hissiyle dikkat dağılmadan çalışmayı sürdür.",
  },
];

const OTHER_FEATURES = [
  {
    icon: BarChart3,
    title: "İlerleme takibi",
    description: "Çalışma süreni ve tamamlanan görevlerini tek bakışta gör.",
  },
  {
    icon: Flame,
    title: "Streak / motivasyon",
    description: "Düzenli çalışma serini koru, ritmini canlı tut.",
  },
  {
    icon: Trophy,
    title: "Liderlik tablosu",
    description: "Topluluk içindeki yerini sade bir sıralama ile takip et.",
  },
  {
    icon: ListChecks,
    title: "Program oluşturma",
    description: "Hedefine göre günlük ve haftalık çalışma planını netleştir.",
  },
];

function HeroRoomMockup() {
  return (
    <div
      className="relative mx-auto w-full max-w-[560px] rounded-[28px] border border-[#EDEEF3] bg-[#FFFFFF] p-5 shadow-[0_20px_50px_-36px_rgba(79,70,229,0.25)] sm:p-6"
      data-testid="landing-hero-mockup"
    >
      <div className="pointer-events-none absolute -right-8 -top-8 h-28 w-28 rounded-full bg-[#4F46E5]/10 blur-2xl" />

      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[#6B7280]">Sessiz Oda</p>
        <span className="rounded-full bg-[#EEF2FF] px-2.5 py-1 text-[11px] font-medium text-[#4F46E5]">18 kişi aktif</span>
      </div>

      <div className="mt-5 flex items-center gap-2" data-testid="landing-mockup-participants">
        {["A", "B", "C", "+15"].map((item, index) => (
          <div
            key={`${item}-${index}`}
            className="flex h-9 w-9 items-center justify-center rounded-full bg-[#F3F4F6] text-xs font-semibold text-[#374151]"
          >
            {item}
          </div>
        ))}
      </div>

      <div className="mt-7 rounded-2xl bg-[#F8FAFF] px-4 py-6 text-center">
        <p className="text-xs uppercase tracking-[0.14em] text-[#6B7280]">Ortak Timer</p>
        <p className="mt-2 text-5xl font-semibold tracking-tight text-[#111827] sm:text-6xl">24:10</p>
      </div>

      <div className="mt-6 space-y-2.5" data-testid="landing-mockup-chat">
        <div className="ml-auto max-w-[78%] rounded-2xl bg-[#EEF2FF] px-3 py-2 text-sm text-[#3730A3]">
          Bugün 2 pomodoro daha 🚀
        </div>
        <div className="max-w-[82%] rounded-2xl bg-[#F3F4F6] px-3 py-2 text-sm text-[#374151]">
          Başlıyorum, 25 dk odak 💪
        </div>
      </div>
    </div>
  );
}

function RoomExperienceSurface() {
  return (
    <div className="rounded-[28px] border border-[#ECEEF4] bg-white p-5 shadow-[0_20px_45px_-35px_rgba(17,24,39,0.15)] sm:p-6" data-testid="landing-room-experience-surface">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-lg font-semibold text-[#111827]">Canlı oda deneyimi</h3>
        <span className="rounded-full bg-[#EEF2FF] px-3 py-1 text-xs font-medium text-[#4F46E5]">Senkron akış</span>
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-2xl bg-[#F8FAFF] px-4 py-5">
          <p className="text-xs uppercase tracking-[0.14em] text-[#6B7280]">Timer</p>
          <p className="mt-2 text-4xl font-semibold tracking-tight text-[#111827]">24:10</p>
          <p className="mt-2 text-sm text-[#6B7280]">Herkes aynı sürede başlar, birlikte odakta kalır.</p>
        </div>

        <div className="rounded-2xl bg-[#F9FAFB] px-4 py-5">
          <p className="text-xs uppercase tracking-[0.14em] text-[#6B7280]">Katılımcılar</p>
          <div className="mt-2 flex items-center gap-2">
            {["A", "E", "S", "+15"].map((value, index) => (
              <span key={`${value}-${index}`} className="flex h-8 w-8 items-center justify-center rounded-full bg-white text-xs font-semibold text-[#374151]">
                {value}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-3 rounded-2xl bg-[#F9FAFB] px-4 py-4">
        <p className="text-xs uppercase tracking-[0.14em] text-[#6B7280]">Minimal sohbet</p>
        <div className="mt-2 space-y-2 text-sm text-[#374151]">
          <p className="w-fit rounded-xl bg-white px-3 py-1.5">5 dk mola sonrası devam 🔁</p>
          <p className="w-fit rounded-xl bg-[#EEF2FF] px-3 py-1.5 text-[#3730A3]">Tamam, yeni seans başlasın.</p>
        </div>
      </div>
    </div>
  );
}

export default function LandingPage() {
  const navigate = useNavigate();
  const { currentUser } = useAuth();

  useEffect(() => {
    if (currentUser) {
      navigate("/dashboard");
    }
  }, [currentUser, navigate]);

  return (
    <div className="min-h-screen bg-[#F7F7F7] text-[#111827]" data-testid="landing-page">
      <nav className="sticky top-0 z-50 border-b border-[#ECEEF4] bg-[#F7F7F7]/95 backdrop-blur" data-testid="landing-nav">
        <div className="mx-auto flex h-16 w-full max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          <button type="button" onClick={() => navigate("/")} className="text-left" data-testid="landing-brand-button">
            <p className="font-display text-[1.35rem] font-semibold tracking-[0.06em] text-[#111827]" data-testid="landing-brand-name">izlek</p>
            <p className="text-xs text-[#6B7280]" data-testid="landing-brand-tagline">birlikte odaklanma alanı</p>
          </button>

          <div className="flex items-center gap-2" data-testid="landing-nav-actions">
            <Button variant="ghost" onClick={() => navigate("/login")} className="text-[#374151] hover:bg-[#EEF2FF]" data-testid="landing-nav-login-button">
              Giriş Yap
            </Button>
            <Button
              onClick={() => navigate("/register")}
              className="bg-[#4F46E5] text-white shadow-[0_12px_28px_-20px_rgba(79,70,229,0.65)] hover:bg-[#6366F1]"
              data-testid="landing-nav-register-button"
            >
              Hemen Başla
            </Button>
          </div>
        </div>
      </nav>

      <main>
        <section className="px-4 pb-16 pt-14 sm:px-6 sm:pb-20 sm:pt-20 lg:px-8" data-testid="landing-hero-section">
          <div className="mx-auto grid w-full max-w-7xl items-center gap-12 lg:grid-cols-[1fr_minmax(0,560px)] lg:gap-14">
            <div>
              <h1 className="font-display text-4xl font-semibold leading-tight tracking-tight text-[#111827] sm:text-5xl lg:text-6xl" data-testid="hero-title">
                Odaklanmak için yalnız değilsin.
              </h1>
              <p className="mt-6 max-w-2xl text-base leading-7 text-[#6B7280] sm:text-lg" data-testid="hero-description">
                İzlek, birlikte sessizce çalıştığın odak odaları sunar. Ortak timer, canlı oda hissi ve sade sohbet tek yerde.
              </p>

              <div className="mt-8 flex flex-col gap-3 sm:flex-row" data-testid="hero-cta-group">
                <Button
                  size="lg"
                  onClick={() => navigate("/rooms")}
                  className="h-12 rounded-xl bg-[#4F46E5] px-7 text-sm font-medium text-white hover:bg-[#6366F1]"
                  data-testid="hero-create-room-button"
                >
                  Oda oluştur
                  <ArrowRight className="h-4 w-4" />
                </Button>
                <Button
                  size="lg"
                  variant="outline"
                  onClick={() => navigate("/rooms")}
                  className="h-12 rounded-xl border-[#E5E7EB] bg-white px-7 text-sm text-[#374151] hover:bg-[#F3F4F6]"
                  data-testid="hero-join-room-button"
                >
                  Odaya katıl
                </Button>
              </div>
            </div>

            <HeroRoomMockup />
          </div>
        </section>

        <section className="px-4 py-16 sm:px-6 sm:py-20 lg:px-8" data-testid="how-it-works-section">
          <div className="mx-auto w-full max-w-7xl">
            <div className="max-w-2xl">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#4F46E5]">Nasıl çalışır</p>
              <h2 className="mt-4 font-display text-3xl font-semibold tracking-tight text-[#111827] sm:text-4xl">Üç adımda odak ritmi</h2>
            </div>

            <div className="mt-10 grid gap-8 md:grid-cols-3 md:gap-10" data-testid="how-it-works-grid">
              {HOW_IT_WORKS_STEPS.map((step, index) => {
                const Icon = step.icon;
                return (
                  <div key={step.title} className="relative md:pr-6" data-testid={`how-it-works-item-${index}`}>
                    {index < HOW_IT_WORKS_STEPS.length - 1 && (
                      <span className="absolute right-0 top-3 hidden h-[calc(100%-24px)] w-px bg-[#E5E7EB] md:block" aria-hidden="true" />
                    )}
                    <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[#EEF2FF] text-[#4F46E5]">
                      <Icon className="h-5 w-5" />
                    </div>
                    <h3 className="mt-4 text-lg font-semibold text-[#111827]">{step.title}</h3>
                    <p className="mt-2 text-sm leading-6 text-[#6B7280]">{step.description}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        <section className="px-4 py-16 sm:px-6 sm:py-20 lg:px-8" data-testid="room-experience-section">
          <div className="mx-auto grid w-full max-w-7xl items-center gap-10 lg:grid-cols-[1.05fr_0.95fr]">
            <RoomExperienceSurface />

            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#4F46E5]">Ürünü göster</p>
              <h2 className="mt-4 font-display text-3xl font-semibold tracking-tight text-[#111827] sm:text-4xl">Oda deneyimi bir bakışta net</h2>
              <p className="mt-4 text-base leading-7 text-[#6B7280]">
                Timer, katılımcılar ve minimal sohbet tek yüzeyde buluşur. Karmaşa olmadan sadece çalışmaya odaklanırsın.
              </p>
              <div className="mt-6 space-y-2 text-sm text-[#4B5563]">
                <p className="flex items-center gap-2"><Timer className="h-4 w-4 text-[#4F46E5]" /> Ortak timer ile senkron odak</p>
                <p className="flex items-center gap-2"><Users className="h-4 w-4 text-[#4F46E5]" /> Katılımcı görünürlüğü</p>
                <p className="flex items-center gap-2"><MessageSquare className="h-4 w-4 text-[#4F46E5]" /> Sade ve dikkat dağıtmayan chat</p>
              </div>
            </div>
          </div>
        </section>

        <section className="px-4 py-16 sm:px-6 sm:py-20 lg:px-8" data-testid="other-features-section">
          <div className="mx-auto w-full max-w-7xl">
            <div className="max-w-3xl">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#4F46E5]">Diğer özellikler</p>
              <h2 className="mt-4 font-display text-3xl font-semibold tracking-tight text-[#111827] sm:text-4xl">İzlek sadece odalardan ibaret değil</h2>
            </div>

            <div className="mt-10 grid gap-5 sm:grid-cols-2" data-testid="other-features-grid">
              {OTHER_FEATURES.map((feature, index) => {
                const Icon = feature.icon;
                return (
                  <div key={feature.title} className="rounded-2xl bg-white px-5 py-5 shadow-[0_14px_32px_-26px_rgba(17,24,39,0.16)]" data-testid={`other-feature-${index}`}>
                    <div className="flex items-center gap-3">
                      <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#EEF2FF] text-[#4F46E5]">
                        <Icon className="h-5 w-5" />
                      </span>
                      <h3 className="text-base font-semibold text-[#111827]">{feature.title}</h3>
                    </div>
                    <p className="mt-3 text-sm leading-6 text-[#6B7280]">{feature.description}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        <section className="px-4 pb-20 pt-6 sm:px-6 lg:px-8" data-testid="final-cta-section">
          <div className="mx-auto w-full max-w-7xl rounded-[28px] bg-white px-6 py-12 text-center shadow-[0_24px_52px_-40px_rgba(79,70,229,0.35)] sm:px-10">
            <h2 className="font-display text-3xl font-semibold tracking-tight text-[#111827] sm:text-4xl" data-testid="final-cta-title">
              Sessiz bir odada, birlikte odaklanmaya başla.
            </h2>
            <p className="mx-auto mt-4 max-w-2xl text-base leading-7 text-[#6B7280]" data-testid="final-cta-description">
              Hazırsan bir oda aç ve çalışma ritmini şimdi netleştir.
            </p>
            <div className="mt-8" data-testid="final-cta-actions">
              <Button
                size="lg"
                onClick={() => navigate("/rooms")}
                className="h-12 rounded-xl bg-[#4F46E5] px-8 text-sm font-medium text-white hover:bg-[#6366F1]"
                data-testid="final-cta-create-room-button"
              >
                Oda oluştur
              </Button>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
