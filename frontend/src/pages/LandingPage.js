import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import {
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

function FocusRoomMockup({ timerValue = "24:32", className = "" }) {
  return (
    <div
      className={`relative mx-auto w-full max-w-[580px] rounded-2xl border border-[#E5E7EB] bg-[#FFFFFF] px-7 py-8 shadow-sm lg:scale-[1.06] ${className}`.trim()}
      data-testid="landing-room-mockup"
    >
      <div className="pointer-events-none absolute -inset-4 -z-10 rounded-[28px] bg-[#4F46E5]/6 blur-3xl" />

      <span className="absolute right-5 top-5 inline-flex items-center gap-1.5 rounded-full border border-[#E5E7EB] bg-white px-2.5 py-1 text-[11px] font-medium text-[#6B7280]">
        <span className="h-1.5 w-1.5 rounded-full bg-[#4F46E5]" />
        Canlı • 18 kişi
      </span>

      <div className="text-center">
        <p className="text-[11px] font-medium uppercase tracking-[0.16em] text-[#6B7280]">Ortak Timer</p>
        <p className="mt-3 text-[88px] font-bold leading-[0.95] tracking-tight text-[#111111] sm:text-[100px]">{timerValue}</p>
        <p className="mt-3 text-sm text-[#6B7280]">Odak modundasın</p>
      </div>

      <div className="mt-7 flex items-center justify-center gap-2.5" data-testid="landing-mockup-participants">
        {["A", "B", "C", "+15"].map((item, index) => (
          <span
            key={`${item}-${index}`}
            className="flex h-8 w-8 items-center justify-center rounded-full bg-[#F3F4F6] text-[11px] font-semibold text-[#374151]"
          >
            {item}
          </span>
        ))}
      </div>

      <div className="mt-6 space-y-2">
        <p className="inline-flex items-center gap-1.5 text-xs text-[#6B7280]">
          <Timer className="h-3.5 w-3.5 text-[#4F46E5]" />
          Sessiz oda akışı aktif
        </p>
        <p className="flex items-center gap-2 text-xs leading-none text-[#6B7280]">
          <MessageSquare className="h-4 w-4 mr-2 text-[#4F46E5]" />
          Sade sohbet ile devam et
        </p>
      </div>
    </div>
  );
}

export default function LandingPage() {
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    if (currentUser) {
      navigate("/dashboard");
    }
  }, [currentUser, navigate]);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 12);
    };

    handleScroll();
    window.addEventListener("scroll", handleScroll, { passive: true });

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  return (
    <div className="min-h-screen bg-[#F7F7F7] text-[#111111]" data-testid="landing-page">
      <nav
        className={`sticky top-0 z-50 border-b transition-all duration-200 ${
          isScrolled
            ? "border-gray-200/90 bg-white/95 shadow-sm backdrop-blur-sm"
            : "border-gray-100 bg-white"
        }`}
        data-testid="landing-nav"
      >
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-4 py-5 sm:px-6 lg:px-8">
          <button
            type="button"
            onClick={() => navigate("/")}
            className="flex items-center text-left transition-opacity duration-200 hover:opacity-90"
            data-testid="landing-brand-button"
          >
            <p className="font-display text-xl font-bold tracking-tight text-gray-900" data-testid="landing-brand-name">izlek</p>
          </button>

          <div className="flex items-center gap-8" data-testid="landing-nav-actions">
            <Button
              variant="ghost"
              onClick={() => navigate("/login")}
              className="font-medium text-gray-700 transition-colors duration-200 hover:bg-transparent hover:text-black"
              data-testid="landing-nav-login-button"
            >
              Giriş Yap
            </Button>
            <Button
              onClick={() => navigate("/register")}
              className="rounded-xl bg-indigo-600 px-5 py-2.5 text-white font-medium shadow-sm transition-all duration-200 hover:bg-indigo-700 hover:shadow-md"
              data-testid="landing-nav-register-button"
            >
              Hemen Başla
            </Button>
          </div>
        </div>
      </nav>

      <main>
        <section className="relative overflow-hidden border-b border-[#E5E7EB] bg-[#F7F7F7] px-4 py-24 sm:px-6 lg:py-28 lg:px-8" data-testid="landing-hero-section">
          <div className="pointer-events-none absolute -left-20 top-16 h-44 w-44 rounded-full bg-[#4F46E5]/6 blur-3xl" />
          <div className="pointer-events-none absolute right-0 top-10 h-36 w-36 rounded-full bg-[#111111]/[0.03] blur-3xl" />

          <div className="relative mx-auto grid w-full max-w-7xl items-center gap-12 lg:grid-cols-[1fr_minmax(0,580px)] lg:gap-16">
            <div>
              <h1 className="font-display text-4xl font-bold leading-tight tracking-tight text-[#111111] sm:text-5xl lg:text-6xl" data-testid="hero-title">
                Birlikte odaklan, gerçekten ilerle.
              </h1>
              <p className="mt-5 max-w-xl text-lg font-medium leading-8 text-[#6B7280]" data-testid="hero-description">
                Sessiz odalarda dikkat dağılmadan çalış. Ritmini koru.
              </p>

              <div className="mt-8" data-testid="hero-cta-group">
                <Button
                  size="lg"
                  onClick={() => navigate("/rooms")}
                  className="h-12 rounded-xl bg-[#4F46E5] px-8 text-sm font-medium text-white transition-colors duration-200 hover:bg-[#4338CA]"
                  data-testid="hero-create-room-button"
                >
                  Hemen başla
                </Button>
              </div>

              <p className="mt-4 text-sm text-[#6B7280]" data-testid="hero-trust-line">Şu anda 120+ kişi odaklanıyor</p>
            </div>

            <FocusRoomMockup />
          </div>
        </section>

        <section className="border-b border-[#E5E7EB] bg-[#FFFFFF] px-4 py-24 sm:px-6 lg:py-28 lg:px-8" data-testid="how-it-works-section">
          <div className="mx-auto w-full max-w-7xl">
            <div className="max-w-2xl">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#4F46E5]">Nasıl çalışır</p>
              <h2 className="mt-5 font-display text-3xl font-bold tracking-tight text-[#111111] sm:text-4xl">Üç adımda odak ritmi</h2>
            </div>

            <div className="mt-10 grid gap-6 md:grid-cols-3" data-testid="how-it-works-grid">
              {HOW_IT_WORKS_STEPS.map((step, index) => {
                const Icon = step.icon;
                const number = String(index + 1).padStart(2, "0");
                return (
                  <div
                    key={step.title}
                    className="rounded-xl px-1 py-2 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-sm"
                    data-testid={`how-it-works-item-${index}`}
                  >
                    <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[#9CA3AF]">{number}</p>
                    <div className="mt-2 flex h-11 w-11 items-center justify-center rounded-lg border border-[#E5E7EB] bg-[#FAFAFA] text-[#4F46E5]">
                      <Icon className="h-[22px] w-[22px]" />
                    </div>
                    <h3 className="mt-3 text-lg font-semibold text-[#111111]">{step.title}</h3>
                    <p className="mt-2 text-sm leading-6 text-[#6B7280]">{step.description}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        <section className="border-b border-[#E5E7EB] bg-[#FAFAFA] px-4 py-24 sm:px-6 lg:py-28 lg:px-8" data-testid="room-experience-section">
          <div className="mx-auto grid w-full max-w-7xl items-center gap-10 lg:grid-cols-[1.02fr_0.98fr]">
            <div>
              <FocusRoomMockup timerValue="24:10" className="max-w-[540px]" />
              <div className="mt-4 flex flex-wrap gap-2" data-testid="room-experience-chips">
                {[
                  "Gerçek zamanlı",
                  "Senkron",
                  "Dikkat dağıtmaz",
                ].map((chip) => (
                  <span key={chip} className="rounded-full border border-[#E5E7EB] bg-white px-3 py-1 text-xs text-[#6B7280]">
                    {chip}
                  </span>
                ))}
              </div>
            </div>

            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#4F46E5]">Ürünü göster</p>
              <h2 className="mt-5 font-display text-3xl font-bold tracking-tight text-[#111111] sm:text-4xl">Oda deneyimi tek bakışta net</h2>
              <ul className="mt-6 space-y-3 text-lg leading-8 text-[#6B7280]">
                <li className="flex items-center gap-2">
                  <Timer className="h-[18px] w-[18px] text-[#4F46E5]" />
                  Ortak timer ile senkron odak
                </li>
                <li className="flex items-center gap-2">
                  <Users className="h-[18px] w-[18px] text-[#4F46E5]" />
                  Katılımcı görünürlüğü
                </li>
                <li className="flex items-center gap-2">
                  <MessageSquare className="h-[18px] w-[18px] text-[#4F46E5]" />
                  Dikkat dağıtmayan sade chat
                </li>
              </ul>
            </div>
          </div>
        </section>

        <section className="border-b border-[#E5E7EB] bg-[#FFFFFF] px-4 py-24 sm:px-6 lg:py-28 lg:px-8" data-testid="other-features-section">
          <div className="mx-auto w-full max-w-7xl">
            <div className="max-w-3xl">
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#4F46E5]">Diğer özellikler</p>
              <h2 className="mt-5 font-display text-3xl font-bold tracking-tight text-[#111111] sm:text-4xl">İzlek sadece odalardan ibaret değil</h2>
            </div>

            <div className="mt-10 grid gap-4 sm:grid-cols-2" data-testid="other-features-grid">
              {OTHER_FEATURES.map((feature, index) => {
                const Icon = feature.icon;
                return (
                  <div
                    key={feature.title}
                    className="rounded-xl border border-[#E5E7EB] bg-[#FFFFFF] px-5 py-5 transition-colors duration-200 hover:bg-[#FAFAFA]"
                    data-testid={`other-feature-${index}`}
                  >
                    <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#EEF2FF] text-[#4F46E5]">
                      <Icon className="h-[18px] w-[18px]" />
                    </div>
                    <h3 className="mt-3 text-base font-semibold text-[#111111]">{feature.title}</h3>
                    <p className="mt-1.5 text-sm leading-7 text-[#6B7280]">{feature.description}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        <section className="bg-[#FAFAFA] px-4 py-20 sm:px-6 lg:py-24 lg:px-8" data-testid="final-cta-section">
          <div className="mx-auto w-full max-w-5xl rounded-2xl border border-[#E5E7EB] bg-[#FFFFFF] px-8 py-12 text-center">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#6B7280]">Bugün başla</p>
            <h2 className="mt-4 font-display text-3xl font-bold tracking-tight text-[#111111] sm:text-4xl" data-testid="final-cta-title">
              Hazırsan, odaklanmaya başla.
            </h2>
            <div className="mt-7" data-testid="final-cta-actions">
              <Button
                size="lg"
                onClick={() => navigate("/rooms")}
                className="h-14 rounded-xl bg-[#4F46E5] px-10 text-sm font-medium text-white shadow-sm transition-colors duration-200 hover:bg-[#4338CA]"
                data-testid="final-cta-create-room-button"
              >
                Oda oluştur
              </Button>
            </div>
          </div>
        </section>
      </main>

      <footer className="bg-[#FAFAFA] px-4 pb-8 sm:px-6 lg:px-8" data-testid="landing-footer">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between border-t border-[#E5E7EB] pt-5 text-xs text-[#9CA3AF]">
          <p>© 2026 İzlek</p>
          <p>Odaklanmak için birlikte.</p>
        </div>
      </footer>
    </div>
  );
}

