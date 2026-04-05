import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import axios from "axios";
import { API } from "@/App";
import { ArrowLeft, ArrowRight, Check } from "lucide-react";
import { saveProfile, savePrograms, getProgramDraft, saveProgramDraft, clearProgramDraft, setOnboardingCompleted } from "@/lib/storage";
import { useAuth } from "@/contexts/AuthContext";
import { cn } from "@/lib/utils";

const HANDLE_PATTERN = /^[a-z0-9_]{3,20}$/;
const DEFAULT_EXAM_GOAL = "TYT";
const DEFAULT_DAILY_HOURS = "2-4";
const CURRENT_DRAFT_VERSION = 2;
const TOTAL_STEPS = 5;

const CLASS_STATUS_OPTIONS = [
  {
    value: "11",
    title: "11. sınıf",
    description: "Temelini oturtup düzenli ilerleyenler için.",
  },
  {
    value: "12",
    title: "12. sınıf",
    description: "Sınava daha yakın bir tempoda ilerleyenler için.",
  },
  {
    value: "mezun",
    title: "Mezun",
    description: "Planını yeniden kurup ritmini toparlayanlar için.",
  },
];

const STUDY_FIELD_OPTIONS = [
  {
    value: "Sayısal",
    title: "Sayısal",
    description: "Matematik ve fen ağırlıklı ilerliyorsan.",
  },
  {
    value: "EA",
    title: "Eşit Ağırlık",
    description: "Matematik ve sözel dersleri dengeli götürüyorsan.",
  },
  {
    value: "Sözel",
    title: "Sözel",
    description: "Sözel derslerde derinleşiyorsan.",
  },
  {
    value: "Dil",
    title: "Dil",
    description: "Yabancı dil odaklı hazırlanıyorsan.",
  },
];

const STUDY_FREQUENCY_OPTIONS = [
  {
    value: 2,
    title: "1–2 gün",
    description: "Daha hafif ama sürdürülebilir bir tempo.",
  },
  {
    value: 4,
    title: "3–4 gün",
    description: "Hafta içine dengeli yayılan bir düzen.",
  },
  {
    value: 6,
    title: "5–6 gün",
    description: "Yoğun ama kontrollü ilerleyen bir ritim.",
  },
  {
    value: 7,
    title: "Her gün",
    description: "Her güne kısa da olsa çalışma yerleştiriyorsan.",
  },
];

const STEP_CONTENT = {
  1: {
    eyebrow: "Profil başlangıcı",
    title: "Kendine bir takma isim belirle",
    description: "Diğer kullanıcılar seni bu isimle görecek.",
  },
  2: {
    eyebrow: "Hesap kimliği",
    title: "Kendine bir @ belirle",
    description: "Bu senin benzersiz kullanıcı adın olacak.",
  },
  3: {
    eyebrow: "Sınıf durumu",
    title: "Hangi aşamadasın?",
    description: "Şu anki sınıf durumunu tek seçimle belirt.",
  },
  4: {
    eyebrow: "Alan bilgisi",
    title: "Alanını seç",
    description: "Profilinde görünecek alan bilgisini seç.",
  },
  5: {
    eyebrow: "Çalışma ritmi",
    title: "Haftada kaç gün ders çalışıyorsun?",
    description: "İlk çalışma planın bu ritme göre hazırlanacak.",
  },
};

const createDefaultFormData = () => ({
  name: "",
  handle: "",
  class_status: "",
  study_field: "",
  exam_goal: DEFAULT_EXAM_GOAL,
  daily_hours: DEFAULT_DAILY_HOURS,
  study_days: null,
});

const clampStep = (value) => Math.min(Math.max(value || 1, 1), TOTAL_STEPS);

const getHandleValidationMessage = (handle) => {
  if (!handle) {
    return {
      tone: "default",
      text: "3-20 karakter; yalnızca küçük harf, rakam ve alt çizgi kullanabilirsin.",
    };
  }

  if (!HANDLE_PATTERN.test(handle)) {
    return {
      tone: "error",
      text: "Handle 3-20 karakter olmalı; yalnızca küçük harf, rakam ve alt çizgi içerebilir.",
    };
  }

  return {
    tone: "success",
    text: `Kullanım biçimi hazır: @${handle}`,
  };
};

function SelectionCard({ option, selected, onSelect, testId }) {
  return (
    <button
      type="button"
      onClick={() => onSelect(option.value)}
      className={cn(
        "group flex w-full items-start justify-between gap-4 rounded-2xl border px-4 py-4 text-left transition-[background-color,border-color,box-shadow,transform] duration-200",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
        selected
          ? "border-slate-900 bg-slate-900 text-slate-50 shadow-lg dark:border-slate-100 dark:bg-slate-100 dark:text-slate-900"
          : "border-border/80 bg-background/80 text-slate-900 shadow-sm hover:-translate-y-0.5 hover:border-slate-300 hover:bg-card hover:shadow-md dark:border-slate-800 dark:bg-slate-950/50 dark:text-slate-100 dark:hover:border-slate-700 dark:hover:bg-slate-900/70"
      )}
      data-testid={testId}
    >
      <div className="space-y-1" data-testid={`${testId}-content`}>
        <div className="text-base font-semibold" data-testid={`${testId}-title`}>
          {option.title}
        </div>
        <p
          className={cn(
            "text-sm leading-6",
            selected
              ? "text-slate-200 dark:text-slate-700"
              : "text-slate-600 dark:text-slate-300"
          )}
          data-testid={`${testId}-description`}
        >
          {option.description}
        </p>
      </div>

      <div
        className={cn(
          "mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border transition-[background-color,border-color,color] duration-200",
          selected
            ? "border-current bg-white/15 text-white dark:bg-slate-900 dark:text-slate-100"
            : "border-slate-300 text-transparent dark:border-slate-700"
        )}
        data-testid={`${testId}-indicator`}
      >
        <Check className="h-3.5 w-3.5" />
      </div>
    </button>
  );
}

export default function ProgramCreation() {
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  const [step, setStep] = useState(1);
  const [formData, setFormData] = useState(createDefaultFormData);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const autosaveTimerRef = useRef(null);
  const normalizedHandle = formData.handle.trim().replace(/^@+/, "").toLowerCase();
  const handleFeedback = getHandleValidationMessage(normalizedHandle);
  const currentStepContent = STEP_CONTENT[step];
  const progressValue = (step / TOTAL_STEPS) * 100;

  // Load draft on mount
  useEffect(() => {
    const draft = getProgramDraft();
    if (draft && draft.formData) {
      setFormData({
        ...createDefaultFormData(),
        ...draft.formData,
        handle: typeof draft.formData.handle === "string" ? draft.formData.handle : "",
        class_status: typeof draft.formData.class_status === "string" ? draft.formData.class_status : "",
        exam_goal: draft.formData.exam_goal || DEFAULT_EXAM_GOAL,
        daily_hours: draft.formData.daily_hours || DEFAULT_DAILY_HOURS,
        study_days: Number.isInteger(draft.formData.study_days) ? draft.formData.study_days : null,
      });
      setStep(draft.version === CURRENT_DRAFT_VERSION ? clampStep(draft.step) : 1);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Autosave draft on formData or step changes (debounced)
  useEffect(() => {
    // Clear existing timer
    if (autosaveTimerRef.current) {
      clearTimeout(autosaveTimerRef.current);
    }

    // Set new timer for autosave (500ms debounce)
    autosaveTimerRef.current = setTimeout(() => {
      saveProgramDraft({
        version: CURRENT_DRAFT_VERSION,
        formData,
        step,
        timestamp: new Date().toISOString()
      });
    }, 500);

    // Cleanup timer on unmount
    return () => {
      if (autosaveTimerRef.current) {
        clearTimeout(autosaveTimerRef.current);
      }
    };
  }, [formData, step]);

  const handleNext = () => {
    // Clear any previous errors
    setError("");
    
    // Validation
    if (step === 1 && !formData.name.trim()) {
      setError("Lütfen bir takma isim gir.");
      return;
    }
    if (step === 2 && !normalizedHandle) {
      setError("Lütfen bir @ belirle.");
      return;
    }
    if (step === 2 && !HANDLE_PATTERN.test(normalizedHandle)) {
      setError("Handle 3-20 karakter olmalı; yalnızca küçük harf, rakam ve alt çizgi içerebilir.");
      return;
    }
    if (step === 3 && !formData.class_status) {
      setError("Lütfen sınıf durumunu seç.");
      return;
    }
    if (step === 4 && !formData.study_field) {
      setError("Lütfen alanını seç.");
      return;
    }
    if (step === 5 && !formData.study_days) {
      setError("Lütfen çalışma sıklığını seç.");
      return;
    }
    
    if (step < TOTAL_STEPS) {
      setStep(step + 1);
    } else {
      handleSubmit();
    }
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError("");
    
    try {
      // Get Firebase UID
      const firebaseUid = currentUser?.uid;
      
      if (!firebaseUid) {
        setError("Kullanıcı kimliği bulunamadı. Lütfen tekrar giriş yapın.");
        setLoading(false);
        return;
      }

      // Create profile with Firebase UID
      const profileRes = await axios.post(`${API}/profiles`, {
        firebase_uid: firebaseUid,
        name: formData.name.trim(),
        username: formData.name.trim(),
        handle: normalizedHandle,
        study_field: formData.study_field || null
      });
      
      const profileId = profileRes.data.id;

      // Create program
      const programRes = await axios.post(`${API}/programs`, {
        profile_id: profileId,
        exam_goal: formData.exam_goal,
        daily_hours: formData.daily_hours,
        study_days: formData.study_days
      });

      const resolvedName = profileRes.data.username || profileRes.data.name || formData.name.trim();

      // Store profile ID and name in localStorage FIRST
      localStorage.setItem("profileId", profileId);
      localStorage.setItem("userName", resolvedName);
      
      // Save to userData storage
      saveProfile(profileId, resolvedName);
      savePrograms([programRes.data]);
      
      // Mark onboarding as completed to prevent re-showing
      setOnboardingCompleted();
      
      // Clear the draft after successful program creation
      clearProgramDraft();
      
      // Ensure all localStorage operations are flushed
      // Use a small delay to guarantee sync
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // Navigate to dashboard with replace to prevent back navigation
      navigate("/dashboard", { replace: true });
    } catch (error) {
      console.error("Error creating program:", error);

      const detail = error.response?.data?.detail;

      if (error.response?.status === 409 || (typeof detail === "string" && detail.toLowerCase().includes("handle"))) {
        setStep(2);
        setError(detail || "Bu @ zaten kullanımda.");
        return;
      }
      
      // More specific error messages
      if (error.response) {
        setError(detail || `Sunucu hatası: ${error.response.status}`);
      } else if (error.request) {
        // Network error - request was made but no response
        setError("Bağlantı hatası. Lütfen internet bağlantınızı kontrol edin.");
      } else {
        // Something else happened
        setError("Bir hata oluştu. Lütfen tekrar dene.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background px-4 py-6 text-slate-900 dark:text-slate-100 sm:px-6 sm:py-10" data-testid="program-create-page">
      <div className="mx-auto flex min-h-[calc(100vh-3rem)] max-w-2xl items-center justify-center" data-testid="program-create-page-container">
        <Card
          className="w-full border border-border/70 bg-card/95 shadow-[0_24px_60px_-38px_rgba(15,23,42,0.38)] backdrop-blur"
          data-testid="program-creation-card"
        >
          <CardHeader className="space-y-6" data-testid="program-create-header">
            <div className="space-y-3" data-testid="program-create-header-copy">
              <div className="flex items-center justify-between gap-3" data-testid="program-create-step-meta-row">
                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500 dark:text-slate-400" data-testid="program-create-step-eyebrow">
                  {currentStepContent.eyebrow}
                </p>
                <p className="text-sm font-medium text-slate-500 dark:text-slate-400" data-testid="program-create-step-counter">
                  Adım {step} / {TOTAL_STEPS}
                </p>
              </div>

              <div className="space-y-2" data-testid="program-create-title-group">
                <CardTitle className="text-4xl font-bold tracking-tight text-slate-900 dark:text-slate-100" data-testid="program-create-title">
                  {currentStepContent.title}
                </CardTitle>
                <CardDescription className="max-w-xl text-base leading-7 text-slate-600 dark:text-slate-300" data-testid="program-create-description">
                  {currentStepContent.description}
                </CardDescription>
              </div>
            </div>

            <div className="space-y-3" data-testid="program-create-progress-group">
              <div className="h-1.5 rounded-full bg-slate-200 dark:bg-slate-800" data-testid="program-create-progress-track">
                <div
                  className="h-1.5 rounded-full bg-slate-900 transition-[width] duration-300 dark:bg-slate-100"
                  style={{ width: `${progressValue}%` }}
                  data-testid="program-create-progress-bar"
                />
              </div>
            </div>
        </CardHeader>

          <CardContent className="space-y-6" data-testid="program-create-content">
          {/* Error Message */}
          {error && (
              <div
                className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-700 dark:border-red-900/60 dark:bg-red-950/30 dark:text-red-300"
                data-testid="program-create-error-message"
              >
              {error}
            </div>
          )}
          
            {step === 1 && (
              <div className="space-y-5" data-testid="program-create-step-name">
                <div className="space-y-2.5" data-testid="program-create-name-field-group">
                  <Label htmlFor="name" className="text-sm font-semibold text-slate-800 dark:text-slate-200" data-testid="program-create-name-label">
                    Takma isim
                  </Label>
                <Input
                  id="name"
                  placeholder="Örn: atlas"
                  value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="h-12 border-border/80 bg-background/90 text-base text-slate-900 placeholder:text-slate-400 focus-visible:ring-slate-400 dark:border-slate-700 dark:bg-slate-950/60 dark:text-slate-100 dark:placeholder:text-slate-500"
                    data-testid="program-create-name-input"
                    autoFocus
                />
                  <p className="text-sm text-slate-500 dark:text-slate-400" data-testid="program-create-name-helper">
                    İstediğin zaman profilinden güncelleyebilirsin.
                  </p>
              </div>
            </div>
          )}

            {step === 2 && (
              <div className="space-y-5" data-testid="program-create-step-handle">
                <div className="space-y-2.5" data-testid="program-create-handle-field-group">
                  <Label htmlFor="handle" className="text-sm font-semibold text-slate-800 dark:text-slate-200" data-testid="program-create-handle-label">
                    @handle
                  </Label>
                  <div className="relative" data-testid="program-create-handle-input-wrap">
                    <span className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-4 text-base text-slate-500 dark:text-slate-400" data-testid="program-create-handle-prefix">
                      @
                    </span>
                    <Input
                      id="handle"
                      value={formData.handle}
                      onChange={(e) => setFormData({ ...formData, handle: e.target.value.replace(/^@+/, "").toLowerCase() })}
                      placeholder="ornek_handle"
                      className={cn(
                        "h-12 border-border/80 bg-background/90 pl-8 text-base text-slate-900 placeholder:text-slate-400 focus-visible:ring-slate-400 dark:border-slate-700 dark:bg-slate-950/60 dark:text-slate-100 dark:placeholder:text-slate-500",
                        handleFeedback.tone === "error" && normalizedHandle ? "border-red-300 focus-visible:ring-red-400 dark:border-red-800" : ""
                      )}
                      data-testid="program-create-handle-input"
                      autoFocus
                    />
                  </div>
                  <p
                    className={cn(
                      "text-sm",
                      handleFeedback.tone === "error"
                        ? "text-red-600 dark:text-red-300"
                        : handleFeedback.tone === "success"
                          ? "text-emerald-600 dark:text-emerald-300"
                          : "text-slate-500 dark:text-slate-400"
                    )}
                    data-testid="program-create-handle-feedback"
                  >
                    {handleFeedback.text}
                  </p>
                  <p className="text-sm text-slate-500 dark:text-slate-400" data-testid="program-create-handle-helper">
                    Arkadaşların seni bu kullanıcı adıyla bulabilecek.
                  </p>
                </div>
              </div>
          )}

            {step === 3 && (
              <div className="grid gap-3" data-testid="program-create-step-class-status">
                {CLASS_STATUS_OPTIONS.map((option) => (
                  <SelectionCard
                    key={option.value}
                    option={option}
                    selected={formData.class_status === option.value}
                    onSelect={(value) => setFormData({ ...formData, class_status: value })}
                    testId={`program-create-class-option-${option.value}`}
                  />
                ))}
              </div>
          )}

            {step === 4 && (
              <div className="grid gap-3 sm:grid-cols-2" data-testid="program-create-step-study-field">
                {STUDY_FIELD_OPTIONS.map((option) => (
                  <SelectionCard
                    key={option.value}
                    option={option}
                    selected={formData.study_field === option.value}
                    onSelect={(value) => setFormData({ ...formData, study_field: value })}
                    testId={`program-create-field-option-${option.value.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`}
                  />
                ))}
              </div>
            )}

            {step === 5 && (
              <div className="grid gap-3 sm:grid-cols-2" data-testid="program-create-step-frequency">
                {STUDY_FREQUENCY_OPTIONS.map((option) => (
                  <SelectionCard
                    key={option.value}
                    option={option}
                    selected={formData.study_days === option.value}
                    onSelect={(value) => setFormData({ ...formData, study_days: value })}
                    testId={`program-create-frequency-option-${option.value}`}
                  />
                ))}
              </div>
            )}

          {/* Navigation Buttons */}
            <div className="flex flex-col-reverse gap-3 pt-2 sm:flex-row sm:items-center sm:justify-between" data-testid="program-create-actions">
            <Button
              variant="outline"
              onClick={() => step === 1 ? navigate("/") : setStep(step - 1)}
              disabled={loading}
                className="h-11 rounded-xl border-border/70 bg-background/90 text-slate-700 hover:bg-secondary dark:text-slate-100"
                data-testid="program-create-back-button"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Geri
            </Button>

            <Button
              onClick={handleNext}
              disabled={loading}
                className="h-11 rounded-xl px-5 shadow-sm hover:shadow-md"
                data-testid="program-create-next-button"
            >
                {loading ? "Hazırlanıyor..." : step === TOTAL_STEPS ? (
                <>
                  <Check className="mr-2 h-4 w-4" />
                  Tamamla
                </>
              ) : (
                <>
                  İleri
                  <ArrowRight className="ml-2 h-4 w-4" />
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
      </div>
    </div>
  );
}
