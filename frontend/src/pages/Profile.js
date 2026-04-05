import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import ThemeToggle from "@/components/ThemeToggle";
import { useAuth } from "@/contexts/AuthContext";
import { saveProfile } from "@/lib/storage";
import { API } from "@/App";
import { ArrowLeft, Home, Save } from "lucide-react";

const getDefaultFormData = (currentUser) => ({
  username: localStorage.getItem("userName") || currentUser?.displayName || "",
  handle: "",
  streak_count: 0,
  email: currentUser?.email || "",
  grade_level: "",
  study_field: "",
  avatar_url: ""
});

const mapProfileToFormData = (profile, currentUser) => ({
  username: profile?.username || profile?.name || localStorage.getItem("userName") || currentUser?.displayName || "",
  handle: profile?.handle || "",
  streak_count: profile?.streak_count || 0,
  email: currentUser?.email || profile?.email || "",
  grade_level: profile?.grade_level || "",
  study_field: profile?.study_field || "",
  avatar_url: profile?.avatar_url || ""
});

const HANDLE_PATTERN = /^[a-z0-9_]{3,20}$/;
const GRADE_LEVEL_OPTIONS = [
  { value: "11", label: "11. Sınıf" },
  { value: "12", label: "12. Sınıf" },
  { value: "mezun", label: "Mezun" }
];
const GRADE_LEVEL_LABELS = Object.fromEntries(GRADE_LEVEL_OPTIONS.map((option) => [option.value, option.label]));
const STUDY_FIELD_OPTIONS = [
  { value: "Sayısal", label: "Sayısal" },
  { value: "EA", label: "Eşit Ağırlık" },
  { value: "Sözel", label: "Sözel" },
  { value: "Dil", label: "Dil" }
];
const STUDY_FIELD_LABELS = {
  Sayısal: "Sayısal",
  EA: "Eşit Ağırlık",
  "Eşit Ağırlık": "Eşit Ağırlık",
  Sözel: "Sözel",
  Dil: "Dil"
};

const getProfileMetaLine = (gradeLevel, studyField) => {
  const parts = [GRADE_LEVEL_LABELS[gradeLevel] || gradeLevel, STUDY_FIELD_LABELS[studyField] || studyField].filter(Boolean);
  return parts.length ? parts.join(" • ") : null;
};

export default function Profile() {
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  const [formData, setFormData] = useState(getDefaultFormData(currentUser));
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [profileExists, setProfileExists] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const profileMetaLine = getProfileMetaLine(formData.grade_level, formData.study_field);

  useEffect(() => {
    const loadProfile = async () => {
      if (!currentUser?.uid) {
        setLoading(false);
        return;
      }

      setLoading(true);
      setError("");

      try {
        const response = await axios.get(`${API}/profile`, {
          params: { firebase_uid: currentUser.uid }
        });

        const profile = response.data;

        if (profile?.id) {
          setProfileExists(true);
          setFormData(mapProfileToFormData(profile, currentUser));
          saveProfile(profile.id, profile.username || profile.name || "");
        } else {
          setProfileExists(false);
          setFormData(getDefaultFormData(currentUser));
        }
      } catch (loadError) {
        console.error("Error loading profile:", loadError);
        setError("Profil bilgileri yüklenirken bir hata oluştu.");
        setFormData(getDefaultFormData(currentUser));
      } finally {
        setLoading(false);
      }
    };

    loadProfile();
  }, [currentUser]);

  const handleChange = (field, value) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setSuccess("");

    if (!currentUser?.uid) {
      setError("Giriş yapan kullanıcı bulunamadı.");
      return;
    }

    if (!formData.username.trim()) {
      setError("Kullanıcı adı zorunludur.");
      return;
    }

    const normalizedHandle = formData.handle.trim().replace(/^@+/, "");
    if (normalizedHandle && !HANDLE_PATTERN.test(normalizedHandle)) {
      setError("Handle 3-20 karakter olmalı ve yalnızca küçük harf, rakam, underscore içermelidir.");
      return;
    }

    const resolvedEmail = currentUser.email || formData.email;
    if (!resolvedEmail) {
      setError("Bu hesap için e-posta bilgisi bulunamadı.");
      return;
    }

    const payload = {
      firebase_uid: currentUser.uid,
      username: formData.username.trim(),
      handle: normalizedHandle,
      email: resolvedEmail,
      avatar_url: formData.avatar_url.trim() || null,
      grade_level: formData.grade_level || null,
      study_field: formData.study_field || null
    };

    try {
      setSaving(true);

      const response = profileExists
        ? await axios.put(`${API}/profile`, payload)
        : await axios.post(`${API}/profile`, payload);

      const savedProfile = response.data;
      const resolvedName = savedProfile.username || savedProfile.name || payload.username;

      setProfileExists(true);
      setFormData(mapProfileToFormData(savedProfile, currentUser));
      saveProfile(savedProfile.id, resolvedName);
      setSuccess(profileExists ? "Profil güncellendi." : "Profil oluşturuldu.");
    } catch (saveError) {
      console.error("Error saving profile:", saveError);
      setError(saveError.response?.data?.detail || "Profil kaydedilirken bir hata oluştu.");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center text-gray-900 dark:text-gray-200" data-testid="profile-loading-state">
        Profil yükleniyor...
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background p-6 md:p-10" data-testid="profile-page">
      <div className="mx-auto max-w-4xl space-y-6" data-testid="profile-page-container">
        <Card className="rounded-2xl border border-border/70 bg-card shadow-[0_18px_36px_-28px_rgba(15,23,42,0.16)]" data-testid="profile-header-card">
          <CardContent className="p-6 sm:p-8">
            <div className="flex flex-wrap items-start justify-between gap-4" data-testid="profile-header-content">
              <div>
                <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-slate-900 dark:text-slate-100" data-testid="profile-title">
                  Profil
                </h1>
                <p className="mt-2 text-base md:text-lg text-slate-600 dark:text-slate-300" data-testid="profile-subtitle">
                  Hesap bilgilerini görüntüle ve profil detaylarını güncelle.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-2" data-testid="profile-header-actions">
                <ThemeToggle className="h-10 w-10 rounded-xl border border-border/70 bg-background text-slate-700 hover:bg-secondary dark:text-slate-100" />
                <Button
                  variant="outline"
                  onClick={() => navigate("/dashboard")}
                  className="h-10 rounded-xl border-border/70 bg-background text-slate-700 hover:bg-secondary dark:text-slate-100"
                  data-testid="profile-dashboard-button"
                >
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Dashboard
                </Button>
                <Button
                  variant="outline"
                  onClick={() => navigate("/")}
                  className="h-10 rounded-xl border-border/70 bg-background text-slate-700 hover:bg-secondary dark:text-slate-100"
                  data-testid="profile-home-button"
                >
                  <Home className="mr-2 h-4 w-4" />
                  Ana Sayfa
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {(error || success) && (
          <div
            className={`rounded-2xl border px-4 py-3 text-sm font-medium ${
              error
                ? "border-red-200 bg-red-50 text-red-700 dark:border-red-900/60 dark:bg-red-950/30 dark:text-red-300"
                : "border-green-200 bg-green-50 text-green-700 dark:border-green-900/60 dark:bg-green-950/30 dark:text-green-300"
            }`}
            data-testid={error ? "profile-error-message" : "profile-success-message"}
          >
            {error || success}
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_300px]" data-testid="profile-content-grid">
          <Card className="rounded-2xl border border-border/70 bg-card shadow-[0_18px_36px_-28px_rgba(15,23,42,0.16)]" data-testid="profile-form-card">
            <CardHeader className="pb-4 border-b border-border/70">
              <CardTitle className="text-xl font-bold text-slate-900 dark:text-slate-100" data-testid="profile-form-title">
                {profileExists ? "Profil Bilgileri" : "Profil Oluştur"}
              </CardTitle>
            </CardHeader>
            <CardContent className="p-5 sm:p-6">
              <form className="space-y-5" onSubmit={handleSubmit} data-testid="profile-form">
                <div className="space-y-2">
                  <Label htmlFor="profile-username" className="font-semibold text-slate-700 dark:text-slate-200" data-testid="profile-username-label">
                    Kullanıcı Adı
                  </Label>
                  <Input
                    id="profile-username"
                    value={formData.username}
                    onChange={(event) => handleChange("username", event.target.value)}
                    placeholder="Örn: Ayşe"
                    className="h-11 rounded-xl border-border/70 bg-background"
                    data-testid="profile-username-input"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="profile-handle" className="font-semibold text-slate-700 dark:text-slate-200" data-testid="profile-handle-label">
                    Handle
                  </Label>
                  <div className="relative" data-testid="profile-handle-input-wrap">
                    <span className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-slate-500 dark:text-slate-400" data-testid="profile-handle-prefix">
                      @
                    </span>
                    <Input
                      id="profile-handle"
                      value={formData.handle}
                      onChange={(event) => handleChange("handle", event.target.value.replace(/^@+/, "").toLowerCase())}
                      placeholder="ornek_handle"
                      className="h-11 rounded-xl border-border/70 bg-background pl-8"
                      data-testid="profile-handle-input"
                    />
                  </div>
                  <p className="text-sm text-slate-500 dark:text-slate-400" data-testid="profile-handle-helper-text">
                    3-20 karakter, sadece küçük harf, rakam ve underscore.
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="profile-email" className="font-semibold text-slate-700 dark:text-slate-200" data-testid="profile-email-label">
                    E-posta
                  </Label>
                  <Input
                    id="profile-email"
                    value={currentUser?.email || formData.email}
                    readOnly
                    className="h-11 rounded-xl border-border/70 bg-secondary text-slate-600 dark:text-slate-300"
                    data-testid="profile-email-input"
                  />
                  <p className="text-sm text-slate-500 dark:text-slate-400" data-testid="profile-email-helper-text">
                    Bu alan giriş yaptığın hesapla eşleşir ve düzenlenemez.
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="profile-grade-level" className="font-semibold text-slate-700 dark:text-slate-200" data-testid="profile-grade-level-label">
                    Sınıf Durumu
                  </Label>
                  <select
                    id="profile-grade-level"
                    value={formData.grade_level}
                    onChange={(event) => handleChange("grade_level", event.target.value)}
                    className="h-11 w-full rounded-xl border border-border/70 bg-background px-3 text-sm text-slate-900 outline-none ring-offset-background transition-[border-color,box-shadow] focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 dark:text-slate-100"
                    data-testid="profile-grade-level-select"
                  >
                    <option value="">Seçiniz</option>
                    {GRADE_LEVEL_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="profile-study-field" className="font-semibold text-slate-700 dark:text-slate-200" data-testid="profile-study-field-label">
                    Alan
                  </Label>
                  <select
                    id="profile-study-field"
                    value={formData.study_field}
                    onChange={(event) => handleChange("study_field", event.target.value)}
                    className="h-11 w-full rounded-xl border border-border/70 bg-background px-3 text-sm text-slate-900 outline-none ring-offset-background transition-[border-color,box-shadow] focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 dark:text-slate-100"
                    data-testid="profile-study-field-select"
                  >
                    <option value="">Seçiniz</option>
                    {STUDY_FIELD_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="profile-avatar-url" className="font-semibold text-slate-700 dark:text-slate-200" data-testid="profile-avatar-url-label">
                    Avatar URL
                  </Label>
                  <Input
                    id="profile-avatar-url"
                    value={formData.avatar_url}
                    onChange={(event) => handleChange("avatar_url", event.target.value)}
                    placeholder="https://..."
                    className="h-11 rounded-xl border-border/70 bg-background"
                    data-testid="profile-avatar-url-input"
                  />
                </div>

                <Button
                  type="submit"
                  disabled={saving}
                  className="h-11 w-full rounded-xl bg-primary text-primary-foreground hover:bg-slate-800"
                  data-testid="profile-form-submit-button"
                >
                  <Save className="mr-2 h-4 w-4" />
                  {saving ? "Kaydediliyor..." : profileExists ? "Profili Güncelle" : "Profili Oluştur"}
                </Button>
              </form>
            </CardContent>
          </Card>

          <Card className="rounded-2xl border border-border/70 bg-card shadow-[0_18px_36px_-28px_rgba(15,23,42,0.16)]" data-testid="profile-preview-card">
            <CardHeader className="pb-4 border-b border-border/70">
              <CardTitle className="text-xl font-bold text-slate-900 dark:text-slate-100" data-testid="profile-preview-title">
                Önizleme
              </CardTitle>
            </CardHeader>
            <CardContent className="p-6 space-y-5">
              <div className="flex justify-center" data-testid="profile-avatar-preview-wrap">
                {formData.avatar_url.trim() ? (
                  <img
                    src={formData.avatar_url}
                    alt="Avatar önizleme"
                    className="h-24 w-24 rounded-2xl border border-border/70 object-cover shadow-sm"
                    data-testid="profile-avatar-preview-image"
                  />
                ) : (
                  <div className="flex h-24 w-24 items-center justify-center rounded-2xl border border-dashed border-border/70 bg-background text-sm text-slate-500 dark:text-slate-400" data-testid="profile-avatar-preview-placeholder">
                    Avatar yok
                  </div>
                )}
              </div>

              <div className="space-y-3" data-testid="profile-preview-details">
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-slate-400" data-testid="profile-preview-handle-label">
                    Handle
                  </p>
                  <p className="mt-1 text-lg font-semibold text-slate-900 dark:text-slate-100" data-testid="profile-preview-handle-value">
                    {formData.handle ? `@${formData.handle}` : "Henüz eklenmedi"}
                  </p>
                </div>

                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-slate-400" data-testid="profile-preview-username-label">
                    Kullanıcı Adı
                  </p>
                  <p className="mt-1 text-lg font-semibold text-slate-900 dark:text-slate-100" data-testid="profile-preview-username-value">
                    {formData.username || "Henüz eklenmedi"}
                  </p>
                  {formData.streak_count > 0 && (
                    <p className="mt-2 text-sm font-semibold text-orange-600 dark:text-orange-300" data-testid="profile-streak-value">
                      🔥 {formData.streak_count} Günlük Seri
                    </p>
                  )}
                </div>

                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-slate-400" data-testid="profile-preview-email-label">
                    E-posta
                  </p>
                  <p className="mt-1 text-sm text-slate-700 dark:text-slate-300 break-all" data-testid="profile-preview-email-value">
                    {currentUser?.email || formData.email || "Bulunamadı"}
                  </p>
                </div>

                {profileMetaLine && (
                  <div data-testid="profile-preview-meta-wrap">
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-slate-400" data-testid="profile-preview-meta-label">
                      Profil Özeti
                    </p>
                    <p className="mt-1 text-sm text-slate-700 dark:text-slate-300" data-testid="profile-preview-meta-value">
                      {profileMetaLine}
                    </p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}