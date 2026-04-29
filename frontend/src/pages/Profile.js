import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
      <div className="flex min-h-screen items-center justify-center text-gray-900 " data-testid="profile-loading-state">
        Profil yükleniyor...
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background px-4 py-6 sm:px-6 lg:px-10 lg:py-8" data-testid="profile-page">
      <div className="mx-auto w-full max-w-6xl space-y-5" data-testid="profile-page-container">
        <Card className="rounded-2xl border border-transparent bg-transparent shadow-none" data-testid="profile-header-card">
          <CardContent className="px-0 py-2 sm:py-3">
            <div className="flex flex-wrap items-start justify-between gap-3" data-testid="profile-header-content">
              <div>
                <h1 className="text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl" data-testid="profile-title">
                  Profil
                </h1>
                <p className="mt-1 text-sm text-gray-500" data-testid="profile-subtitle">
                  Hesap bilgilerini görüntüle ve profil detaylarını güncelle.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-2" data-testid="profile-header-actions">
                <Button
                  variant="outline"
                  onClick={() => navigate("/dashboard")}
                  className="h-10 rounded-lg border-gray-200 bg-white text-slate-700 hover:bg-indigo-50 hover:text-indigo-700"
                  data-testid="profile-dashboard-button"
                >
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Dashboard
                </Button>
                <Button
                  variant="outline"
                  onClick={() => navigate("/")}
                  className="h-10 rounded-lg border-gray-200 bg-white text-slate-700 hover:bg-indigo-50 hover:text-indigo-700"
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
                ? "border-red-200 bg-red-50 text-red-700   "
                : "border-green-200 bg-green-50 text-green-700   "
            }`}
            data-testid={error ? "profile-error-message" : "profile-success-message"}
          >
            {error || success}
          </div>
        )}

        <div className="grid gap-5 lg:grid-cols-[minmax(0,1.85fr)_minmax(0,1fr)] lg:items-start" data-testid="profile-content-grid">
          <Card className="rounded-2xl border border-gray-200 bg-white shadow-sm" data-testid="profile-form-card">
            <CardHeader className="border-b border-gray-100 pb-3">
              <CardTitle className="text-lg font-semibold text-slate-900" data-testid="profile-form-title">
                {profileExists ? "Profil Bilgileri" : "Profil Oluştur"}
              </CardTitle>
            </CardHeader>
            <CardContent className="p-5 sm:p-6">
              <form className="space-y-4" onSubmit={handleSubmit} data-testid="profile-form">
                <div className="space-y-1.5">
                  <Label htmlFor="profile-username" className="text-sm font-medium text-gray-700" data-testid="profile-username-label">
                    Kullanıcı Adı
                  </Label>
                  <Input
                    id="profile-username"
                    value={formData.username}
                    onChange={(event) => handleChange("username", event.target.value)}
                    placeholder="Örn: Ayşe"
                    className="h-11 rounded-lg border-gray-200 bg-white focus-visible:border-indigo-500 focus-visible:ring-indigo-500"
                    data-testid="profile-username-input"
                  />
                </div>

                <div className="space-y-1.5">
                  <Label htmlFor="profile-handle" className="text-sm font-medium text-gray-700" data-testid="profile-handle-label">
                    Handle
                  </Label>
                  <div className="relative" data-testid="profile-handle-input-wrap">
                    <span className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-slate-400" data-testid="profile-handle-prefix">
                      @
                    </span>
                    <Input
                      id="profile-handle"
                      value={formData.handle}
                      onChange={(event) => handleChange("handle", event.target.value.replace(/^@+/, "").toLowerCase())}
                      placeholder="ornek_handle"
                      className="h-11 rounded-lg border-gray-200 bg-white pl-8 focus-visible:border-indigo-500 focus-visible:ring-indigo-500"
                      data-testid="profile-handle-input"
                    />
                  </div>
                  <p className="text-xs text-slate-500" data-testid="profile-handle-helper-text">
                    3-20 karakter, sadece küçük harf, rakam ve underscore.
                  </p>
                </div>

                <div className="space-y-1.5">
                  <Label htmlFor="profile-email" className="text-sm font-medium text-gray-700" data-testid="profile-email-label">
                    E-posta
                  </Label>
                  <Input
                    id="profile-email"
                    value={currentUser?.email || formData.email}
                    readOnly
                    className="h-11 rounded-lg border-gray-200 bg-gray-100 text-slate-500 focus-visible:border-indigo-500 focus-visible:ring-indigo-500"
                    data-testid="profile-email-input"
                  />
                  <p className="text-xs text-slate-500" data-testid="profile-email-helper-text">
                    Bu alan giriş yaptığın hesapla eşleşir ve düzenlenemez.
                  </p>
                </div>

                <div className="space-y-1.5">
                  <Label htmlFor="profile-grade-level" className="text-sm font-medium text-gray-700" data-testid="profile-grade-level-label">
                    Sınıf Durumu
                  </Label>
                  <select
                    id="profile-grade-level"
                    value={formData.grade_level}
                    onChange={(event) => handleChange("grade_level", event.target.value)}
                    className="h-11 w-full rounded-lg border border-gray-200 bg-white px-3 text-sm text-slate-900 outline-none ring-offset-background transition-[border-color,box-shadow] focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-0"
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

                <div className="space-y-1.5">
                  <Label htmlFor="profile-study-field" className="text-sm font-medium text-gray-700" data-testid="profile-study-field-label">
                    Alan
                  </Label>
                  <select
                    id="profile-study-field"
                    value={formData.study_field}
                    onChange={(event) => handleChange("study_field", event.target.value)}
                    className="h-11 w-full rounded-lg border border-gray-200 bg-white px-3 text-sm text-slate-900 outline-none ring-offset-background transition-[border-color,box-shadow] focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-0"
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

                <div className="space-y-1.5">
                  <Label htmlFor="profile-avatar-url" className="text-sm font-medium text-gray-700" data-testid="profile-avatar-url-label">
                    Avatar URL
                  </Label>
                  <Input
                    id="profile-avatar-url"
                    value={formData.avatar_url}
                    onChange={(event) => handleChange("avatar_url", event.target.value)}
                    placeholder="https://..."
                    className="h-11 rounded-lg border-gray-200 bg-white focus-visible:border-indigo-500 focus-visible:ring-indigo-500"
                    data-testid="profile-avatar-url-input"
                  />
                </div>

                <Button
                  type="submit"
                  disabled={saving}
                  className="h-12 w-full rounded-lg bg-indigo-600 text-white shadow-sm hover:bg-indigo-700 disabled:opacity-70"
                  data-testid="profile-form-submit-button"
                >
                  <Save className="mr-2 h-4 w-4" />
                  {saving ? "Kaydediliyor..." : profileExists ? "Profili Güncelle" : "Profili Oluştur"}
                </Button>
              </form>
            </CardContent>
          </Card>

          <Card className="rounded-2xl border border-gray-200 bg-white shadow-sm" data-testid="profile-preview-card">
            <CardHeader className="border-b border-gray-100 pb-3">
              <CardTitle className="text-lg font-semibold text-slate-900" data-testid="profile-preview-title">
                Önizleme
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-5 p-5 sm:p-6">
              <div className="rounded-xl border border-gray-100 bg-gradient-to-b from-indigo-50/60 to-white p-4">
                <div className="flex flex-col items-center text-center" data-testid="profile-avatar-preview-wrap">
                  {formData.avatar_url.trim() ? (
                    <img
                      src={formData.avatar_url}
                      alt="Avatar önizleme"
                      className="h-20 w-20 rounded-2xl border border-gray-200 object-cover shadow-sm"
                      data-testid="profile-avatar-preview-image"
                    />
                  ) : (
                    <div className="flex h-20 w-20 items-center justify-center rounded-2xl border border-dashed border-gray-300 bg-white text-xs text-slate-500" data-testid="profile-avatar-preview-placeholder">
                      Avatar yok
                    </div>
                  )}

                  <p className="mt-3 text-lg font-semibold text-slate-900" data-testid="profile-preview-handle-value">
                    {formData.handle ? `@${formData.handle}` : "@handle"}
                  </p>
                  <p className="text-sm text-slate-500" data-testid="profile-preview-username-value">
                    {formData.username || "Kullanıcı adı"}
                  </p>

                  {formData.streak_count > 0 && (
                    <span className="mt-3 inline-flex items-center rounded-full bg-orange-50 px-2 py-1 text-xs font-semibold text-orange-600" data-testid="profile-streak-value">
                      🔥 {formData.streak_count} Günlük Seri
                    </span>
                  )}
                </div>
              </div>

              <div className="space-y-3" data-testid="profile-preview-details">
                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-500" data-testid="profile-preview-handle-label">
                    Handle
                  </p>
                  <p className="mt-1 text-sm font-semibold text-slate-900">
                    {formData.handle ? `@${formData.handle}` : "Belirtilmedi"}
                  </p>
                </div>

                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-500" data-testid="profile-preview-username-label">
                    Kullanıcı Adı
                  </p>
                  <p className="mt-1 text-sm text-slate-600">
                    {formData.username || "Belirtilmedi"}
                  </p>
                </div>

                <div>
                  <p className="text-xs uppercase tracking-[0.18em] text-slate-500" data-testid="profile-preview-email-label">
                    E-posta
                  </p>
                  <p className="mt-1 text-sm text-slate-700 break-all" data-testid="profile-preview-email-value">
                    {currentUser?.email || formData.email || "Bulunamadı"}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-500" data-testid="profile-preview-grade-label">
                      Sınıf
                    </p>
                    <p className="mt-1 text-sm text-slate-700" data-testid="profile-preview-grade-value">
                      {GRADE_LEVEL_LABELS[formData.grade_level] || "Belirtilmedi"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-500" data-testid="profile-preview-field-label">
                      Alan
                    </p>
                    <p className="mt-1 text-sm text-slate-700" data-testid="profile-preview-field-value">
                      {STUDY_FIELD_LABELS[formData.study_field] || formData.study_field || "Belirtilmedi"}
                    </p>
                  </div>
                </div>

                {profileMetaLine && (
                  <div data-testid="profile-preview-meta-wrap">
                    <p className="text-xs uppercase tracking-[0.18em] text-slate-500" data-testid="profile-preview-meta-label">
                      Profil Özeti
                    </p>
                    <p className="mt-1 text-sm text-slate-600" data-testid="profile-preview-meta-value">
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