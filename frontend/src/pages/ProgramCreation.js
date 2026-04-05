import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import axios from "axios";
import { API } from "@/App";
import { ArrowLeft, ArrowRight, Check } from "lucide-react";
import { saveProfile, savePrograms, getProgramDraft, saveProgramDraft, clearProgramDraft, setOnboardingCompleted } from "@/lib/storage";
import { useAuth } from "@/contexts/AuthContext";

export default function ProgramCreation() {
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  const [step, setStep] = useState(1);
  const [formData, setFormData] = useState({
    name: "",
    study_field: "",
    exam_goal: "",
    daily_hours: "",
    study_days: 5
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const autosaveTimerRef = useRef(null);

  // Load draft on mount
  useEffect(() => {
    const draft = getProgramDraft();
    if (draft && draft.formData) {
      setFormData(draft.formData);
      setStep(draft.step || 1);
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
      setError("Lütfen adını gir");
      return;
    }
    if (step === 2 && !formData.exam_goal) {
      setError("Lütfen sınav hedefini seç");
      return;
    }
    if (step === 3 && !formData.daily_hours) {
      setError("Lütfen günlük çalışma süresini seç");
      return;
    }
    
    if (step < 3) {
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
        name: formData.name,
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

      // Store profile ID and name in localStorage FIRST
      localStorage.setItem("profileId", profileId);
      localStorage.setItem("userName", formData.name);
      
      // Save to userData storage
      saveProfile(profileId, formData.name);
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
      
      // More specific error messages
      if (error.response) {
        // Backend returned an error
        setError(`Sunucu hatası: ${error.response.status}`);
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
    <div className="min-h-screen flex items-center justify-center p-4">
      <Card className="w-full max-w-2xl shadow-lg border border-gray-200" data-testid="program-creation-card">
        <CardHeader>
          <CardTitle className="text-3xl text-center font-extrabold text-gray-900">
            {step === 1 && "Kendini Tanıt"}
            {step === 2 && "Sınav Hedefin"}
            {step === 3 && "Çalışma Programın"}
          </CardTitle>
          <div className="flex justify-center gap-3 mt-6">
            <div className={`h-2.5 w-24 rounded-full transition-all duration-300 ${step >= 1 ? 'bg-indigo-600 shadow-sm' : 'bg-gray-300'}`}></div>
            <div className={`h-2.5 w-24 rounded-full transition-all duration-300 ${step >= 2 ? 'bg-indigo-600 shadow-sm' : 'bg-gray-300'}`}></div>
            <div className={`h-2.5 w-24 rounded-full transition-all duration-300 ${step >= 3 ? 'bg-indigo-600 shadow-sm' : 'bg-gray-300'}`}></div>
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Error Message */}
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded" data-testid="error-message">
              {error}
            </div>
          )}
          
          {/* Step 1: Profile */}
          {step === 1 && (
            <div className="space-y-5" data-testid="step-profile">
              <div>
                <Label htmlFor="name" className="font-semibold">Adın *</Label>
                <Input
                  id="name"
                  placeholder="Örn: Ahmet"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="mt-2 border-2"
                  data-testid="input-name"
                />
              </div>

              <div>
                <Label className="font-semibold">Alan (Opsiyonel)</Label>
                <RadioGroup value={formData.study_field} onValueChange={(val) => setFormData({ ...formData, study_field: val })} className="mt-3 space-y-3">
                  <div className="flex items-center space-x-3">
                    <RadioGroupItem value="Sayısal" id="sayisal" data-testid="radio-sayisal" />
                    <Label htmlFor="sayisal" className="cursor-pointer font-medium">Sayısal</Label>
                  </div>
                  <div className="flex items-center space-x-3">
                    <RadioGroupItem value="EA" id="ea" data-testid="radio-ea" />
                    <Label htmlFor="ea" className="cursor-pointer font-medium">Eşit Ağırlık</Label>
                  </div>
                  <div className="flex items-center space-x-3">
                    <RadioGroupItem value="Sözel" id="sozel" data-testid="radio-sozel" />
                    <Label htmlFor="sozel" className="cursor-pointer font-medium">Sözel</Label>
                  </div>
                </RadioGroup>
              </div>
            </div>
          )}

          {/* Step 2: Exam Goal */}
          {step === 2 && (
            <div className="space-y-5" data-testid="step-exam-goal">
              <Label className="font-semibold">Sınav Hedefin *</Label>
              <RadioGroup value={formData.exam_goal} onValueChange={(val) => setFormData({ ...formData, exam_goal: val })} className="mt-3 space-y-3">
                <div className="flex items-center space-x-3">
                  <RadioGroupItem value="TYT" id="tyt" data-testid="radio-tyt" />
                  <Label htmlFor="tyt" className="cursor-pointer font-medium">TYT</Label>
                </div>
                <div className="flex items-center space-x-3">
                  <RadioGroupItem value="AYT" id="ayt" data-testid="radio-ayt" />
                  <Label htmlFor="ayt" className="cursor-pointer font-medium">AYT</Label>
                </div>
                <div className="flex items-center space-x-3">
                  <RadioGroupItem value="TYT + AYT" id="both" data-testid="radio-both" />
                  <Label htmlFor="both" className="cursor-pointer font-medium">TYT + AYT</Label>
                </div>
              </RadioGroup>
            </div>
          )}

          {/* Step 3: Study Schedule */}
          {step === 3 && (
            <div className="space-y-6" data-testid="step-study-schedule">
              <div>
                <Label className="font-semibold">Günlük Çalışma Süresi *</Label>
                <RadioGroup value={formData.daily_hours} onValueChange={(val) => setFormData({ ...formData, daily_hours: val })} className="mt-3 space-y-3">
                  <div className="flex items-center space-x-3">
                    <RadioGroupItem value="1-2" id="hours-1-2" data-testid="radio-hours-1-2" />
                    <Label htmlFor="hours-1-2" className="cursor-pointer font-medium">1-2 saat</Label>
                  </div>
                  <div className="flex items-center space-x-3">
                    <RadioGroupItem value="2-4" id="hours-2-4" data-testid="radio-hours-2-4" />
                    <Label htmlFor="hours-2-4" className="cursor-pointer font-medium">2-4 saat</Label>
                  </div>
                  <div className="flex items-center space-x-3">
                    <RadioGroupItem value="4+" id="hours-4plus" data-testid="radio-hours-4plus" />
                    <Label htmlFor="hours-4plus" className="cursor-pointer font-medium">4+ saat</Label>
                  </div>
                </RadioGroup>
              </div>

              <div>
                <Label htmlFor="study-days" className="font-semibold">Haftada Kaç Gün Çalışacaksın?</Label>
                <Input
                  id="study-days"
                  type="number"
                  min="1"
                  max="7"
                  value={formData.study_days}
                  onChange={(e) => setFormData({ ...formData, study_days: parseInt(e.target.value) || 1 })}
                  className="mt-2 border-2"
                  data-testid="input-study-days"
                />
                <p className="text-sm text-gray-500 mt-2">1-7 gün arası</p>
              </div>
            </div>
          )}

          {/* Navigation Buttons */}
          <div className="flex justify-between pt-6">
            <Button
              variant="outline"
              onClick={() => step === 1 ? navigate("/") : setStep(step - 1)}
              disabled={loading}
              className="shadow-sm hover:shadow-md transition-shadow"
              data-testid="btn-back"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Geri
            </Button>

            <Button
              onClick={handleNext}
              disabled={loading}
              className="shadow-md hover:shadow-lg transition-shadow font-semibold"
              data-testid="btn-next"
            >
              {loading ? "Oluşturuluyor..." : step === 3 ? (
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
  );
}
