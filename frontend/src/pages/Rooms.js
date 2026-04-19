import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import AppLogo from "@/components/AppLogo";
import axios from "axios";
import { API } from "@/App";
import { Home, Plus, LogIn, Trophy } from "lucide-react";
import { saveRoom } from "@/lib/storage";
import { useAuth } from "@/contexts/AuthContext";

const ROOM_PASSWORD_MIN_LENGTH = 6;

export default function Rooms() {
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  const [identity, setIdentity] = useState({
    name: "",
    handle: "",
    avatar_url: ""
  });
  const [identityLoading, setIdentityLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("create");

  const [createForm, setCreateForm] = useState({
    room_name: "",
    room_type: "public",
    room_password: ""
  });

  const [joinForm, setJoinForm] = useState({
    room_code: "",
    room_password: ""
  });

  const [loading, setLoading] = useState(false);
  const [joinRequiresPassword, setJoinRequiresPassword] = useState(false);
  const [joinResolvedRoomName, setJoinResolvedRoomName] = useState("");
  const [joinError, setJoinError] = useState("");

  useEffect(() => {
    const loadIdentity = async () => {
      const fallbackName =
        currentUser?.displayName?.trim() ||
        currentUser?.email?.trim() ||
        "İzlek Kullanıcısı";

      const applyIdentity = (nextIdentity) => {
        setIdentity(nextIdentity);
        localStorage.setItem("userName", nextIdentity.name);

        if (nextIdentity.avatar_url) {
          localStorage.setItem("userAvatarUrl", nextIdentity.avatar_url);
        } else {
          localStorage.removeItem("userAvatarUrl");
        }

      };

      if (!currentUser?.uid) {
        applyIdentity({
          name: fallbackName,
          handle: "",
          avatar_url: ""
        });
        setIdentityLoading(false);
        return;
      }

      try {
        const response = await axios.get(`${API}/profile`, {
          params: { firebase_uid: currentUser.uid }
        });

        const profile = response.data;
        applyIdentity({
          name: profile?.username || profile?.name || fallbackName,
          handle: profile?.handle || "",
          avatar_url: profile?.avatar_url || ""
        });
      } catch (error) {
        console.error("Error loading room identity:", error);
        applyIdentity({
          name: fallbackName,
          handle: "",
          avatar_url: ""
        });
      } finally {
        setIdentityLoading(false);
      }
    };

    loadIdentity();
  }, [currentUser]);

  const ensureHandleReady = () => {
    if (identityLoading) {
      alert("Profil bilgileri yükleniyor. Lütfen tekrar dene.");
      return false;
    }

    if (!identity.handle?.trim()) {
      navigate("/profile");
      return false;
    }

    return true;
  };

  const handleCreateRoom = async () => {
    if (!createForm.room_name.trim()) {
      alert("Oda adı boş olamaz");
      return;
    }
    if (!identity.name.trim()) {
      alert("Profil bilgileri yükleniyor. Lütfen tekrar dene.");
      return;
    }
    if (!ensureHandleReady()) {
      return;
    }
    if (createForm.room_type === "private" && createForm.room_password.trim().length < ROOM_PASSWORD_MIN_LENGTH) {
      alert(`Özel odalar için şifre en az ${ROOM_PASSWORD_MIN_LENGTH} karakter olmalıdır.`);
      return;
    }

    const stableUserId = currentUser?.uid || localStorage.getItem("userId");

    setLoading(true);
    try {
      const res = await axios.post(`${API}/rooms`, {
        name: createForm.room_name,
        owner_name: identity.name,
        owner_id: stableUserId,
        owner_avatar_url: identity.avatar_url || null,
        room_type: createForm.room_type,
        room_password: createForm.room_type === "private" ? createForm.room_password : null
      });

      const roomId = res.data.id;
      const userId = stableUserId || res.data.owner_id;

      localStorage.setItem("currentRoomId", roomId);
      localStorage.setItem("currentUserId", userId);
      localStorage.setItem("userName", identity.name);

      if (identity.avatar_url) {
        localStorage.setItem("userAvatarUrl", identity.avatar_url);
      } else {
        localStorage.removeItem("userAvatarUrl");
      }

      saveRoom(roomId, userId);
      navigate(`/room/${roomId}`);
    } catch (error) {
      console.error("Error creating room:", error);
      console.error("Error response:", error.response?.data);
      console.error("Error status:", error.response?.status);
      alert(`Oda oluşturulurken hata: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleJoinRoom = async () => {
    if (!joinForm.room_code.trim()) {
      alert("Oda kodu boş olamaz");
      return;
    }
    if (!identity.name.trim()) {
      alert("Profil bilgileri yükleniyor. Lütfen tekrar dene.");
      return;
    }
    if (!ensureHandleReady()) {
      return;
    }

    const stableUserId = currentUser?.uid || localStorage.getItem("userId");
    const normalizedRoomCode = joinForm.room_code.trim().toUpperCase();

    if (joinRequiresPassword && !joinForm.room_password.trim()) {
      setJoinError("Bu oda özel. Devam etmek için oda şifresini gir.");
      return;
    }

    setLoading(true);
    try {
      if (!joinRequiresPassword) {
        const roomLookup = await axios.get(`${API}/rooms/code/${normalizedRoomCode}`);

        if (roomLookup.data?.is_private) {
          setJoinRequiresPassword(true);
          setJoinResolvedRoomName(roomLookup.data?.name || "");
          setJoinError("");
          return;
        }
      }

      const res = await axios.post(`${API}/rooms/join`, {
        room_code: normalizedRoomCode,
        user_id: stableUserId,
        user_name: identity.name,
        user_avatar_url: identity.avatar_url || null,
        room_password: joinRequiresPassword ? joinForm.room_password : null
      });

      if (res.data.error) {
        alert(res.data.error);
        return;
      }

      const roomId = res.data.id;
      const joinedUserId = stableUserId;

      localStorage.setItem("currentRoomId", roomId);
      localStorage.setItem("currentUserId", joinedUserId);
      localStorage.setItem("userName", identity.name);

      if (identity.avatar_url) {
        localStorage.setItem("userAvatarUrl", identity.avatar_url);
      } else {
        localStorage.removeItem("userAvatarUrl");
      }

      saveRoom(roomId, joinedUserId);
      navigate(`/room/${roomId}`);
    } catch (error) {
      console.error("Error joining room:", error);
      console.error("Error response:", error.response?.data);
      console.error("Error status:", error.response?.status);

      if (error.response?.status === 404) {
        setJoinRequiresPassword(false);
        setJoinResolvedRoomName("");
        setJoinForm((prev) => ({ ...prev, room_password: "" }));
        setJoinError("Oda bulunamadı.");
        return;
      }

      if (error.response?.status === 403) {
        setJoinError(error.response?.data?.detail || "Oda şifresi hatalı.");
        return;
      }

      alert(`Odaya katılırken hata: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const focusFormsSection = (nextTab) => {
    setActiveTab(nextTab);
    requestAnimationFrame(() => {
      document.getElementById("rooms-forms-section")?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  };

  const sharedInputClass =
    "mt-2 h-11 rounded-xl border border-border/70 bg-secondary/70 text-foreground placeholder:text-muted-foreground shadow-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-0 ";

  const sharedSelectClass =
    "mt-2 h-11 w-full rounded-xl border border-border/70 bg-secondary/70 px-3 text-sm text-foreground shadow-none focus:outline-none focus:ring-2 focus:ring-ring ";

  const surfaceCardClass =
    "rounded-2xl border border-border/60 bg-background/85 text-card-foreground shadow-none";

  const formMeta =
    activeTab === "create"
      ? {
          title: "Yeni oda hazırlığı",
          description: "Sadece gerekli alanları doldur; oda oluşturma akışı aynı şekilde çalışmaya devam eder.",
          submitText: loading ? "Oluşturuluyor..." : "Oda Oluştur",
          submitTestId: "btn-create-room"
        }
      : {
          title: "Var olan odaya katıl",
          description: joinRequiresPassword
            ? "Bu oda özel. Şifreni girerek aynı akış içinde devam et."
            : "Oda kodunu gir ve mevcut çalışma alanına aynı akışla katıl.",
          submitText: joinRequiresPassword
            ? (loading ? "Katılınıyor..." : "Şifre ile Katıl")
            : (loading ? "Kod Kontrol Ediliyor..." : "Odaya Katıl"),
          submitTestId: "btn-join-room"
        };

  return (
    <div className="min-h-screen bg-background text-foreground" data-testid="rooms-page">
      <div className="px-4 pb-10 pt-6 sm:px-6 sm:pb-12 lg:px-10 xl:px-12">
        <div className="space-y-10 sm:space-y-12">
          <header className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <button
              type="button"
              onClick={() => navigate("/dashboard")}
              className="inline-flex h-10 w-fit items-center"
              data-testid="rooms-header-brand"
              aria-label="Dashboard"
            >
              <AppLogo />
            </button>

            <div className="flex items-center justify-end gap-2" data-testid="rooms-theme-toggle-wrap">
              <Button variant="outline" size="sm" onClick={() => navigate("/")} data-testid="btn-home">
                <Home className="h-4 w-4" />
                Ana Sayfa
              </Button>
              <Button variant="outline" size="sm" onClick={() => navigate("/leaderboard")} data-testid="btn-leaderboard">
                <Trophy className="h-4 w-4" />
                Liderlik Tablosu
              </Button>
            </div>
          </header>

          <section className="relative overflow-hidden rounded-2xl border border-indigo-100 bg-indigo-50/25 px-7 py-9 shadow-sm sm:px-9 sm:py-10" data-testid="rooms-hero-section">

            <div className="relative max-w-[980px] space-y-7">
              <div className="space-y-3">
                <p className="text-sm font-medium uppercase tracking-[0.18em] text-muted-foreground">
                  Birlikte daha odaklı çalış
                </p>
                <h1 className="font-display text-4xl font-bold tracking-tight text-foreground sm:text-5xl lg:text-6xl" data-testid="rooms-title">
                  Online Çalışma Odaları
                </h1>
                <p className="max-w-2xl text-sm leading-6 text-muted-foreground sm:text-base" data-testid="rooms-subtitle">
                  Bu sayfa oda aksiyonlarına odaklanır: ihtiyacın olanı yukarıdan seç, aşağıda sadece ilgili formu temiz ve tekrar etmeyen bir akışla kullan.
                </p>
              </div>

              <div className="grid grid-cols-1 gap-4 sm:max-w-3xl sm:grid-cols-2" data-testid="rooms-quick-actions">
                <Button
                  size="lg"
                  onClick={() => focusFormsSection("create")}
                  className={`h-auto items-start justify-between whitespace-normal rounded-2xl border px-6 py-5 text-left shadow-sm transition-all duration-200 ${
                    activeTab === "create"
                      ? "border-indigo-600 bg-indigo-600 text-white hover:bg-indigo-700 hover:border-indigo-700"
                      : "border-indigo-200 bg-white text-indigo-700 hover:border-indigo-300 hover:bg-indigo-50"
                  }`}
                  data-testid="rooms-quick-create-button"
                >
                  <div>
                    <div className="flex items-center gap-2 text-base font-semibold">
                      <Plus className="h-4 w-4" />
                      Oda Oluştur
                    </div>
                    <p className={`mt-1 text-sm font-medium sm:text-sm ${
                      activeTab === "create" ? "text-indigo-100" : "text-indigo-500"
                    }`}>
                      Yeni oda açma formunu aşağıda göster.
                    </p>
                  </div>
                </Button>

                <Button
                  size="lg"
                  onClick={() => focusFormsSection("join")}
                  className={`h-auto items-start justify-between whitespace-normal rounded-2xl border px-6 py-5 text-left shadow-sm transition-all duration-200 ${
                    activeTab === "join"
                      ? "border-indigo-600 bg-indigo-600 text-white hover:bg-indigo-700 hover:border-indigo-700"
                      : "border-indigo-200 bg-white text-indigo-700 hover:border-indigo-300 hover:bg-indigo-50"
                  }`}
                  data-testid="rooms-quick-join-button"
                >
                  <div>
                    <div className="flex items-center gap-2 text-base font-semibold">
                      <LogIn className="h-4 w-4" />
                      Odaya Katıl
                    </div>
                    <p className={`mt-1 text-sm font-medium sm:text-sm ${
                      activeTab === "join" ? "text-indigo-100" : "text-indigo-500"
                    }`}>
                      Mevcut oda kodunla ilerle.
                    </p>
                  </div>
                </Button>
              </div>
            </div>
          </section>

          <Card id="rooms-forms-section" className={surfaceCardClass} data-testid="rooms-main-card">
            <CardHeader className="pb-0">
              <div className="space-y-1.5">
                <CardTitle className="font-display text-xl text-foreground/90 sm:text-2xl" data-testid="rooms-form-title">{formMeta.title}</CardTitle>
                <p className="text-sm leading-6 text-muted-foreground/90" data-testid="rooms-form-description">
                  {formMeta.description}
                </p>
              </div>
            </CardHeader>
            <CardContent className="pt-5 sm:pt-6">
              {activeTab === "create" ? (
                <div className="space-y-6" data-testid="create-room-form">
                  <div className="space-y-2">
                    <Label htmlFor="room-name" className="text-sm font-medium text-foreground" data-testid="label-room-name">Oda Adı *</Label>
                    <Input
                      id="room-name"
                      placeholder="Örn: TYT Matematik Çalışma"
                      value={createForm.room_name}
                      onChange={(e) => setCreateForm({ ...createForm, room_name: e.target.value })}
                      className={sharedInputClass}
                      data-testid="input-room-name"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="room-type" className="text-sm font-medium text-foreground" data-testid="label-room-type">Oda Türü</Label>
                    <select
                      id="room-type"
                      className={sharedSelectClass}
                      value={createForm.room_type}
                      onChange={(e) => setCreateForm((prev) => ({
                        ...prev,
                        room_type: e.target.value,
                        room_password: e.target.value === "private" ? prev.room_password : ""
                      }))}
                      data-testid="select-room-type"
                    >
                      <option value="public">Herkese Açık</option>
                      <option value="private">Özel</option>
                    </select>
                  </div>

                  {createForm.room_type === "private" && (
                    <div className="space-y-2" data-testid="create-room-password-wrap">
                      <Label htmlFor="room-password" className="text-sm font-medium text-foreground" data-testid="label-room-password">Oda Şifresi *</Label>
                      <Input
                        id="room-password"
                        type="password"
                        placeholder="En az 6 karakter"
                        value={createForm.room_password}
                        onChange={(e) => setCreateForm({ ...createForm, room_password: e.target.value })}
                        className={sharedInputClass}
                        data-testid="input-room-password"
                      />
                      <p className="mt-2 text-sm text-muted-foreground" data-testid="room-password-help-text">
                        Özel odalara yalnızca kod ve doğru şifre ile girilir.
                      </p>
                    </div>
                  )}

                  <Button
                    className="h-11 w-full rounded-xl bg-indigo-600 text-white shadow-sm transition-all duration-200 hover:bg-indigo-700 hover:shadow-md"
                    onClick={handleCreateRoom}
                    disabled={loading || identityLoading}
                    data-testid={formMeta.submitTestId}
                  >
                    {formMeta.submitText}
                  </Button>
                </div>
              ) : (
                <div className="space-y-6" data-testid="join-room-form">
                  <div className="space-y-2">
                    <Label htmlFor="room-code" className="text-sm font-medium text-foreground" data-testid="label-room-code">Oda Kodu *</Label>
                    <Input
                      id="room-code"
                      placeholder="Örn: ABC123"
                      value={joinForm.room_code}
                      onChange={(e) => {
                        setJoinForm({ ...joinForm, room_code: e.target.value.toUpperCase(), room_password: "" });
                        setJoinRequiresPassword(false);
                        setJoinResolvedRoomName("");
                        setJoinError("");
                      }}
                      className={sharedInputClass}
                      data-testid="input-room-code"
                    />
                    <p className="mt-2 text-sm text-muted-foreground" data-testid="room-code-help-text">Oda sahibinden aldığın kodu buraya gir.</p>
                  </div>

                  {joinRequiresPassword && (
                    <div className="space-y-2" data-testid="join-room-password-wrap">
                      <Label htmlFor="join-room-password" className="text-sm font-medium text-foreground" data-testid="label-join-room-password">Oda Şifresi *</Label>
                      <Input
                        id="join-room-password"
                        type="password"
                        placeholder="Özel oda şifresini gir"
                        value={joinForm.room_password}
                        onChange={(e) => {
                          setJoinForm({ ...joinForm, room_password: e.target.value });
                          setJoinError("");
                        }}
                        className={sharedInputClass}
                        data-testid="input-join-room-password"
                      />
                      <p className="mt-2 text-sm text-muted-foreground" data-testid="join-room-password-help-text">
                        {joinResolvedRoomName
                          ? `"${joinResolvedRoomName}" özel oda olarak ayarlı. Devam etmek için şifre gir.`
                          : "Bu oda özel. Devam etmek için şifre gir."}
                      </p>
                    </div>
                  )}

                  {joinError && (
                    <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-700   " data-testid="join-room-error-message">
                      {joinError}
                    </div>
                  )}

                  <Button
                    className="h-11 w-full rounded-xl bg-indigo-600 text-white shadow-sm transition-all duration-200 hover:bg-indigo-700 hover:shadow-md"
                    onClick={handleJoinRoom}
                    disabled={loading || identityLoading}
                    data-testid={formMeta.submitTestId}
                  >
                    {formMeta.submitText}
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
