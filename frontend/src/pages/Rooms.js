import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import ThemeToggle from "@/components/ThemeToggle";
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
    "mt-2 h-11 rounded-xl border border-border/70 bg-secondary/70 text-foreground placeholder:text-muted-foreground shadow-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-0 dark:bg-secondary/80";

  const sharedSelectClass =
    "mt-2 h-11 w-full rounded-xl border border-border/70 bg-secondary/70 px-3 text-sm text-foreground shadow-none focus:outline-none focus:ring-2 focus:ring-ring dark:bg-secondary/80";

  const surfaceCardClass =
    "rounded-2xl border border-border/70 bg-card/95 text-card-foreground shadow-[0_12px_30px_-24px_rgba(15,23,42,0.45)]";

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
        <div className="space-y-8 sm:space-y-10">
          <header className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="space-y-1 pl-1 sm:pl-2">
              <button
                type="button"
                onClick={() => navigate("/dashboard")}
                className="font-display text-left text-[2rem] font-semibold tracking-[0.06em] text-foreground sm:text-[2.45rem] dark:bg-gradient-to-r dark:from-slate-50 dark:via-white dark:to-cyan-200 dark:bg-clip-text dark:text-transparent"
                data-testid="rooms-header-brand"
              >
                izlek
              </button>
              <p className="text-sm text-muted-foreground">odaklı çalışma alanın</p>
            </div>

            <div className="flex items-center justify-end gap-2" data-testid="rooms-theme-toggle-wrap">
              <ThemeToggle
                dataTestId="theme-toggle"
                className="border border-border bg-background/90 shadow-sm hover:bg-secondary"
              />
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

          <section className="relative overflow-hidden rounded-2xl border border-border/70 bg-card px-7 py-8 shadow-sm sm:px-9 sm:py-9" data-testid="rooms-hero-section">
            <div className="pointer-events-none absolute left-0 top-0 h-40 w-40 rounded-full bg-[radial-gradient(circle,rgba(56,189,248,0.18),transparent_68%)]" />
            <div className="pointer-events-none absolute bottom-0 right-0 h-36 w-36 rounded-full bg-[radial-gradient(circle,rgba(45,212,191,0.14),transparent_68%)]" />

            <div className="relative max-w-[960px] space-y-7">
              <div className="space-y-4">
                <p className="text-sm font-medium uppercase tracking-[0.18em] text-muted-foreground">
                  Birlikte daha odaklı çalış
                </p>
                <h1 className="font-display text-4xl font-semibold tracking-tight text-foreground sm:text-5xl lg:text-6xl" data-testid="rooms-title">
                  Online Çalışma Odaları
                </h1>
                <p className="max-w-3xl text-base leading-7 text-muted-foreground sm:text-lg" data-testid="rooms-subtitle">
                  Bu sayfa oda aksiyonlarına odaklanır: ihtiyacın olanı yukarıdan seç, aşağıda sadece ilgili formu temiz ve tekrar etmeyen bir akışla kullan.
                </p>
              </div>

              <div className="grid grid-cols-1 gap-3 sm:max-w-3xl sm:grid-cols-2" data-testid="rooms-quick-actions">
                <Button
                  variant="secondary"
                  size="lg"
                  onClick={() => focusFormsSection("create")}
                  className="hover-lift h-auto items-start justify-between whitespace-normal rounded-2xl px-5 py-4 text-left"
                  data-testid="rooms-quick-create-button"
                >
                  <div>
                    <div className="flex items-center gap-2 text-sm font-semibold">
                      <Plus className="h-4 w-4" />
                      Oda Oluştur
                    </div>
                    <p className="mt-1 text-xs font-medium text-muted-foreground sm:text-sm">
                      Yeni oda açma formunu aşağıda göster.
                    </p>
                  </div>
                </Button>

                <Button
                  variant="outline"
                  size="lg"
                  onClick={() => focusFormsSection("join")}
                  className="hover-lift h-auto items-start justify-between whitespace-normal rounded-2xl px-5 py-4 text-left"
                  data-testid="rooms-quick-join-button"
                >
                  <div>
                    <div className="flex items-center gap-2 text-sm font-semibold">
                      <LogIn className="h-4 w-4" />
                      Odaya Katıl
                    </div>
                    <p className="mt-1 text-xs font-medium text-muted-foreground sm:text-sm">
                      Mevcut oda kodunla ilerle.
                    </p>
                  </div>
                </Button>
              </div>
            </div>
          </section>

          <Card id="rooms-forms-section" className={surfaceCardClass} data-testid="rooms-main-card">
            <CardHeader className="pb-0">
              <div className="space-y-2">
                <CardTitle className="font-display text-2xl sm:text-3xl" data-testid="rooms-form-title">{formMeta.title}</CardTitle>
                <p className="text-sm leading-6 text-muted-foreground" data-testid="rooms-form-description">
                  {formMeta.description}
                </p>
              </div>
            </CardHeader>
            <CardContent className="pt-6 sm:pt-7">
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
                    className="h-11 w-full rounded-xl"
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
                    <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-700 dark:border-red-900/60 dark:bg-red-950/30 dark:text-red-300" data-testid="join-room-error-message">
                      {joinError}
                    </div>
                  )}

                  <Button
                    className="h-11 w-full rounded-xl"
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
