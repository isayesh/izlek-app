import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import axios from "axios";
import { API } from "@/App";
import { Home, Plus, LogIn, Trophy, Users, Globe, Lock } from "lucide-react";
import { saveRoom } from "@/lib/storage";
import { useAuth } from "@/contexts/AuthContext";

const ROOM_PASSWORD_MIN_LENGTH = 6;
const SHARED_ROOMS_STORAGE_KEY = "rooms_shared_public_v1";
const SHARED_PUBLIC_ROOMS = [
  { key: "izlek-ortak-oda", name: "İzlek Ortak Oda" },
  { key: "sessiz-calisma-odasi", name: "Sessiz Çalışma Odası" }
];

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
  const [availableRooms, setAvailableRooms] = useState([]);
  const [availableRoomsLoading, setAvailableRoomsLoading] = useState(true);

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

  useEffect(() => {
    const loadAvailableRooms = async () => {
      setAvailableRoomsLoading(true);

      try {
        const storedRaw = localStorage.getItem(SHARED_ROOMS_STORAGE_KEY);
        let storedSharedRooms = {};

        if (storedRaw) {
          try {
            storedSharedRooms = JSON.parse(storedRaw) || {};
          } catch {
            storedSharedRooms = {};
          }
        }

        const nextStoredSharedRooms = { ...storedSharedRooms };
        const resolvedRooms = [];

        for (const template of SHARED_PUBLIC_ROOMS) {
          let room = null;
          const savedCode = nextStoredSharedRooms?.[template.key]?.code;

          if (savedCode) {
            try {
              const lookup = await axios.get(`${API}/rooms/code/${savedCode}`);
              if (lookup.data?.id && !lookup.data?.is_private) {
                room = lookup.data;
              }
            } catch {
              room = null;
            }
          }

          if (!room) {
            const ownerId = currentUser?.uid || localStorage.getItem("userId") || "shared-lobby-system";
            const ownerName = currentUser?.displayName?.trim() || localStorage.getItem("userName") || "İzlek Ortak";

            try {
              const createRes = await axios.post(`${API}/rooms`, {
                name: template.name,
                owner_name: ownerName,
                owner_id: ownerId,
                owner_avatar_url: null,
                room_type: "public",
                room_password: null
              });

              if (createRes.data?.id) {
                room = createRes.data;
                nextStoredSharedRooms[template.key] = {
                  id: createRes.data.id,
                  code: createRes.data.code
                };
              }
            } catch (error) {
              console.error(`Error creating shared room ${template.name}:`, error);
            }
          }

          if (room?.id) {
            resolvedRooms.push(room);
          }
        }

        const lastRoomId = localStorage.getItem("currentRoomId");
        if (lastRoomId) {
          try {
            const response = await axios.get(`${API}/rooms/${lastRoomId}`);
            const room = response.data;
            if (room?.id) {
              resolvedRooms.push(room);
            }
          } catch {
            // ignore stale current room id
          }
        }

        const uniqueRooms = Array.from(
          new Map(resolvedRooms.filter((room) => room?.id).map((room) => [room.id, room])).values()
        );

        localStorage.setItem(SHARED_ROOMS_STORAGE_KEY, JSON.stringify(nextStoredSharedRooms));
        setAvailableRooms(uniqueRooms);
      } catch (error) {
        console.error("Error loading available rooms:", error);
        setAvailableRooms([]);
      } finally {
        setAvailableRoomsLoading(false);
      }
    };

    loadAvailableRooms();
  }, [currentUser?.uid]);

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

  const handleAvailableRoomJoin = (room) => {
    const roomCode = (room?.code || "").toUpperCase();
    if (!roomCode) return;

    const requiresPassword = room?.room_type === "private" || Boolean(room?.is_private);

    setJoinForm({ room_code: roomCode, room_password: "" });
    setJoinResolvedRoomName(room?.name || "");
    setJoinRequiresPassword(requiresPassword);
    setJoinError(requiresPassword ? `"${room?.name || "Bu oda"}" özel oda. Devam etmek için şifre gir.` : "");

    focusFormsSection("join");
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
          <header className="flex flex-col gap-3 border-b border-indigo-200 bg-indigo-50/35 pb-4 xl:flex-row xl:items-center xl:gap-4">
            <button
              type="button"
              onClick={() => navigate("/dashboard")}
              className="inline-flex h-12 w-fit shrink-0 items-center px-1"
              data-testid="rooms-header-brand"
              aria-label="Dashboard"
            >
              <span className="font-display text-4xl font-extrabold tracking-[-0.03em] leading-none text-gray-900">izlek</span>
            </button>

            <div className="w-full max-w-full flex-1 overflow-x-auto pb-1 xl:pb-0" data-testid="rooms-theme-toggle-wrap">
              <div className="flex items-center gap-2.5 lg:gap-3 xl:justify-end xl:gap-3.5">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => navigate("/")}
                  className="h-11 shrink-0 rounded-[15px] border border-transparent bg-transparent px-5 text-sm font-semibold tracking-[0.01em] text-slate-600 shadow-none [&_svg]:size-4 transition-colors duration-200 hover:border-indigo-200 hover:bg-indigo-100/70 hover:text-indigo-700 active:bg-indigo-100/80 focus-visible:border-indigo-200 focus-visible:bg-indigo-100/70"
                  data-testid="btn-home"
                >
                  <Home className="h-4 w-4" />
                  Ana Sayfa
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => navigate("/leaderboard")}
                  className="h-11 shrink-0 rounded-[15px] border border-transparent bg-transparent px-5 text-sm font-semibold tracking-[0.01em] text-slate-600 shadow-none [&_svg]:size-4 transition-colors duration-200 hover:border-indigo-200 hover:bg-indigo-100/70 hover:text-indigo-700 active:bg-indigo-100/80 focus-visible:border-indigo-200 focus-visible:bg-indigo-100/70"
                  data-testid="btn-leaderboard"
                >
                  <Trophy className="h-4 w-4" />
                  Liderlik Tablosu
                </Button>
              </div>
            </div>
          </header>

          <section className="grid gap-6 lg:grid-cols-[55%_45%] lg:gap-7" data-testid="rooms-lobby-layout">
            <Card id="rooms-forms-section" className={surfaceCardClass} data-testid="rooms-left-panel">
              <CardHeader className="pb-2">
                <div className="space-y-2">
                  <h1 className="font-display text-3xl font-bold tracking-tight text-foreground sm:text-4xl lg:text-5xl" data-testid="rooms-title">
                    Online Çalışma Odaları
                  </h1>
                  <p className="max-w-2xl text-sm leading-6 text-muted-foreground sm:text-base" data-testid="rooms-subtitle">
                    Oda oluştur veya bir odaya katıl.
                  </p>
                </div>

                <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2" data-testid="rooms-mode-switch">
                  <Button
                    type="button"
                    onClick={() => focusFormsSection("create")}
                    className={`h-auto items-start justify-between whitespace-normal rounded-xl border px-5 py-4 text-left shadow-none transition-all duration-200 ${
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
                    </div>
                  </Button>

                  <Button
                    type="button"
                    onClick={() => focusFormsSection("join")}
                    className={`h-auto items-start justify-between whitespace-normal rounded-xl border px-5 py-4 text-left shadow-none transition-all duration-200 ${
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
                    </div>
                  </Button>
                </div>
              </CardHeader>

              <CardContent className="pt-5 sm:pt-6">
                <div className="space-y-1.5">
                  <CardTitle className="font-display text-xl text-foreground/90 sm:text-2xl" data-testid="rooms-form-title">{formMeta.title}</CardTitle>
                  <p className="text-sm leading-6 text-muted-foreground/90" data-testid="rooms-form-description">
                    {formMeta.description}
                  </p>
                </div>

                {activeTab === "create" ? (
                  <div className="mt-6 space-y-6" data-testid="create-room-form">
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
                  <div className="mt-6 space-y-6" data-testid="join-room-form">
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
                      <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-700" data-testid="join-room-error-message">
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

            <Card className="rounded-2xl border border-border/60 bg-background/80 text-card-foreground shadow-none" data-testid="rooms-available-section">
              <CardHeader className="pb-2">
                <CardTitle className="font-display text-xl font-semibold tracking-tight text-foreground sm:text-2xl" data-testid="rooms-available-title">
                  Aktif Odalar
                </CardTitle>
                <p className="text-sm text-muted-foreground" data-testid="rooms-available-subtitle">
                  Her zaman açık ortak odalara doğrudan katılabilirsin.
                </p>
              </CardHeader>

              <CardContent className="pt-4">
                <div className="rounded-xl border border-border/60 bg-background/75 overflow-hidden" data-testid="rooms-available-list-container">
                  <div className="grid grid-cols-[1fr_auto_auto] items-center gap-2 border-b border-border/60 bg-indigo-50/40 px-4 py-2 text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
                    <span>Oda</span>
                    <span>Durum</span>
                    <span className="text-right">Aksiyon</span>
                  </div>

                  {availableRoomsLoading ? (
                    <div className="px-4 py-4 text-sm text-muted-foreground" data-testid="rooms-available-loading">
                      Odalar yükleniyor...
                    </div>
                  ) : availableRooms.length > 0 ? (
                    <div className="divide-y divide-border/50" data-testid="rooms-available-grid">
                      {availableRooms.map((room) => {
                        const participantCount = room?.participants?.length || 0;
                        const isPrivate = room?.room_type === "private" || Boolean(room?.is_private);
                        const isSharedRoom = SHARED_PUBLIC_ROOMS.some((template) => template.name === room?.name);

                        return (
                          <div
                            key={room.id}
                            className="group flex items-start justify-between gap-4 px-4 py-5 transition-colors duration-200 hover:bg-indigo-50/45"
                            data-testid={`rooms-available-item-${room.id}`}
                          >
                            <div className="min-w-0">
                              <div className="flex items-center gap-2">
                                <p className="text-base font-semibold text-foreground">{room.name}</p>
                                {isSharedRoom && (
                                  <span className="inline-flex h-5 items-center rounded-md border border-indigo-200 bg-indigo-50 px-1.5 text-[10px] font-medium uppercase tracking-[0.08em] text-indigo-700">
                                    Ortak
                                  </span>
                                )}
                              </div>
                              <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                                <span className="inline-flex items-center gap-1.5">
                                  <Users className="h-3.5 w-3.5 text-indigo-500" />
                                  {participantCount} kişi
                                </span>
                                <span aria-hidden="true">•</span>
                                <span className="inline-flex items-center gap-1.5">
                                  {isPrivate ? <Lock className="h-3.5 w-3.5 text-indigo-500" /> : <Globe className="h-3.5 w-3.5 text-indigo-500" />}
                                  {isPrivate ? "Özel oda" : "Herkese açık"}
                                </span>
                              </div>
                            </div>

                            <div className="flex shrink-0 flex-col items-end gap-2">
                              <span className={`inline-flex h-7 items-center rounded-md px-2 text-xs font-medium ${participantCount > 0 ? "bg-indigo-50 text-indigo-700" : "bg-slate-100 text-slate-600"}`}>
                                {participantCount > 0 ? "Aktif" : "Bekliyor"}
                              </span>
                              <Button
                                onClick={() => handleAvailableRoomJoin(room)}
                                className="h-8 rounded-lg bg-indigo-600 px-3 text-xs text-white shadow-sm transition-all duration-200 hover:bg-indigo-700"
                                data-testid={`rooms-available-join-${room.id}`}
                              >
                                Katıl
                              </Button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="px-4 py-5" data-testid="rooms-available-empty">
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <p className="text-sm font-medium text-foreground">Ortak odalar şu an listelenemiyor</p>
                          <p className="mt-1 text-sm text-muted-foreground">
                            Birkaç saniye sonra tekrar deneyebilir veya soldan oda kodu ile doğrudan katılabilirsin.
                          </p>
                        </div>
                        <span className="inline-flex h-7 items-center rounded-md bg-indigo-50 px-2 text-xs font-medium text-indigo-700">Lobi</span>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </section>
        </div>
      </div>
    </div>
  );
}
