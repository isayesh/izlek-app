import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import {
  AlertTriangle,
  BarChart3,
  Edit,
  Flame,
  Home,
  LogOut,
  MessageSquare,
  Plus,
  Search,
  Star,
  Trash2,
  Trophy,
  User,
  Users,
} from "lucide-react";

import { API } from "@/App";
import { useAuth } from "@/contexts/AuthContext";
import { getStoredPrograms, savePrograms } from "@/lib/storage";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const DAY_ORDER = [
  "Pazartesi",
  "Salı",
  "Çarşamba",
  "Perşembe",
  "Cuma",
  "Cumartesi",
  "Pazar",
];

const findKnownDayName = (dayName = "") => {
  const normalizedInput = dayName.toLocaleLowerCase("tr-TR");
  return DAY_ORDER.find((day) => day.toLocaleLowerCase("tr-TR") === normalizedInput) || null;
};

const normalizeTaskDay = (dayName = "") => findKnownDayName(dayName) || dayName;

const getCurrentDayName = () => findKnownDayName(new Date().toLocaleDateString("tr-TR", { weekday: "long" })) || DAY_ORDER[0];
const GRADE_LEVEL_LABELS = {
  "11": "11. Sınıf",
  "12": "12. Sınıf",
  mezun: "Mezun",
};
const STUDY_FIELD_LABELS = {
  Sayısal: "Sayısal",
  EA: "Eşit Ağırlık",
  "Eşit Ağırlık": "Eşit Ağırlık",
  Sözel: "Sözel",
  Dil: "Dil",
};

const getDashboardProfileMeta = (gradeLevel, studyField) => {
  const parts = [GRADE_LEVEL_LABELS[gradeLevel] || gradeLevel, STUDY_FIELD_LABELS[studyField] || studyField].filter(Boolean);
  return parts.length ? parts.join(" • ") : null;
};

const YKS_TARGET_DATE = new Date("2026-06-20T10:15:00");

const getYksCountdown = () => {
  const remaining = Math.max(YKS_TARGET_DATE.getTime() - Date.now(), 0);
  const totalSeconds = Math.floor(remaining / 1000);

  return {
    days: Math.floor(totalSeconds / 86400),
    hours: Math.floor((totalSeconds % 86400) / 3600),
    minutes: Math.floor((totalSeconds % 3600) / 60),
    seconds: totalSeconds % 60,
  };
};

function TaskFormFields({ draft, setDraft, prefix }) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor={`${prefix}-lesson`}>Ders</Label>
        <Input
          id={`${prefix}-lesson`}
          value={draft.lesson}
          onChange={(e) => setDraft({ ...draft, lesson: e.target.value })}
          placeholder="Örn: Matematik"
          className="h-11 rounded-xl"
          data-testid={`input-${prefix}-lesson`}
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor={`${prefix}-topic`}>Konu</Label>
        <Input
          id={`${prefix}-topic`}
          value={draft.topic}
          onChange={(e) => setDraft({ ...draft, topic: e.target.value })}
          placeholder="Örn: İntegral"
          className="h-11 rounded-xl"
          data-testid={`input-${prefix}-topic`}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor={`${prefix}-duration`}>Süre (dakika)</Label>
          <Input
            id={`${prefix}-duration`}
            type="number"
            value={draft.duration}
            onChange={(e) => setDraft({ ...draft, duration: parseInt(e.target.value, 10) || 30 })}
            className="h-11 rounded-xl"
            data-testid={`input-${prefix}-duration`}
          />
        </div>

        <div className="space-y-2">
          <Label>Gün</Label>
          <Select
            value={draft.day}
            onValueChange={(value) => setDraft({ ...draft, day: value })}
          >
            <SelectTrigger className="h-11 rounded-xl" data-testid={`select-${prefix}-day`}>
              <SelectValue placeholder="Gün seçin" />
            </SelectTrigger>
            <SelectContent>
              {DAY_ORDER.map((day) => (
                <SelectItem key={day} value={day}>
                  {day}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const navigate = useNavigate();
  const { logout, currentUser } = useAuth();

  const [programs, setPrograms] = useState([]);
  const [selectedProgram, setSelectedProgram] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showAddTask, setShowAddTask] = useState(false);
  const [networkError, setNetworkError] = useState(false);
  const [activeWeeklyDay, setActiveWeeklyDay] = useState(getCurrentDayName());
  const [pendingFriendRequestCount, setPendingFriendRequestCount] = useState(0);
  const [unreadDmCount, setUnreadDmCount] = useState(0);

  const [newTask, setNewTask] = useState({
    lesson: "",
    topic: "",
    duration: 30,
    day: "Pazartesi",
  });

  const [showEditTask, setShowEditTask] = useState(false);
  const [editingTask, setEditingTask] = useState(null);
  const [editTaskDraft, setEditTaskDraft] = useState({
    lesson: "",
    topic: "",
    duration: 30,
    day: "Pazartesi",
  });

  const [profileId, setProfileId] = useState(null);
  const [userName, setUserName] = useState("");
  const [profileData, setProfileData] = useState({
    username: "",
    handle: "",
    display_name: "",
    avatar_url: "",
    grade_level: "",
    study_field: "",
    streak_count: 0,
    last_active_date: null,
  });
  const [avatarLoadFailed, setAvatarLoadFailed] = useState(false);
  const [yksCountdown, setYksCountdown] = useState(() => getYksCountdown());
  const [dailyRating, setDailyRating] = useState(0);
  const [hoverRating, setHoverRating] = useState(0);
  const profileMetaLine = getDashboardProfileMeta(profileData.grade_level, profileData.study_field);

  const getTodayDateString = () => {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, "0");
    const day = String(now.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
  };

  const getFallbackName = () =>
    currentUser?.displayName?.trim() ||
    currentUser?.email?.trim() ||
    "İzlek Kullanıcısı";

  const displayNameCandidate = profileData.display_name?.trim() || currentUser?.displayName?.trim();
  const heroGreetingName = profileData.username?.trim() || displayNameCandidate || getFallbackName();

  useEffect(() => {
    const intervalId = setInterval(() => {
      setYksCountdown(getYksCountdown());
    }, 1000);

    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => {
    const checkProfileExists = async () => {
      if (!currentUser?.uid) return;

      const storedProfileId = localStorage.getItem("profileId");
      const fallbackName = getFallbackName();

      try {
        const res = await axios.get(`${API}/profile`, {
          params: { firebase_uid: currentUser.uid },
        });

        if (res.data && res.data.id) {
          const backendProfileId = res.data.id;
          const backendUserName = res.data.username || res.data.name || fallbackName;

          localStorage.setItem("profileId", backendProfileId);
          localStorage.setItem("userName", backendUserName);

          setProfileId(backendProfileId);
          setUserName(backendUserName);
          setProfileData({
            username: backendUserName,
            handle: res.data.handle || "",
            display_name: res.data.display_name || res.data.displayName || res.data.name || "",
            avatar_url: res.data.avatar_url || "",
            grade_level: res.data.grade_level || "",
            study_field: res.data.study_field || "",
            streak_count: res.data.streak_count || 0,
            last_active_date: res.data.last_active_date || null,
          });
        } else if (storedProfileId) {
          setProfileId(storedProfileId);
          setUserName(fallbackName);
          setProfileData({
            username: fallbackName,
            handle: "",
            display_name: currentUser?.displayName?.trim() || "",
            avatar_url: "",
            grade_level: "",
            study_field: "",
            streak_count: 0,
            last_active_date: null,
          });
        } else {
          navigate("/program/create");
        }
      } catch (error) {
        console.error("Error fetching profile from backend:", error);
        if (storedProfileId) {
          const storedUserName = localStorage.getItem("userName") || fallbackName;
          setProfileId(storedProfileId);
          setUserName(storedUserName);
          setProfileData({
            username: storedUserName,
            handle: "",
            display_name: currentUser?.displayName?.trim() || "",
            avatar_url: "",
            grade_level: "",
            study_field: "",
            streak_count: 0,
            last_active_date: null,
          });
        } else {
          navigate("/program/create");
        }
      }
    };

    checkProfileExists();
  }, [currentUser, navigate]);

  useEffect(() => {
    const loadPendingFriendRequestCount = async () => {
      if (!currentUser?.uid) {
        setPendingFriendRequestCount(0);
        return;
      }

      try {
        const response = await axios.get(`${API}/friends/requests/incoming/count`, {
          headers: { "X-Firebase-UID": currentUser.uid },
        });
        setPendingFriendRequestCount(Number(response.data?.count) || 0);
      } catch (countError) {
        console.error("Error loading pending friend request count:", countError);
      }
    };

    loadPendingFriendRequestCount();
  }, [currentUser]);

  useEffect(() => {
    const loadUnreadDmCount = async () => {
      if (!currentUser?.uid) {
        setUnreadDmCount(0);
        return;
      }

      try {
        const response = await axios.get(`${API}/messages/direct/unread-count`, {
          headers: { "X-Firebase-UID": currentUser.uid },
        });
        setUnreadDmCount(Number(response.data?.count) || 0);
      } catch (countError) {
        console.error("Error loading unread DM count:", countError);
      }
    };

    loadUnreadDmCount();
  }, [currentUser]);

  useEffect(() => {
    setAvatarLoadFailed(false);
  }, [profileData.avatar_url]);

  useEffect(() => {
    if (!profileId) return;

    const storedPrograms = getStoredPrograms();
    if (storedPrograms && storedPrograms.length > 0) {
      setPrograms(storedPrograms);
      setSelectedProgram(storedPrograms[0]);
      setLoading(false);
    }

    loadPrograms();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [profileId]);

  const loadPrograms = async () => {
    try {
      const res = await axios.get(`${API}/programs/${profileId}`);
      const fetchedPrograms = res.data;

      if (Array.isArray(fetchedPrograms) && fetchedPrograms.length > 0) {
        setPrograms(fetchedPrograms);
        setSelectedProgram(fetchedPrograms[0]);
        savePrograms(fetchedPrograms);
        setNetworkError(false);
      }
    } catch (error) {
      console.error("Error loading programs:", error);

      if (error.request) {
        console.warn("Network error, using cached data");
        setNetworkError(true);
      } else {
        console.warn("Backend error, using cached data");
      }
    } finally {
      setLoading(false);
    }
  };

  const toggleTaskComplete = async (taskId) => {
    const updatedTasks = selectedProgram.tasks.map((task) =>
      task.id === taskId ? { ...task, completed: !task.completed } : task
    );

    try {
      await axios.put(`${API}/programs/${selectedProgram.id}`, { tasks: updatedTasks });

      const updatedProgram = { ...selectedProgram, tasks: updatedTasks };
      setSelectedProgram(updatedProgram);

      const updatedPrograms = programs.map((program) =>
        program.id === selectedProgram.id ? updatedProgram : program
      );
      setPrograms(updatedPrograms);
      savePrograms(updatedPrograms);
    } catch (error) {
      console.error("Error updating task:", error);
    }
  };

  const addTask = async () => {
    if (!newTask.lesson.trim() || !newTask.topic.trim()) {
      alert("Ders ve konu alanları boş olamaz");
      return;
    }

    const task = {
      id: Date.now().toString(),
      lesson: newTask.lesson,
      topic: newTask.topic,
      duration: newTask.duration,
      day: newTask.day,
      completed: false,
    };

    const updatedTasks = [...selectedProgram.tasks, task];

    try {
      await axios.put(`${API}/programs/${selectedProgram.id}`, { tasks: updatedTasks });

      const updatedProgram = { ...selectedProgram, tasks: updatedTasks };
      setSelectedProgram(updatedProgram);
      setNewTask({ lesson: "", topic: "", duration: 30, day: "Pazartesi" });
      setShowAddTask(false);

      const updatedPrograms = programs.map((program) =>
        program.id === selectedProgram.id ? updatedProgram : program
      );
      setPrograms(updatedPrograms);
      savePrograms(updatedPrograms);
    } catch (error) {
      console.error("Error adding task:", error);
    }
  };

  const deleteTask = async (taskId) => {
    const updatedTasks = selectedProgram.tasks.filter((task) => task.id !== taskId);

    try {
      await axios.put(`${API}/programs/${selectedProgram.id}`, { tasks: updatedTasks });

      const updatedProgram = { ...selectedProgram, tasks: updatedTasks };
      setSelectedProgram(updatedProgram);

      const updatedPrograms = programs.map((program) =>
        program.id === selectedProgram.id ? updatedProgram : program
      );
      setPrograms(updatedPrograms);
      savePrograms(updatedPrograms);
    } catch (error) {
      console.error("Error deleting task:", error);
    }
  };

  const editTask = (task) => {
    setEditingTask(task);
    setEditTaskDraft({
      lesson: task.lesson || "",
      topic: task.topic || "",
      duration: task.duration || 30,
      day: task.day || "Pazartesi",
    });
    setShowEditTask(true);
  };

  const saveEditedTask = async () => {
    if (!editingTask) return;

    if (!editTaskDraft.lesson.trim() || !editTaskDraft.topic.trim()) {
      alert("Ders ve konu alanları boş olamaz");
      return;
    }

    const updatedTasks = selectedProgram.tasks.map((task) =>
      task.id === editingTask.id
        ? {
            ...task,
            lesson: editTaskDraft.lesson,
            topic: editTaskDraft.topic,
            duration: editTaskDraft.duration,
            day: editTaskDraft.day,
          }
        : task
    );

    try {
      await axios.put(`${API}/programs/${selectedProgram.id}`, { tasks: updatedTasks });

      const updatedProgram = { ...selectedProgram, tasks: updatedTasks };
      setSelectedProgram(updatedProgram);

      const updatedPrograms = programs.map((program) =>
        program.id === selectedProgram.id ? updatedProgram : program
      );
      setPrograms(updatedPrograms);
      savePrograms(updatedPrograms);

      setShowEditTask(false);
      setEditingTask(null);
    } catch (error) {
      console.error("Error editing task:", error);
    }
  };

  const getTodaysTasks = () => {
    const currentDay = getCurrentDayName();
    return selectedProgram?.tasks.filter((task) => normalizeTaskDay(task.day) === currentDay) || [];
  };

  const groupTasksByDay = () => {
    const grouped = Object.fromEntries(DAY_ORDER.map((day) => [day, []]));

    selectedProgram?.tasks.forEach((task) => {
      const day = normalizeTaskDay(task.day);
      if (!grouped[day]) grouped[day] = [];
      grouped[day].push(task);
    });

    return grouped;
  };

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center px-6 py-10">
        <div className="rounded-2xl border border-border/70 bg-card px-8 py-10 text-center shadow-sm">
          <p className="font-display text-2xl font-semibold text-foreground">Dashboard hazırlanıyor</p>
          <p className="mt-2 text-sm text-muted-foreground">Programın ve günlük görevlerin yükleniyor.</p>
        </div>
      </div>
    );
  }

  if (!selectedProgram) {
    return (
      <div className="flex min-h-screen items-center justify-center px-6 py-10">
        <Card className="w-full max-w-xl border-border/70 bg-card/95 text-left shadow-sm">
          <CardHeader>
            <CardTitle className="font-display text-3xl">Henüz program oluşturmadın</CardTitle>
            <CardDescription>
              İlk çalışma planını oluşturarak dashboard alanını görevlerinle doldurabilirsin.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={() => navigate("/program/create")} data-testid="dashboard-create-program-button">
              Program Oluştur
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const todaysTasks = getTodaysTasks();
  const tasksByDay = groupTasksByDay();
  const completedTodayCount = todaysTasks.filter((task) => task.completed).length;
  const progressValue = todaysTasks.length === 0 ? 0 : Math.round((completedTodayCount / todaysTasks.length) * 100);
  const isDailyGoalCompleted = todaysTasks.length > 0 && progressValue === 100;
  const activeRating = hoverRating || dailyRating;
  const shouldShowStreakReminder =
    profileData.streak_count > 0 && profileData.last_active_date !== getTodayDateString();

  const navigationActions = [
    { label: "Ana Sayfa", icon: Home, onClick: () => navigate("/"), testId: "btn-home" },
    { label: "Net Takibi", icon: BarChart3, onClick: () => navigate("/net-tracking"), testId: "btn-net-tracking" },
    { label: "Odalar", icon: Users, onClick: () => navigate("/rooms"), testId: "btn-rooms" },
    { label: "Liderlik", icon: Trophy, onClick: () => navigate("/leaderboard"), testId: "btn-leaderboard" },
    { label: "Profil", icon: User, onClick: () => navigate("/profile"), testId: "dashboard-btn-profile" },
    { label: "DM Kutum", icon: MessageSquare, onClick: () => navigate("/messages"), testId: "dashboard-btn-messages", badgeCount: unreadDmCount },
    { label: "Arkadaşlar", icon: Users, onClick: () => navigate("/friends"), testId: "dashboard-btn-friends", badgeCount: pendingFriendRequestCount },
  ];

  const quickActions = [
    {
      label: "Oda Bul",
      description: "Hazır çalışma odalarını keşfet",
      icon: Search,
      onClick: () => navigate("/rooms"),
      variant: "default",
      emphasis: "primary",
      className: "border border-indigo-200 bg-indigo-600 text-white shadow-sm hover:bg-indigo-700 hover:border-indigo-300 hover:shadow-sm",
      testId: "dashboard-quick-action-find-room-button",
    },
  ];

  const countdownSegments = [
    { label: "Gün", value: yksCountdown.days },
    { label: "Saat", value: yksCountdown.hours },
    { label: "Dakika", value: yksCountdown.minutes },
    { label: "Saniye", value: yksCountdown.seconds },
  ];

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="px-4 pb-10 pt-6 sm:px-6 sm:pb-12 lg:px-10 xl:px-12">
        <div className="space-y-8 sm:space-y-10">
          <header className="flex flex-col gap-4 py-1 sm:py-2 xl:flex-row xl:items-center xl:justify-between">
            <button
              type="button"
              onClick={() => navigate("/dashboard")}
              className="mr-8 inline-flex h-12 w-fit shrink-0 items-center px-1 xl:mr-10"
              data-testid="dashboard-header-brand"
              aria-label="Dashboard"
            >
              <span className="font-display text-4xl font-bold tracking-tight leading-none text-gray-900">izlek</span>
            </button>

            <div className="flex flex-col gap-3 xl:items-end">
              <div className="flex items-center justify-end gap-2" data-testid="dashboard-theme-toggle">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={async () => {
                    await logout();
                    navigate("/");
                  }}
                  className="shrink-0 border-border/50 bg-background/70 text-foreground shadow-none hover:border-indigo-200 hover:bg-indigo-50/70 hover:text-indigo-700"
                  data-testid="btn-logout"
                >
                  <LogOut className="h-4 w-4" />
                  Çıkış
                </Button>
              </div>

              <div className="w-full max-w-full overflow-x-auto pb-1" data-testid="dashboard-header-nav">
                <div className="flex items-center gap-2.5 lg:gap-3 xl:justify-end xl:gap-3.5">
                  {navigationActions.map((action) => {
                    const Icon = action.icon;
                    const badgeLabel = action.badgeCount > 9 ? "9+" : action.badgeCount;
                    return (
                      <Button
                        key={action.label}
                        variant="ghost"
                        size="sm"
                        onClick={action.onClick}
                        className="h-11 shrink-0 rounded-[15px] border border-transparent px-5 text-sm font-semibold tracking-[0.01em] text-muted-foreground shadow-none [&_svg]:size-4 hover:border-indigo-100 hover:bg-indigo-50/70 hover:text-indigo-700 active:bg-indigo-100/70"
                        data-testid={action.testId}
                      >
                        <span className="relative inline-flex items-center justify-center">
                          <Icon className="h-4 w-4" />
                          {action.badgeCount > 0 && (
                            <span className="absolute -right-2 -top-2 inline-flex min-w-[18px] items-center justify-center rounded-full bg-red-500 px-1.5 py-0.5 text-[10px] font-semibold leading-none text-white shadow-sm" data-testid={`${action.testId}-badge`}>
                              {badgeLabel}
                            </span>
                          )}
                        </span>
                        <span>{action.label}</span>
                      </Button>
                    );
                  })}
                </div>
              </div>
            </div>
          </header>

          {networkError && (
            <Alert
              variant="destructive"
              className="rounded-2xl border border-destructive/20 bg-destructive/10 text-foreground"
              data-testid="network-error-banner"
            >
              <AlertTriangle className="h-4 w-4" />
              <div>
                <AlertTitle>Bağlantı sorunu algılandı</AlertTitle>
                <AlertDescription>
                  Şu anda kaydedilmiş veriler gösteriliyor. İnternet bağlantın düzeldiğinde programın yeniden senkronize edilecek.
                </AlertDescription>
              </div>
            </Alert>
          )}

          <section
            className="relative overflow-hidden rounded-[28px] border border-border/50 bg-card/90 px-6 py-7 shadow-sm sm:px-8 sm:py-9"
            data-testid="dashboard-hero-section"
          >

            <div className="relative flex flex-col gap-8 md:gap-8 xl:flex-row xl:items-start xl:justify-between xl:gap-10">
              <div className="w-full space-y-5 xl:max-w-[62%]">
                <div className="max-w-2xl space-y-3">
                  <h1 className="font-display text-5xl font-bold tracking-tight text-foreground sm:text-6xl lg:text-7xl" data-testid="dashboard-title">
                    Merhaba, {heroGreetingName}
                  </h1>
                  <p className="text-base leading-7 text-muted-foreground/90 sm:text-lg" data-testid="dashboard-program-summary">
                    Bugünün planını netleştir ve çalışmaya başla.
                  </p>
                </div>

                <div className="w-full rounded-[24px] border border-border/40 bg-background/35 px-5 py-4 sm:px-6 sm:py-5 xl:max-w-[680px]">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <p className="text-[11px] font-medium uppercase tracking-[0.2em] text-muted-foreground sm:text-xs">
                      YKS'ye kalan süre
                    </p>
                    <p className="text-xs text-muted-foreground">20 Haziran 2026 • 10:15</p>
                  </div>

                  <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-4 sm:grid-cols-4 sm:gap-x-5 sm:gap-y-0 md:gap-x-6">
                    {countdownSegments.map((segment, index) => (
                      <div
                        key={segment.label}
                        className={`flex flex-col gap-1 ${index > 0 ? "sm:border-l sm:border-border/45 sm:pl-5" : ""}`}
                      >
                        <span className="font-display text-3xl font-semibold tracking-[-0.04em] text-foreground sm:text-[2.1rem] ">
                          {String(segment.value).padStart(2, "0")}
                        </span>
                        <span className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
                          {segment.label}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="w-full xl:max-w-[320px]">
                <div className="rounded-[24px] border border-border/45 bg-background/45 p-5 sm:p-6" data-testid="dashboard-identity-block">
                  <div className="flex items-center gap-4">
                    <div
                      className="flex h-16 w-16 flex-shrink-0 items-center justify-center overflow-hidden rounded-[20px] border border-border/50 bg-secondary/80 text-xl font-semibold text-foreground"
                      data-testid="dashboard-avatar"
                    >
                      {profileData.avatar_url && !avatarLoadFailed ? (
                        <img
                          src={profileData.avatar_url}
                          alt={`${userName} avatar`}
                          className="h-full w-full object-cover"
                          data-testid="dashboard-avatar-image"
                          onError={() => setAvatarLoadFailed(true)}
                        />
                      ) : (
                        (userName || getFallbackName()).trim().charAt(0).toUpperCase()
                      )}
                    </div>

                    <div className="min-w-0 space-y-1">
                      <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">Profil özeti</p>
                      <p className="truncate text-lg font-semibold text-foreground">{userName}</p>
                      <p className="text-sm leading-6 text-muted-foreground">
                        {profileMetaLine || "Profil detaylarınla dashboard özetini kişiselleştir."}
                      </p>
                    </div>
                  </div>

                  <div className="my-5 h-px bg-border/60" />

                  <div className="flex items-start justify-between gap-4" data-testid="dashboard-progress-summary">
                    <div>
                      <p className="text-sm text-muted-foreground">Günlük ilerlemen</p>
                      <div className={`mt-1 flex items-start ${isDailyGoalCompleted ? "gap-4" : ""}`}>
                        <p
                          className={`text-3xl tracking-[-0.03em] ${
                            isDailyGoalCompleted ? "font-bold text-indigo-700" : "font-semibold text-foreground"
                          }`}
                          data-testid="progress-percentage"
                        >
                          %{progressValue}
                        </p>

                        {isDailyGoalCompleted && (
                          <div className="pt-1" data-testid="daily-rating-widget">
                            <div className="flex items-center gap-1">
                              {[1, 2, 3, 4, 5].map((value) => {
                                const isFilled = value <= activeRating;
                                return (
                                  <button
                                    key={value}
                                    type="button"
                                    onClick={() => setDailyRating(value)}
                                    onMouseEnter={() => setHoverRating(value)}
                                    onMouseLeave={() => setHoverRating(0)}
                                    className="inline-flex items-center justify-center text-indigo-500"
                                    aria-label={`Günü ${value} yıldız ile değerlendir`}
                                    data-testid={`daily-rating-star-${value}`}
                                  >
                                    <Star className={`h-4 w-4 ${isFilled ? "fill-indigo-500" : "fill-transparent"}`} />
                                  </button>
                                );
                              })}
                            </div>
                            {dailyRating > 0 && (
                              <p className="mt-1 text-xs text-muted-foreground">Bugününü değerlendirdin: {dailyRating}/5</p>
                            )}
                          </div>
                        )}
                      </div>

                      {isDailyGoalCompleted && (
                        <p className="mt-2 inline-flex items-center gap-1.5 text-sm font-medium text-indigo-600">
                          <span className="text-indigo-700" aria-hidden="true">✓</span>
                          Günlük hedef tamamlandı
                        </p>
                      )}
                    </div>

                    {!isDailyGoalCompleted && (
                      <div className="text-right text-sm leading-6 text-muted-foreground">
                        <p>{completedTodayCount}/{todaysTasks.length || 0} görev bugün bitti</p>
                      </div>
                    )}
                  </div>
                  <Progress value={progressValue} className="mt-4 h-2.5 bg-secondary/80" />

                  {shouldShowStreakReminder && (
                    <div className="mt-4 flex items-start gap-2 text-sm text-indigo-700" data-testid="dashboard-streak-reminder">
                      <Flame className="mt-0.5 h-4 w-4 flex-shrink-0 text-indigo-500" />
                      <span>Bugün çalışmanı işaretleyerek serini koru.</span>
                    </div>
                  )}
                </div>

                <div className="mt-5 flex justify-start xl:justify-end">
                  {quickActions.map((action) => {
                    const Icon = action.icon;
                    return (
                      <Button
                        key={action.label}
                        variant={action.variant}
                        size="lg"
                        onClick={action.onClick}
                        className={`hover-lift h-auto min-h-[88px] w-full items-start justify-between whitespace-normal rounded-[22px] px-5 py-4 text-left xl:w-auto xl:min-w-[200px] ${action.className}`}
                        data-testid={action.testId}
                      >
                        <div>
                          <div className="flex items-center gap-2 text-sm font-semibold sm:text-[15px]">
                            <Icon className="h-4 w-4" />
                            {action.label}
                          </div>
                          <p className={`mt-2 text-sm leading-6 ${action.emphasis === "primary" ? "text-indigo-100" : "text-muted-foreground"}`}>
                            {action.description}
                          </p>
                        </div>
                      </Button>
                    );
                  })}
                </div>
              </div>
            </div>
          </section>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,6fr)_minmax(0,4fr)] lg:gap-8">
            <Card className="border-border/50 bg-card/85 shadow-none" data-testid="todays-tasks-card">
              <CardHeader className="pb-0">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                  <div className="space-y-2">
                    <CardTitle className="font-display text-2xl tracking-[-0.03em] sm:text-3xl">Bugün Yapılacaklar</CardTitle>
                    <CardDescription>
                      Günlük önceliklerini ferah bir listede takip et, tamamlananları anında işaretle.
                    </CardDescription>
                  </div>

                  <div className="flex flex-wrap items-center justify-end gap-3">
                    <Button
                      onClick={() => setShowAddTask(true)}
                      className="h-auto rounded-lg bg-indigo-600 px-3 py-1.5 text-white shadow-sm transition-all duration-200 hover:bg-indigo-700 hover:shadow-md"
                      data-testid="btn-add-task"
                    >
                      <Plus className="h-4 w-4" />
                      Görev Ekle
                    </Button>
                  </div>
                </div>
              </CardHeader>

              <CardContent className="pt-6">
                {todaysTasks.length === 0 ? (
                  <div className="rounded-[24px] border border-dashed border-border/55 bg-background/35 px-6 py-12 text-center" data-testid="today-tasks-empty-state">
                    <p className="font-display text-2xl font-semibold tracking-[-0.03em] text-foreground">Bugün için görev görünmüyor</p>
                    <p className="mt-2 text-sm leading-6 text-muted-foreground">
                      Haftalık programına yeni bir görev ekleyerek bugünün odağını oluşturabilirsin.
                    </p>
                    <Button className="mt-6 shadow-none" onClick={() => setShowAddTask(true)}>
                      <Plus className="h-4 w-4" />
                      İlk görevi ekle
                    </Button>
                  </div>
                ) : (
                  <ScrollArea className="max-h-[560px] pr-3" data-testid="today-tasks-list">
                    <div className="overflow-hidden rounded-[24px] border border-border/45 bg-background/40">
                      <div className="divide-y divide-border/50">
                        {todaysTasks.map((task) => (
                          <div
                            key={task.id}
                            className={`group px-4 py-4 transition-colors duration-200 sm:px-5 ${
                              task.completed ? "bg-indigo-50/35 hover:bg-indigo-50/55" : "hover:bg-indigo-50/50"
                            }`}
                            data-testid={`task-${task.id}`}
                          >
                            <div className="flex items-start gap-4">
                              <Checkbox
                                checked={task.completed}
                                onCheckedChange={() => toggleTaskComplete(task.id)}
                                className="mt-1 border-border/70 data-[state=checked]:border-indigo-300 data-[state=checked]:bg-indigo-50 data-[state=checked]:text-indigo-600"
                                data-testid={`checkbox-${task.id}`}
                              />

                              <div className="min-w-0 flex-1">
                                <div className={`flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-[0.18em] ${
                                  task.completed ? "text-muted-foreground/75" : "text-muted-foreground"
                                }`}>
                                  <span>{task.day}</span>
                                  <span className="h-1 w-1 rounded-full bg-border" />
                                  <span>{task.duration} dakika</span>
                                </div>
                                <p
                                  className={`mt-3 text-lg font-semibold tracking-[-0.02em] ${
                                    task.completed
                                      ? "text-muted-foreground/85 line-through decoration-1 decoration-muted-foreground/50"
                                      : "text-foreground"
                                  }`}
                                >
                                  {task.lesson}
                                </p>
                                <p className={`mt-1 text-sm leading-6 ${task.completed ? "text-muted-foreground/75" : "text-muted-foreground"}`}>
                                  {task.topic}
                                </p>
                              </div>

                              <div className="flex items-center gap-1">
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => editTask(task)}
                                  aria-label="Görevi düzenle"
                                  className="rounded-md text-slate-500 hover:bg-slate-100/80 hover:text-slate-700"
                                  data-testid={`edit-${task.id}`}
                                >
                                  <Edit className="h-4 w-4" />
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => deleteTask(task.id)}
                                  aria-label="Görevi sil"
                                  className="rounded-md text-red-500 hover:bg-red-50 hover:text-red-600"
                                  data-testid={`delete-${task.id}`}
                                >
                                  <Trash2 className="h-4 w-4" />
                                </Button>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </ScrollArea>
                )}
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-card/85 shadow-none" data-testid="weekly-program-card">
              <CardHeader className="pb-0">
                <div className="space-y-2.5">
                  <div className="flex items-center justify-between gap-4">
                    <CardTitle className="font-display text-xl tracking-[-0.03em] sm:text-2xl">Haftalık Program</CardTitle>
                    <p className="text-sm text-muted-foreground">
                      {selectedProgram.tasks.length} görev
                    </p>
                  </div>
                  <CardDescription className="leading-6">
                    Tüm haftayı gün gün incele, görevlerini aynı yerden düzenle veya sil.
                  </CardDescription>
                </div>
              </CardHeader>

              <CardContent className="pt-6">
                <Tabs value={activeWeeklyDay} onValueChange={setActiveWeeklyDay} className="space-y-4" data-testid="weekly-program-tabs">
                  <div className="w-full overflow-hidden pb-2">
                    <TabsList className="flex h-auto w-full flex-nowrap gap-1 rounded-[20px] border border-border/50 bg-background/50 p-1">
                      {DAY_ORDER.map((day) => (
                        <TabsTrigger
                          key={day}
                          value={day}
                          className="min-w-0 flex-1 whitespace-nowrap rounded-[16px] px-2 py-2 text-[13px] leading-none text-muted-foreground data-[state=active]:bg-card data-[state=active]:text-foreground data-[state=active]:shadow-none"
                          data-testid={`weekly-program-tab-${day.toLocaleLowerCase("tr-TR").replace(/[^a-z0-9]+/g, "-")}`}
                        >
                          {day}
                        </TabsTrigger>
                      ))}
                    </TabsList>
                  </div>

                  {DAY_ORDER.map((day) => {
                    const dayTasks = tasksByDay[day] || [];

                    return (
                      <TabsContent key={day} value={day} data-testid="weekly-program-day-panel">
                        {dayTasks.length === 0 ? (
                          <div className="rounded-[24px] border border-dashed border-border/55 bg-background/35 px-4 py-10 text-center text-sm text-muted-foreground">
                            Bu gün için planlanmış görev görünmüyor.
                          </div>
                        ) : (
                          <ScrollArea className="max-h-[560px] pr-3">
                            <div className="overflow-hidden rounded-[24px] border border-border/45 bg-background/40">
                              <div className="divide-y divide-border/50">
                                {dayTasks.map((task) => (
                                  <div
                                    key={task.id}
                                    className={`group px-4 py-4 transition-colors duration-200 sm:px-5 ${
                                      task.completed ? "bg-indigo-50/35 hover:bg-indigo-50/55" : "hover:bg-indigo-50/50"
                                    }`}
                                  >
                                    <div className="flex items-start justify-between gap-3">
                                      <div className="min-w-0">
                                        <p
                                          className={`text-sm font-semibold leading-6 ${
                                            task.completed
                                              ? "text-muted-foreground/85 line-through decoration-1 decoration-muted-foreground/50"
                                              : "text-foreground"
                                          }`}
                                        >
                                          {task.lesson} · {task.topic}
                                        </p>
                                        <p className={`mt-1 text-[11px] uppercase tracking-[0.18em] ${task.completed ? "text-muted-foreground/75" : "text-muted-foreground"}`}>
                                          {task.duration} dakika
                                        </p>
                                      </div>

                                      <div className="flex items-center gap-1">
                                        <Button
                                          variant="ghost"
                                          size="icon"
                                          onClick={() => editTask(task)}
                                          aria-label="Görevi düzenle"
                                          className="rounded-md text-slate-500 hover:bg-slate-100/80 hover:text-slate-700"
                                          data-testid={`weekly-edit-${task.id}`}
                                        >
                                          <Edit className="h-4 w-4" />
                                        </Button>
                                        <Button
                                          variant="ghost"
                                          size="icon"
                                          onClick={() => deleteTask(task.id)}
                                          aria-label="Görevi sil"
                                          className="rounded-md text-red-500 hover:bg-red-50 hover:text-red-600"
                                          data-testid={`weekly-delete-${task.id}`}
                                        >
                                          <Trash2 className="h-4 w-4" />
                                        </Button>
                                      </div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </ScrollArea>
                        )}
                      </TabsContent>
                    );
                  })}
                </Tabs>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      <Dialog open={showAddTask} onOpenChange={setShowAddTask}>
        <DialogContent className="w-[calc(100vw-2rem)] rounded-2xl border border-border/70 bg-card p-0 shadow-xl sm:max-w-lg" data-testid="task-create-dialog">
          <div className="p-6 sm:p-7">
            <DialogHeader>
              <DialogTitle className="font-display text-2xl">Yeni Görev Ekle</DialogTitle>
            </DialogHeader>

            <div className="mt-6 space-y-6">
              <TaskFormFields draft={newTask} setDraft={setNewTask} prefix="new" />
              <div className="flex flex-col-reverse gap-3 sm:flex-row sm:justify-end">
                <Button variant="outline" onClick={() => setShowAddTask(false)} data-testid="task-create-cancel-button">
                  İptal
                </Button>
                <Button onClick={addTask} data-testid="btn-save-task">
                  Kaydet
                </Button>
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog
        open={showEditTask}
        onOpenChange={(open) => {
          setShowEditTask(open);
          if (!open) setEditingTask(null);
        }}
      >
        <DialogContent className="w-[calc(100vw-2rem)] rounded-2xl border border-border/70 bg-card p-0 shadow-xl sm:max-w-lg" data-testid="task-edit-dialog">
          <div className="p-6 sm:p-7">
            <DialogHeader>
              <DialogTitle className="font-display text-2xl">Görevi Düzenle</DialogTitle>
            </DialogHeader>

            <div className="mt-6 space-y-6">
              <TaskFormFields draft={editTaskDraft} setDraft={setEditTaskDraft} prefix="edit" />
              <div className="flex flex-col-reverse gap-3 sm:flex-row sm:justify-end">
                <Button
                  variant="outline"
                  onClick={() => {
                    setShowEditTask(false);
                    setEditingTask(null);
                  }}
                  data-testid="task-edit-cancel-button"
                >
                  İptal
                </Button>
                <Button onClick={saveEditedTask} data-testid="task-edit-save-button">Kaydet</Button>
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
