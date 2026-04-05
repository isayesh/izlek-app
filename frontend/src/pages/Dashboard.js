import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import {
  AlertTriangle,
  BarChart3,
  Bell,
  CalendarDays,
  Clock3,
  Edit,
  Flame,
  Home,
  LogOut,
  Plus,
  Search,
  Target,
  Trash2,
  Trophy,
  User,
  Users,
} from "lucide-react";

import { API } from "@/App";
import ThemeToggle from "@/components/ThemeToggle";
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
    avatar_url: "",
    study_goal: "",
    daily_study_hours: null,
    streak_count: 0,
    last_active_date: null,
  });
  const [avatarLoadFailed, setAvatarLoadFailed] = useState(false);

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
            avatar_url: res.data.avatar_url || "",
            study_goal: res.data.study_goal || "",
            daily_study_hours: res.data.daily_study_hours,
            streak_count: res.data.streak_count || 0,
            last_active_date: res.data.last_active_date || null,
          });
        } else if (storedProfileId) {
          setProfileId(storedProfileId);
          setUserName(fallbackName);
          setProfileData({
            username: fallbackName,
            avatar_url: "",
            study_goal: "",
            daily_study_hours: null,
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
            avatar_url: "",
            study_goal: "",
            daily_study_hours: null,
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

  const getProgress = () => {
    if (!selectedProgram || selectedProgram.tasks.length === 0) return 0;
    const completed = selectedProgram.tasks.filter((task) => task.completed).length;
    return Math.round((completed / selectedProgram.tasks.length) * 100);
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
  const progressValue = getProgress();
  const completedTodayCount = todaysTasks.filter((task) => task.completed).length;
  const totalCompletedTasks = selectedProgram.tasks.filter((task) => task.completed).length;
  const shouldShowStreakReminder =
    profileData.streak_count > 0 && profileData.last_active_date !== getTodayDateString();

  const navigationActions = [
    { label: "Ana Sayfa", icon: Home, onClick: () => navigate("/"), testId: "btn-home" },
    { label: "Net Takibi", icon: BarChart3, onClick: () => navigate("/net-tracking"), testId: "btn-net-tracking" },
    { label: "Odalar", icon: Users, onClick: () => navigate("/rooms"), testId: "btn-rooms" },
    { label: "Liderlik", icon: Trophy, onClick: () => navigate("/leaderboard"), testId: "btn-leaderboard" },
    { label: "Profil", icon: User, onClick: () => navigate("/profile"), testId: "dashboard-btn-profile" },
    { label: "Arkadaşlar", icon: Users, onClick: () => navigate("/friends"), testId: "dashboard-btn-friends" },
    { label: "Bildirimler", icon: Bell, onClick: () => navigate("/notifications"), testId: "dashboard-btn-notifications" },
  ];

  const quickActions = [
    {
      label: "Oda Bul",
      description: "Hazır çalışma odalarını keşfet",
      icon: Search,
      onClick: () => navigate("/rooms"),
      variant: "default",
      testId: "dashboard-quick-action-find-room-button",
    },
    {
      label: "Oda Oluştur",
      description: "Kendi odanı aç ve davet et",
      icon: Users,
      onClick: () => navigate("/rooms"),
      variant: "secondary",
      testId: "dashboard-quick-action-create-room-button",
    },
    {
      label: "Görev Ekle",
      description: "Programına yeni bir çalışma ekle",
      icon: Plus,
      onClick: () => setShowAddTask(true),
      variant: "outline",
      testId: "dashboard-quick-action-add-task-button",
    },
  ];

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="px-4 pb-10 pt-6 sm:px-6 sm:pb-12 lg:px-10 xl:px-12">
        <div className="space-y-8 sm:space-y-10">
          <header className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="space-y-1 pl-1 sm:pl-2">
              <button
                type="button"
                onClick={() => navigate("/dashboard")}
                className="font-display text-left text-[2rem] font-semibold tracking-[0.06em] text-foreground sm:text-[2.45rem] dark:bg-gradient-to-r dark:from-slate-50 dark:via-white dark:to-cyan-200 dark:bg-clip-text dark:text-transparent"
                data-testid="dashboard-header-brand"
              >
                izlek
              </button>
              <p className="text-sm text-muted-foreground">odaklı çalışma alanın</p>
            </div>

            <div className="flex flex-col gap-3 lg:items-end">
              <div className="flex items-center justify-end gap-2" data-testid="dashboard-theme-toggle">
                <ThemeToggle
                  dataTestId="theme-toggle"
                  className="border border-border bg-background/90 shadow-sm hover:bg-secondary"
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={async () => {
                    await logout();
                    navigate("/");
                  }}
                  className="shrink-0"
                  data-testid="btn-logout"
                >
                  <LogOut className="h-4 w-4" />
                  Çıkış
                </Button>
              </div>

              <div className="w-full max-w-full overflow-x-auto pb-1" data-testid="dashboard-header-nav">
                <div className="flex items-center gap-2 lg:justify-end">
                  {navigationActions.map((action) => {
                    const Icon = action.icon;
                    return (
                      <Button
                        key={action.label}
                        variant="outline"
                        size="sm"
                        onClick={action.onClick}
                        className="shrink-0"
                        data-testid={action.testId}
                      >
                        <Icon className="h-4 w-4" />
                        {action.label}
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
            className="relative overflow-hidden rounded-2xl border border-border/70 bg-card px-6 py-7 shadow-sm sm:px-8 sm:py-8"
            data-testid="dashboard-hero-section"
          >
            <div className="pointer-events-none absolute left-0 top-0 h-40 w-40 rounded-full bg-[radial-gradient(circle,rgba(56,189,248,0.18),transparent_68%)]" />
            <div className="pointer-events-none absolute bottom-0 right-0 h-36 w-36 rounded-full bg-[radial-gradient(circle,rgba(45,212,191,0.14),transparent_68%)]" />

            <div className="relative flex flex-col gap-8 xl:flex-row xl:items-start xl:justify-between">
              <div className="max-w-3xl space-y-6">
                <div className="space-y-3">
                  <p className="text-sm font-medium uppercase tracking-[0.18em] text-muted-foreground">
                    Bugün için net bir düzen
                  </p>
                  <div className="space-y-3">
                    <h1 className="font-display text-4xl font-semibold tracking-tight text-foreground sm:text-5xl lg:text-6xl" data-testid="dashboard-title">
                      Merhaba, {userName}
                    </h1>
                    <p className="max-w-2xl text-base leading-7 text-muted-foreground sm:text-lg" data-testid="dashboard-program-summary">
                      {selectedProgram.exam_goal} hedefin için bugün yapacakların hazır. Günlük {selectedProgram.daily_hours} saat odağınla ritmini koru ve planını ferah bir ekranda yönet.
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                  <div className="rounded-2xl border border-border/70 bg-background/70 p-4">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Target className="h-4 w-4 text-accent" />
                      Hedef
                    </div>
                    <p className="mt-3 text-lg font-semibold text-foreground">
                      {profileData.study_goal || selectedProgram.exam_goal}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-border/70 bg-background/70 p-4">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Clock3 className="h-4 w-4 text-accent" />
                      Günlük tempo
                    </div>
                    <p className="mt-3 text-lg font-semibold text-foreground">
                      {profileData.daily_study_hours ?? selectedProgram.daily_hours} saat
                    </p>
                  </div>

                  <div className="rounded-2xl border border-border/70 bg-background/70 p-4" data-testid="dashboard-streak-summary">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Flame className="h-4 w-4 text-amber-500" />
                      Çalışma serisi
                    </div>
                    <p className="mt-3 text-lg font-semibold text-foreground" data-testid="dashboard-streak-value">
                      {profileData.streak_count > 0 ? `${profileData.streak_count} gün` : "Yeni seri başlat"}
                    </p>
                  </div>
                </div>
              </div>

              <div className="grid w-full gap-3 xl:max-w-sm">
                <div className="rounded-2xl border border-border/70 bg-background/70 p-5" data-testid="dashboard-identity-block">
                  <div className="flex items-center gap-4">
                    <div
                      className="flex h-16 w-16 flex-shrink-0 items-center justify-center overflow-hidden rounded-2xl border border-border/70 bg-secondary text-xl font-semibold text-foreground"
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

                    <div className="space-y-1">
                      <p className="text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">Profil özeti</p>
                      <p className="text-lg font-semibold text-foreground">{userName}</p>
                      <p className="text-sm text-muted-foreground">
                        {profileData.study_goal || "Planın hazır, odak noktan belli."}
                      </p>
                    </div>
                  </div>

                  {shouldShowStreakReminder && (
                    <div className="mt-4 rounded-2xl border border-amber-500/20 bg-amber-500/10 px-4 py-3 text-sm text-amber-700 dark:text-amber-200" data-testid="dashboard-streak-reminder">
                      Bugün çalışmanı işaretleyerek serini koru.
                    </div>
                  )}
                </div>

                <div className="rounded-2xl border border-border/70 bg-background/70 p-5" data-testid="dashboard-progress-summary">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <p className="text-sm text-muted-foreground">Genel ilerleme</p>
                      <p className="mt-1 text-2xl font-semibold text-foreground" data-testid="progress-percentage">
                        %{progressValue}
                      </p>
                    </div>
                    <div className="text-right text-sm text-muted-foreground">
                      <p>{totalCompletedTasks}/{selectedProgram.tasks.length} görev tamamlandı</p>
                      <p>{completedTodayCount}/{todaysTasks.length || 0} görev bugün bitti</p>
                    </div>
                  </div>
                  <Progress value={progressValue} className="mt-4 h-3 bg-secondary" />
                </div>
              </div>
            </div>

            <div className="relative mt-8 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
              {quickActions.map((action) => {
                const Icon = action.icon;
                return (
                  <Button
                    key={action.label}
                    variant={action.variant}
                    size="lg"
                    onClick={action.onClick}
                    className="hover-lift h-auto items-start justify-between rounded-2xl px-5 py-4 text-left"
                    data-testid={action.testId}
                  >
                    <div>
                      <div className="flex items-center gap-2 text-sm font-semibold">
                        <Icon className="h-4 w-4" />
                        {action.label}
                      </div>
                      <p className="mt-1 text-xs font-medium text-muted-foreground sm:text-sm">
                        {action.description}
                      </p>
                    </div>
                  </Button>
                );
              })}
            </div>
          </section>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,6fr)_minmax(0,4fr)] lg:gap-8">
            <Card className="border-border/70 bg-card/95" data-testid="todays-tasks-card">
              <CardHeader className="pb-0">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                  <div className="space-y-2">
                    <CardTitle className="font-display text-2xl sm:text-3xl">Bugün Yapılacaklar</CardTitle>
                    <CardDescription>
                      Günlük önceliklerini ferah bir listede takip et, tamamlananları anında işaretle.
                    </CardDescription>
                  </div>

                  <div className="flex flex-wrap items-center gap-3">
                    <div className="rounded-2xl border border-border/70 bg-background/70 px-4 py-3 text-right">
                      <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Bugün</p>
                      <p className="mt-1 text-lg font-semibold text-foreground">
                        {completedTodayCount}/{todaysTasks.length}
                      </p>
                    </div>
                    <Button onClick={() => setShowAddTask(true)} data-testid="btn-add-task">
                      <Plus className="h-4 w-4" />
                      Görev Ekle
                    </Button>
                  </div>
                </div>
              </CardHeader>

              <CardContent className="pt-6">
                {todaysTasks.length === 0 ? (
                  <div className="rounded-2xl border border-dashed border-border/70 bg-background/60 px-6 py-12 text-center" data-testid="today-tasks-empty-state">
                    <p className="font-display text-2xl font-semibold text-foreground">Bugün için görev görünmüyor</p>
                    <p className="mt-2 text-sm text-muted-foreground">
                      Haftalık programına yeni bir görev ekleyerek bugünün odağını oluşturabilirsin.
                    </p>
                    <Button className="mt-6" onClick={() => setShowAddTask(true)}>
                      <Plus className="h-4 w-4" />
                      İlk görevi ekle
                    </Button>
                  </div>
                ) : (
                  <ScrollArea className="max-h-[560px] pr-3" data-testid="today-tasks-list">
                    <div className="space-y-3">
                      {todaysTasks.map((task) => (
                        <div
                          key={task.id}
                          className="rounded-2xl border border-border/70 bg-background/80 p-4 shadow-sm transition-shadow duration-200 hover:shadow-md"
                          data-testid={`task-${task.id}`}
                        >
                          <div className="flex items-start gap-4">
                            <Checkbox
                              checked={task.completed}
                              onCheckedChange={() => toggleTaskComplete(task.id)}
                              className="mt-1"
                              data-testid={`checkbox-${task.id}`}
                            />

                            <div className="min-w-0 flex-1">
                              <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
                                <span>{task.day}</span>
                                <span className="h-1 w-1 rounded-full bg-border" />
                                <span>{task.duration} dakika</span>
                              </div>
                              <p
                                className={`mt-3 text-lg font-semibold ${
                                  task.completed ? "text-muted-foreground line-through" : "text-foreground"
                                }`}
                              >
                                {task.lesson}
                              </p>
                              <p className={`mt-1 text-sm ${task.completed ? "text-muted-foreground" : "text-muted-foreground"}`}>
                                {task.topic}
                              </p>
                            </div>

                            <div className="flex items-center gap-1">
                              <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => editTask(task)}
                                aria-label="Görevi düzenle"
                                data-testid={`edit-${task.id}`}
                              >
                                <Edit className="h-4 w-4 text-accent" />
                              </Button>
                              <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => deleteTask(task.id)}
                                aria-label="Görevi sil"
                                data-testid={`delete-${task.id}`}
                              >
                                <Trash2 className="h-4 w-4 text-destructive" />
                              </Button>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                )}
              </CardContent>
            </Card>

            <Card className="border-border/70 bg-card/95" data-testid="weekly-program-card">
              <CardHeader className="pb-0">
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-4">
                    <CardTitle className="font-display text-xl sm:text-2xl">Haftalık Program</CardTitle>
                    <div className="flex items-center gap-2 rounded-2xl border border-border/70 bg-background/70 px-4 py-2 text-sm text-muted-foreground">
                      <CalendarDays className="h-4 w-4 text-accent" />
                      {selectedProgram.tasks.length} görev
                    </div>
                  </div>
                  <CardDescription>
                    Tüm haftayı gün gün incele, görevlerini aynı yerden düzenle veya sil.
                  </CardDescription>
                </div>
              </CardHeader>

              <CardContent className="pt-6">
                <Tabs value={activeWeeklyDay} onValueChange={setActiveWeeklyDay} className="space-y-4" data-testid="weekly-program-tabs">
                  <div className="w-full overflow-hidden pb-2">
                    <TabsList className="flex h-auto w-full flex-nowrap gap-1 rounded-2xl bg-secondary/80 p-1">
                      {DAY_ORDER.map((day) => (
                        <TabsTrigger
                          key={day}
                          value={day}
                          className="min-w-0 flex-1 whitespace-nowrap px-2 py-2 text-[13px] leading-none"
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
                          <div className="rounded-2xl border border-dashed border-border/70 bg-background/60 px-4 py-10 text-center text-sm text-muted-foreground">
                            Bu gün için planlanmış görev görünmüyor.
                          </div>
                        ) : (
                          <ScrollArea className="max-h-[560px] pr-3">
                            <div className="space-y-3">
                              {dayTasks.map((task) => (
                                <div
                                  key={task.id}
                                  className="rounded-2xl border border-border/70 bg-background/80 p-4 shadow-sm transition-shadow duration-200 hover:shadow-md"
                                >
                                  <div className="flex items-start justify-between gap-3">
                                    <div className="min-w-0">
                                      <p className={`text-sm font-semibold ${task.completed ? "text-muted-foreground line-through" : "text-foreground"}`}>
                                        {task.lesson} · {task.topic}
                                      </p>
                                      <p className="mt-1 text-xs uppercase tracking-[0.18em] text-muted-foreground">
                                        {task.duration} dakika
                                      </p>
                                    </div>

                                    <div className="flex items-center gap-1">
                                      <Button
                                        variant="ghost"
                                        size="icon"
                                        onClick={() => editTask(task)}
                                        aria-label="Görevi düzenle"
                                        data-testid={`weekly-edit-${task.id}`}
                                      >
                                        <Edit className="h-4 w-4 text-accent" />
                                      </Button>
                                      <Button
                                        variant="ghost"
                                        size="icon"
                                        onClick={() => deleteTask(task.id)}
                                        aria-label="Görevi sil"
                                        data-testid={`weekly-delete-${task.id}`}
                                      >
                                        <Trash2 className="h-4 w-4 text-destructive" />
                                      </Button>
                                    </div>
                                  </div>
                                </div>
                              ))}
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
