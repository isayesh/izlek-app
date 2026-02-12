import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from "@/components/ui/dialog";
import axios from "axios";
import { API } from "@/App";
import { Plus, Trash2, Edit, Users, Home, LogOut, BarChart3 } from "lucide-react";
import { getStoredPrograms, savePrograms } from "@/lib/storage";
import { useAuth } from "@/contexts/AuthContext";

export default function Dashboard() {
  const navigate = useNavigate();
  const { logout, currentUser } = useAuth();

  const [programs, setPrograms] = useState([]);
  const [selectedProgram, setSelectedProgram] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showAddTask, setShowAddTask] = useState(false);
  const [networkError, setNetworkError] = useState(false);

  const [newTask, setNewTask] = useState({
    lesson: "",
    topic: "",
    duration: 30,
    day: "Pazartesi"
  });

  // Edit modal state (AYRI!)
  const [showEditTask, setShowEditTask] = useState(false);
  const [editingTask, setEditingTask] = useState(null);
  const [editTaskDraft, setEditTaskDraft] = useState({
    lesson: "",
    topic: "",
    duration: 30,
    day: "Pazartesi"
  });

  // Read profileId from localStorage on mount and store in state
  const [profileId, setProfileId] = useState(null);
  const [userName, setUserName] = useState("");

  // Check for profileId on mount - fetch from backend using Firebase UID
  useEffect(() => {
    const checkProfileExists = async () => {
      if (!currentUser?.uid) return;

      // First check localStorage
      const storedProfileId = localStorage.getItem("profileId");
      const storedUserName = localStorage.getItem("userName");

      if (storedProfileId) {
        setProfileId(storedProfileId);
        setUserName(storedUserName || "");
        return;
      }

      // If not in localStorage, check backend
      try {
        const res = await axios.get(`${API}/profiles/by-firebase-uid/${currentUser.uid}`);
        if (res.data && res.data.id) {
          const backendProfileId = res.data.id;
          const backendUserName = res.data.name;

          localStorage.setItem("profileId", backendProfileId);
          localStorage.setItem("userName", backendUserName);

          setProfileId(backendProfileId);
          setUserName(backendUserName);
        } else {
          navigate("/program/create");
        }
      } catch (error) {
        console.error("Error fetching profile from backend:", error);
        navigate("/program/create");
      }
    };

    checkProfileExists();
  }, [currentUser, navigate]);

  useEffect(() => {
    if (!profileId) return;

    // Immediately hydrate from localStorage
    const storedPrograms = getStoredPrograms();
    if (storedPrograms && storedPrograms.length > 0) {
      setPrograms(storedPrograms);
      setSelectedProgram(storedPrograms[0]);
      setLoading(false);
    }

    // Then fetch from backend
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

      const updatedPrograms = programs.map((p) =>
        p.id === selectedProgram.id ? updatedProgram : p
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
      completed: false
    };

    const updatedTasks = [...selectedProgram.tasks, task];

    try {
      await axios.put(`${API}/programs/${selectedProgram.id}`, { tasks: updatedTasks });

      const updatedProgram = { ...selectedProgram, tasks: updatedTasks };
      setSelectedProgram(updatedProgram);
      setNewTask({ lesson: "", topic: "", duration: 30, day: "Pazartesi" });
      setShowAddTask(false);

      const updatedPrograms = programs.map((p) =>
        p.id === selectedProgram.id ? updatedProgram : p
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

      const updatedPrograms = programs.map((p) =>
        p.id === selectedProgram.id ? updatedProgram : p
      );
      setPrograms(updatedPrograms);
      savePrograms(updatedPrograms);
    } catch (error) {
      console.error("Error deleting task:", error);
    }
  };

  // EDIT: open modal + fill draft
  const editTask = (task) => {
    setEditingTask(task);
    setEditTaskDraft({
      lesson: task.lesson || "",
      topic: task.topic || "",
      duration: task.duration || 30,
      day: task.day || "Pazartesi"
    });
    setShowEditTask(true);
  };

  // EDIT: save
  const saveEditedTask = async () => {
    if (!editingTask) return;

    if (!editTaskDraft.lesson.trim() || !editTaskDraft.topic.trim()) {
      alert("Ders ve konu alanları boş olamaz");
      return;
    }

    const updatedTasks = selectedProgram.tasks.map((t) =>
      t.id === editingTask.id
        ? {
            ...t,
            lesson: editTaskDraft.lesson,
            topic: editTaskDraft.topic,
            duration: editTaskDraft.duration,
            day: editTaskDraft.day
          }
        : t
    );

    try {
      await axios.put(`${API}/programs/${selectedProgram.id}`, { tasks: updatedTasks });

      const updatedProgram = { ...selectedProgram, tasks: updatedTasks };
      setSelectedProgram(updatedProgram);

      const updatedPrograms = programs.map((p) =>
        p.id === selectedProgram.id ? updatedProgram : p
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
    const today = new Date().toLocaleDateString("tr-TR", { weekday: "long" });
    const dayMap = {
      Pazartesi: "Pazartesi",
      Salı: "Salı",
      Çarşamba: "Çarşamba",
      Perşembe: "Perşembe",
      Cuma: "Cuma",
      Cumartesi: "Cumartesi",
      Pazar: "Pazar"
    };

    return selectedProgram?.tasks.filter((task) => task.day === dayMap[today] || task.day === today) || [];
  };

  const getProgress = () => {
    if (!selectedProgram || selectedProgram.tasks.length === 0) return 0;
    const completed = selectedProgram.tasks.filter((t) => t.completed).length;
    return Math.round((completed / selectedProgram.tasks.length) * 100);
  };

  const groupTasksByDay = () => {
    const grouped = {};
    selectedProgram?.tasks.forEach((task) => {
      if (!grouped[task.day]) grouped[task.day] = [];
      grouped[task.day].push(task);
    });
    return grouped;
  };

  if (loading) {
    return <div className="flex items-center justify-center min-h-screen">Yükleniyor...</div>;
  }

  if (!selectedProgram) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen p-4">
        <p className="text-xl mb-4">Henüz program oluşturmadın</p>
        <Button onClick={() => navigate("/program/create")}>Program Oluştur</Button>
      </div>
    );
  }

  const todaysTasks = getTodaysTasks();
  const tasksByDay = groupTasksByDay();

  return (
    <div className="min-h-screen p-6 md:p-10">
      {/* Network Error Banner */}
      {networkError && (
        <div
          className="max-w-7xl mx-auto mb-4 bg-yellow-50 border border-yellow-200 text-yellow-800 px-4 py-3 rounded"
          data-testid="network-error-banner"
        >
          <p className="text-sm font-medium">⚠️ Bağlantı hatası. Kaydedilmiş veriler gösteriliyor.</p>
        </div>
      )}

      {/* Header */}
      <div className="max-w-7xl mx-auto mb-12">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-5xl font-extrabold text-gray-900 mb-3" data-testid="dashboard-title">
              Merhaba, {userName}! 👋
            </h1>
            <p className="text-lg text-gray-600 font-medium">
              {selectedProgram.exam_goal} • {selectedProgram.daily_hours} saat/gün
            </p>
          </div>
          <div className="flex gap-3">
            <Button
              variant="outline"
              onClick={() => navigate("/")}
              className="shadow-sm hover:shadow-md transition-shadow"
              data-testid="btn-home"
            >
              <Home className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              onClick={() => navigate("/net-tracking")}
              className="shadow-sm hover:shadow-md transition-shadow"
              data-testid="btn-net-tracking"
            >
              <BarChart3 className="mr-2 h-4 w-4" />
              Net Takibi
            </Button>
            <Button
              variant="outline"
              onClick={() => navigate("/rooms")}
              className="shadow-sm hover:shadow-md transition-shadow"
              data-testid="btn-rooms"
            >
              <Users className="mr-2 h-4 w-4" />
              Odalara Git
            </Button>
            <Button
              variant="outline"
              onClick={async () => {
                await logout();
                navigate("/");
              }}
              className="shadow-sm hover:shadow-md transition-shadow"
              data-testid="btn-logout"
            >
              <LogOut className="mr-2 h-4 w-4" />
              Çıkış
            </Button>
          </div>
        </div>

        {/* Progress */}
        <Card className="mb-10 shadow-md border border-gray-200">
          <CardContent className="pt-6 pb-6">
            <div className="flex justify-between items-center mb-4">
              <span className="text-lg font-bold text-gray-800">Genel İlerleme</span>
              <span className="text-2xl font-extrabold text-primary" data-testid="progress-percentage">
                {getProgress()}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-5 shadow-inner">
              <div
                className="bg-gradient-to-r from-primary to-accent h-5 rounded-full transition-all duration-500 shadow-sm"
                style={{ width: `${getProgress()}%` }}
              />
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="max-w-7xl mx-auto grid md:grid-cols-5 gap-10">
        {/* Today's Tasks */}
        <Card className="md:col-span-3 border-2 border-primary/40 shadow-xl" data-testid="todays-tasks-card">
          <CardHeader className="pb-5">
            <CardTitle className="flex justify-between items-center text-3xl">
              <span className="text-primary font-extrabold">Bugün Yapılacaklar</span>
              <span className="text-lg font-bold text-gray-500">
                {todaysTasks.filter((t) => t.completed).length}/{todaysTasks.length}
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {todaysTasks.length === 0 ? (
              <p className="text-gray-500 text-center py-16 text-lg">Bugün için görev yok</p>
            ) : (
              <div className="space-y-5">
                {todaysTasks.map((task) => (
                  <div
                    key={task.id}
                    className="flex items-start gap-4 p-5 rounded-xl border-2 hover:bg-gray-50 hover:border-primary/40 hover:shadow-md transition-all duration-200"
                    data-testid={`task-${task.id}`}
                  >
                    <Checkbox
                      checked={task.completed}
                      onCheckedChange={() => toggleTaskComplete(task.id)}
                      className="mt-1"
                      data-testid={`checkbox-${task.id}`}
                    />
                    <div className="flex-1">
                      <p
                        className={`font-bold text-lg ${
                          task.completed ? "line-through text-gray-400" : "text-gray-900"
                        }`}
                      >
                        {task.lesson} - {task.topic}
                      </p>
                      <p className="text-sm text-gray-500 mt-1.5 font-medium">{task.duration} dakika</p>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => deleteTask(task.id)}
                      className="hover:bg-red-50 transition-colors"
                      data-testid={`delete-${task.id}`}
                    >
                      <Trash2 className="h-4 w-4 text-red-500" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Weekly Program */}
        <Card className="md:col-span-2 shadow-md border border-gray-200" data-testid="weekly-program-card">
          <CardHeader className="pb-4">
            <CardTitle className="flex justify-between items-center text-lg">
              <span className="text-gray-700">Haftalık Program</span>

              <Dialog open={showAddTask} onOpenChange={setShowAddTask}>
                <DialogTrigger asChild>
                  <Button
                    size="sm"
                    variant="outline"
                    className="shadow-sm hover:shadow-md transition-shadow"
                    data-testid="btn-add-task"
                  >
                    <Plus className="h-4 w-4 mr-1" />
                    Ekle
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Yeni Görev Ekle</DialogTitle>
                  </DialogHeader>
                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="lesson">Ders</Label>
                      <Input
                        id="lesson"
                        value={newTask.lesson}
                        onChange={(e) => setNewTask({ ...newTask, lesson: e.target.value })}
                        placeholder="Örn: Matematik"
                        data-testid="input-new-lesson"
                      />
                    </div>
                    <div>
                      <Label htmlFor="topic">Konu</Label>
                      <Input
                        id="topic"
                        value={newTask.topic}
                        onChange={(e) => setNewTask({ ...newTask, topic: e.target.value })}
                        placeholder="Örn: İntegral"
                        data-testid="input-new-topic"
                      />
                    </div>
                    <div>
                      <Label htmlFor="duration">Süre (dakika)</Label>
                      <Input
                        id="duration"
                        type="number"
                        value={newTask.duration}
                        onChange={(e) =>
                          setNewTask({ ...newTask, duration: parseInt(e.target.value) || 30 })
                        }
                        data-testid="input-new-duration"
                      />
                    </div>
                    <div>
                      <Label htmlFor="day">Gün</Label>
                      <select
                        id="day"
                        className="w-full p-2 border-2 rounded-lg"
                        value={newTask.day}
                        onChange={(e) => setNewTask({ ...newTask, day: e.target.value })}
                        data-testid="select-new-day"
                      >
                        <option>Pazartesi</option>
                        <option>Salı</option>
                        <option>Çarşamba</option>
                        <option>Perşembe</option>
                        <option>Cuma</option>
                        <option>Cumartesi</option>
                        <option>Pazar</option>
                      </select>
                    </div>
                    <Button onClick={addTask} className="w-full" data-testid="btn-save-task">
                      Kaydet
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>
            </CardTitle>
          </CardHeader>

          <CardContent>
            <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2">
              {Object.keys(tasksByDay).length === 0 ? (
                <p className="text-gray-500 text-center py-8 text-sm">Henüz görev eklenmedi</p>
              ) : (
                Object.entries(tasksByDay).map(([day, tasks]) => (
                  <div key={day} className="border-b pb-4 last:border-b-0">
                    <h3 className="font-bold mb-3 text-gray-700 text-sm">{day}</h3>
                    <div className="space-y-2">
                      {tasks.map((task) => (
                        <div
                          key={task.id}
                          className="flex justify-between items-center text-sm p-3 bg-gray-50 rounded-lg border hover:bg-gray-100 transition"
                        >
                          <div>
                            <span className={task.completed ? "line-through text-gray-400" : "text-gray-700"}>
                              {task.lesson} - {task.topic}
                            </span>
                            <div className="text-xs text-gray-500 font-medium">{task.duration} dk</div>
                          </div>

                          {/* Actions */}
                          <div className="flex items-center gap-1">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => editTask(task)}
                              className="hover:bg-blue-50 transition-colors"
                              aria-label="Düzenle"
                            >
                              <Edit className="h-4 w-4 text-blue-600" />
                            </Button>

                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => deleteTask(task.id)}
                              className="hover:bg-red-50 transition-colors"
                              aria-label="Sil"
                            >
                              <Trash2 className="h-4 w-4 text-red-500" />
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* EDIT DIALOG (kalem bunu açar) */}
      <Dialog
        open={showEditTask}
        onOpenChange={(open) => {
          setShowEditTask(open);
          if (!open) setEditingTask(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Görevi Düzenle</DialogTitle>
          </DialogHeader>

          <div className="space-y-4">
            <div>
              <Label htmlFor="edit-lesson">Ders</Label>
              <Input
                id="edit-lesson"
                value={editTaskDraft.lesson}
                onChange={(e) => setEditTaskDraft({ ...editTaskDraft, lesson: e.target.value })}
                placeholder="Örn: Matematik"
              />
            </div>

            <div>
              <Label htmlFor="edit-topic">Konu</Label>
              <Input
                id="edit-topic"
                value={editTaskDraft.topic}
                onChange={(e) => setEditTaskDraft({ ...editTaskDraft, topic: e.target.value })}
                placeholder="Örn: İntegral"
              />
            </div>

            <div>
              <Label htmlFor="edit-duration">Süre (dakika)</Label>
              <Input
                id="edit-duration"
                type="number"
                value={editTaskDraft.duration}
                onChange={(e) =>
                  setEditTaskDraft({ ...editTaskDraft, duration: parseInt(e.target.value) || 30 })
                }
              />
            </div>

            <div>
              <Label htmlFor="edit-day">Gün</Label>
              <select
                id="edit-day"
                className="w-full p-2 border-2 rounded-lg"
                value={editTaskDraft.day}
                onChange={(e) => setEditTaskDraft({ ...editTaskDraft, day: e.target.value })}
              >
                <option>Pazartesi</option>
                <option>Salı</option>
                <option>Çarşamba</option>
                <option>Perşembe</option>
                <option>Cuma</option>
                <option>Cumartesi</option>
                <option>Pazar</option>
              </select>
            </div>

            <div className="flex gap-2">
              <Button
                variant="outline"
                className="w-full"
                onClick={() => {
                  setShowEditTask(false);
                  setEditingTask(null);
                }}
              >
                İptal
              </Button>
              <Button className="w-full" onClick={saveEditedTask}>
                Kaydet
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
