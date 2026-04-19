import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { API } from "@/App";
import { Home, Users, Trophy } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";

const LEADERBOARD_PERIODS = [
  { key: "daily", label: "Günlük", subtitle: "Bugünkü çalışma sürelerine göre sıralama", metricLabel: "Bugün" },
  { key: "weekly", label: "Haftalık", subtitle: "Bu haftaki çalışma sürelerine göre sıralama", metricLabel: "Bu hafta" },
  { key: "all", label: "Genel", subtitle: "Toplam çalışma sürelerine göre global sıralama", metricLabel: "Toplam" },
];

export default function Leaderboard() {
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  const [loading, setLoading] = useState(true);
  const [entries, setEntries] = useState([]);
  const [error, setError] = useState("");
  const [failedAvatars, setFailedAvatars] = useState({});
  const [activePeriod, setActivePeriod] = useState("all");
  const currentUserId = currentUser?.uid || localStorage.getItem("userId") || localStorage.getItem("currentUserId");
  const periodConfig = LEADERBOARD_PERIODS.find((period) => period.key === activePeriod) || LEADERBOARD_PERIODS[2];

  useEffect(() => {
    loadLeaderboard(activePeriod);
  }, [activePeriod]);

  const loadLeaderboard = async (period = activePeriod) => {
    try {
      setLoading(true);
      setError("");
      const res = await axios.get(`${API}/leaderboard`, { params: { period } });
      setEntries(Array.isArray(res.data) ? res.data : []);
    } catch (err) {
      console.error("Error loading leaderboard:", err);
      setError("Liderlik tablosu yüklenirken bir hata oluştu.");
    } finally {
      setLoading(false);
    }
  };

  const formatStudyTime = (totalSeconds) => {
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    return `${hours} saat ${minutes} dk`;
  };

  const getRankMedal = (rank) => {
    if (rank === 1) return "🥇";
    if (rank === 2) return "🥈";
    if (rank === 3) return "🥉";
    return null;
  };

  const getRowClassName = (rank) => {
    if (rank === 1) {
      return "border-amber-200/70 bg-amber-50/55  ";
    }
    if (rank === 2) {
      return "border-slate-300/80 bg-slate-50  ";
    }
    if (rank === 3) {
      return "border-orange-200/70 bg-orange-50/55  ";
    }
    return "border-border/70 bg-card";
  };

  const getEntryDisplayName = (entry) => {
    const backendName = entry.user_name?.trim();
    const currentUserFallback = currentUser?.displayName?.trim() || "Bilinmeyen Kullanıcı";

    if (entry.user_id === currentUserId && (!backendName || backendName === "Bilinmeyen Kullanıcı")) {
      return currentUserFallback;
    }

    return backendName || "Bilinmeyen Kullanıcı";
  };

  const getEntryAvatarUrl = (entry) => (failedAvatars[entry.user_id] ? "" : entry.avatar_url || "");
  const getInitial = (name) => (name || "İ").trim().charAt(0).toUpperCase();
  const myEntry = currentUserId ? entries.find((entry) => entry.user_id === currentUserId) : null;

  return (
    <div className="min-h-screen bg-background p-6 md:p-10" data-testid="leaderboard-page">
      <div className="mx-auto max-w-5xl space-y-6" data-testid="leaderboard-container">
        <Card className="rounded-2xl border border-border/70 bg-card shadow-[0_18px_36px_-28px_rgba(15,23,42,0.16)]" data-testid="leaderboard-header-card">
          <CardContent className="p-6 sm:p-8">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="space-y-4">
                <div>
                  <h1 className="text-4xl font-bold tracking-tight text-slate-900  sm:text-5xl lg:text-6xl" data-testid="leaderboard-title">
                    Liderlik Tablosu
                  </h1>
                  <p className="mt-2 text-base text-slate-600  md:text-lg" data-testid="leaderboard-subtitle">
                    {periodConfig.subtitle}
                  </p>
                </div>

                <div className="inline-flex flex-wrap items-center gap-2 rounded-2xl border border-border/70 bg-background/70 p-1" data-testid="leaderboard-period-tabs">
                  {LEADERBOARD_PERIODS.map((period) => {
                    const isActive = period.key === activePeriod;

                    return (
                      <Button
                        key={period.key}
                        type="button"
                        variant={isActive ? "default" : "ghost"}
                        onClick={() => setActivePeriod(period.key)}
                        className={`h-10 rounded-xl px-4 text-sm font-semibold ${isActive ? "bg-slate-900 text-white shadow-sm hover:bg-slate-800   " : "text-slate-600 hover:bg-secondary  "}`}
                        data-testid={`leaderboard-period-${period.key}`}
                      >
                        {period.label}
                      </Button>
                    );
                  })}
                </div>
              </div>

              <div className="flex items-center gap-2" data-testid="leaderboard-header-actions">
                <Button
                  variant="outline"
                  onClick={() => navigate("/rooms")}
                  className="h-10 rounded-xl border-border/70 bg-background text-slate-700 hover:bg-secondary "
                  data-testid="btn-rooms"
                >
                  <Users className="mr-2 h-4 w-4" />
                  Odalar
                </Button>
                <Button
                  variant="outline"
                  onClick={() => navigate("/dashboard")}
                  className="h-10 rounded-xl border-border/70 bg-background text-slate-700 hover:bg-secondary "
                  data-testid="btn-dashboard"
                >
                  <Home className="mr-2 h-4 w-4" />
                  Dashboard
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {myEntry && (
          <Card className="rounded-2xl border border-border/70 bg-card shadow-[0_18px_36px_-28px_rgba(15,23,42,0.16)]" data-testid="my-rank-card">
            <CardContent className="p-4 sm:p-5">
              <div className="flex flex-wrap items-center justify-between gap-4" data-testid="my-rank-content">
                <p className="text-sm font-semibold text-slate-600 " data-testid="my-rank-title">
                  Senin Sıralaman
                </p>
                <div className="flex items-center gap-8" data-testid="my-rank-metrics">
                  <div className="text-right" data-testid="my-rank-value-wrap">
                    <p className="text-xs text-slate-500 " data-testid="my-rank-label">Sıra</p>
                    <p className="text-lg font-bold text-slate-900 " data-testid="my-rank-value">#{myEntry.rank}</p>
                  </div>
                  <div className="text-right" data-testid="my-time-value-wrap">
                    <p className="text-xs text-slate-500 " data-testid="my-time-label">{periodConfig.metricLabel}</p>
                    <p className="text-lg font-semibold text-slate-900 " data-testid="my-time-value">
                      {formatStudyTime(myEntry.total_seconds)}
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        <Card className="rounded-2xl border border-border/70 bg-card shadow-[0_18px_36px_-28px_rgba(15,23,42,0.16)]" data-testid="leaderboard-list-card">
          <CardHeader className="border-b border-border/70 pb-4">
            <CardTitle className="flex items-center gap-2 text-xl font-bold text-slate-900 " data-testid="leaderboard-list-title">
              <Trophy className="h-5 w-5 text-slate-800 " />
              {periodConfig.label} Sıralama
            </CardTitle>
          </CardHeader>

          <CardContent className="p-5 sm:p-6">
            {loading ? (
              <p className="py-10 text-center text-slate-600 " data-testid="leaderboard-loading-text">
                Yükleniyor...
              </p>
            ) : error ? (
              <p className="py-10 text-center text-red-600 " data-testid="leaderboard-error-text">
                {error}
              </p>
            ) : entries.length === 0 ? (
              <p className="py-10 text-center text-slate-600 " data-testid="leaderboard-empty-text">
                Bu dönem için henüz leaderboard verisi bulunmuyor.
              </p>
            ) : (
              <div className="space-y-3" data-testid="leaderboard-list">
                {entries.map((entry) => {
                  const displayName = getEntryDisplayName(entry);
                  const avatarUrl = getEntryAvatarUrl(entry);
                  const medal = getRankMedal(entry.rank);

                  return (
                    <div
                      key={entry.user_id}
                      className={`grid grid-cols-[84px_1fr_auto] items-center gap-3 rounded-xl border p-4 shadow-sm transition-[border-color,box-shadow] duration-200 hover:shadow-[0_16px_30px_-24px_rgba(15,23,42,0.16)] ${getRowClassName(entry.rank)}`}
                      data-testid={`leaderboard-row-${entry.rank}`}
                    >
                      <div className="flex items-center gap-2 text-slate-800 " data-testid={`leaderboard-rank-${entry.rank}`}>
                        {medal && <span className="text-xl leading-none" data-testid={`leaderboard-medal-${entry.rank}`}>{medal}</span>}
                        <span className="text-2xl font-bold">#{entry.rank}</span>
                      </div>
                      <div className="min-w-0 flex items-center gap-3">
                        <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center overflow-hidden rounded-full bg-slate-900 text-sm font-bold text-white shadow-sm" data-testid={`leaderboard-avatar-${entry.rank}`}>
                          {avatarUrl ? (
                            <img
                              src={avatarUrl}
                              alt={`${displayName} avatar`}
                              className="h-full w-full object-cover"
                              data-testid={`leaderboard-avatar-image-${entry.rank}`}
                              onError={() => setFailedAvatars((prev) => ({ ...prev, [entry.user_id]: true }))}
                            />
                          ) : (
                            getInitial(displayName)
                          )}
                        </div>
                        <div className="min-w-0">
                          <p className="truncate font-semibold text-slate-900 " data-testid={`leaderboard-username-${entry.rank}`}>
                            {displayName}
                          </p>
                          <p className="truncate text-xs text-slate-500 " data-testid={`leaderboard-secondary-text-${entry.rank}`}>
                            Çalışma süresi bazlı sıralama
                          </p>
                        </div>
                      </div>
                      <div className="text-right" data-testid={`leaderboard-time-wrap-${entry.rank}`}>
                        <p className="font-semibold text-slate-900 " data-testid={`leaderboard-time-${entry.rank}`}>
                          {formatStudyTime(entry.total_seconds)}
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
