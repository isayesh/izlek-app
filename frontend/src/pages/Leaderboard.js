import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { Button } from "@/components/ui/button";
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
      return "border-amber-200 bg-amber-50";
    }
    if (rank === 2) {
      return "border-slate-200 bg-slate-50";
    }
    if (rank === 3) {
      return "border-orange-200 bg-orange-50";
    }
    return "border-gray-200 bg-white hover:bg-gray-50";
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
    <div className="min-h-screen bg-gray-50 px-4 py-5 sm:px-6 sm:py-6 lg:px-8" data-testid="leaderboard-page">
      <div className="mx-auto w-full max-w-6xl space-y-4" data-testid="leaderboard-container">
        <div className="rounded-2xl border border-gray-200 bg-white px-5 py-4 shadow-sm sm:px-6 sm:py-5" data-testid="leaderboard-header-card">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="space-y-3 min-w-0">
              <div>
                <h1 className="text-xl font-semibold tracking-tight text-gray-900 sm:text-2xl" data-testid="leaderboard-title">
                  Liderlik Tablosu
                </h1>
                <p className="mt-0.5 text-sm text-gray-500" data-testid="leaderboard-subtitle">
                  {periodConfig.subtitle}
                </p>
              </div>

              <div className="inline-flex flex-wrap items-center gap-1 rounded-full border border-gray-200 bg-gray-50 p-1" data-testid="leaderboard-period-tabs">
                {LEADERBOARD_PERIODS.map((period) => {
                  const isActive = period.key === activePeriod;

                  return (
                    <Button
                      key={period.key}
                      type="button"
                      variant={isActive ? "default" : "ghost"}
                      onClick={() => setActivePeriod(period.key)}
                      className={`h-8 rounded-full px-3.5 text-sm font-medium transition-colors ${isActive ? "bg-indigo-600 text-white shadow-sm hover:bg-indigo-700" : "bg-transparent text-gray-600 hover:bg-white hover:text-gray-900"}`}
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
                className="h-9 rounded-lg border-gray-200 bg-white px-3 text-sm text-gray-700 hover:bg-gray-50 hover:text-gray-900"
                data-testid="btn-rooms"
              >
                <Users className="mr-1.5 h-4 w-4" />
                Odalar
              </Button>
              <Button
                variant="outline"
                onClick={() => navigate("/dashboard")}
                className="h-9 rounded-lg border-gray-200 bg-white px-3 text-sm text-gray-700 hover:bg-gray-50 hover:text-gray-900"
                data-testid="btn-dashboard"
              >
                <Home className="mr-1.5 h-4 w-4" />
                Dashboard
              </Button>
            </div>
          </div>
        </div>

        {myEntry && (
          <div className="rounded-2xl border border-gray-200 bg-white px-4 py-3 shadow-sm sm:px-5 sm:py-3.5" data-testid="my-rank-card">
            <div className="flex flex-wrap items-center justify-between gap-4" data-testid="my-rank-content">
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-indigo-50 text-indigo-600">
                  <Trophy className="h-4 w-4" />
                </div>
                <p className="text-sm font-semibold text-gray-900" data-testid="my-rank-title">
                  Senin Sıralaman
                </p>
              </div>
              <div className="flex items-center gap-6 sm:gap-10" data-testid="my-rank-metrics">
                <div className="text-right" data-testid="my-rank-value-wrap">
                  <p className="text-[11px] uppercase tracking-wide text-gray-500" data-testid="my-rank-label">Sıra</p>
                  <p className="text-lg font-bold text-indigo-600 sm:text-xl" data-testid="my-rank-value">#{myEntry.rank}</p>
                </div>
                <div className="text-right" data-testid="my-time-value-wrap">
                  <p className="text-[11px] uppercase tracking-wide text-gray-500" data-testid="my-time-label">{periodConfig.metricLabel}</p>
                  <p className="text-lg font-semibold text-gray-900 sm:text-xl" data-testid="my-time-value">
                    {formatStudyTime(myEntry.total_seconds)}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="rounded-2xl border border-gray-200 bg-white shadow-sm" data-testid="leaderboard-list-card">
          <div className="flex items-center justify-between border-b border-gray-200 px-5 py-3 sm:px-6">
            <div className="flex items-center gap-2 text-sm font-semibold text-gray-900 sm:text-base" data-testid="leaderboard-list-title">
              <Trophy className="h-4 w-4 text-indigo-600 sm:h-5 sm:w-5" />
              {periodConfig.label} Sıralama
            </div>
            {!loading && !error && entries.length > 0 && (
              <span className="text-xs font-medium text-gray-400">
                {entries.length} katılımcı
              </span>
            )}
          </div>

          <div className="p-3 sm:p-4">
            {loading ? (
              <p className="py-10 text-center text-sm text-gray-500" data-testid="leaderboard-loading-text">
                Yükleniyor...
              </p>
            ) : error ? (
              <p className="py-10 text-center text-sm text-red-600" data-testid="leaderboard-error-text">
                {error}
              </p>
            ) : entries.length === 0 ? (
              <p className="py-10 text-center text-sm text-gray-500" data-testid="leaderboard-empty-text">
                Bu dönem için henüz leaderboard verisi bulunmuyor.
              </p>
            ) : (
              <div className="space-y-1.5" data-testid="leaderboard-list">
                {entries.map((entry) => {
                  const displayName = getEntryDisplayName(entry);
                  const avatarUrl = getEntryAvatarUrl(entry);
                  const medal = getRankMedal(entry.rank);
                  const isTopOne = entry.rank === 1;
                  const isTopThree = entry.rank <= 3;
                  const rowSpacing = isTopOne
                    ? "px-4 py-4 sm:px-5 sm:py-5 shadow-sm"
                    : "px-3 py-2.5 sm:px-4 sm:py-3";
                  const avatarSize = isTopOne ? "h-11 w-11 sm:h-12 sm:w-12" : "h-9 w-9 sm:h-10 sm:w-10";
                  const rankSize = isTopOne ? "text-lg sm:text-xl" : "text-base sm:text-lg";
                  const nameSize = isTopOne ? "text-base sm:text-[17px]" : "text-sm sm:text-base";
                  const timeSize = isTopOne ? "text-base sm:text-lg" : "text-sm sm:text-base";
                  const avatarTone = isTopOne
                    ? "bg-amber-100 text-amber-700"
                    : isTopThree
                      ? "bg-slate-100 text-slate-700"
                      : "bg-indigo-100 text-indigo-700";

                  return (
                    <div
                      key={entry.user_id}
                      className={`grid grid-cols-[56px_1fr_auto] items-center gap-3 rounded-xl border transition-colors duration-150 sm:grid-cols-[68px_1fr_auto] ${rowSpacing} ${getRowClassName(entry.rank)}`}
                      data-testid={`leaderboard-row-${entry.rank}`}
                    >
                      <div className="flex items-center gap-1.5 text-gray-700" data-testid={`leaderboard-rank-${entry.rank}`}>
                        {medal && <span className="leading-none sm:text-lg" data-testid={`leaderboard-medal-${entry.rank}`}>{medal}</span>}
                        <span className={`font-bold text-gray-900 ${rankSize}`}>#{entry.rank}</span>
                      </div>
                      <div className="min-w-0 flex items-center gap-3">
                        <div className={`flex flex-shrink-0 items-center justify-center overflow-hidden rounded-full text-sm font-bold ${avatarSize} ${avatarTone}`} data-testid={`leaderboard-avatar-${entry.rank}`}>
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
                          <p className={`truncate font-semibold text-gray-900 ${nameSize}`} data-testid={`leaderboard-username-${entry.rank}`}>
                            {displayName}
                          </p>
                          <p className="truncate text-xs text-gray-500" data-testid={`leaderboard-secondary-text-${entry.rank}`}>
                            {isTopOne ? "Lider" : isTopThree ? "Podyumda" : "Çalışma süresi bazlı sıralama"}
                          </p>
                        </div>
                      </div>
                      <div className="text-right" data-testid={`leaderboard-time-wrap-${entry.rank}`}>
                        <p className={`whitespace-nowrap font-semibold text-gray-900 ${timeSize}`} data-testid={`leaderboard-time-${entry.rank}`}>
                          {formatStudyTime(entry.total_seconds)}
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
