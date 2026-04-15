import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import ThemeToggle from "@/components/ThemeToggle";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/contexts/AuthContext";
import { API } from "@/App";
import { Bell, Home, Search, UserMinus, UserPlus, Users } from "lucide-react";
import { formatDailyStudyHours, formatPublicHandle, getAvatarFallback, getPublicUsername } from "@/lib/publicProfile";

const getRelationshipLabel = (status) => {
  if (status === "outgoing_pending") return "İstek Gönderildi";
  if (status === "friends") return "Zaten Arkadaş";
  if (status === "incoming_pending") return "Sana İstek Gönderdi";
  return "Arkadaş Ekle";
};

export default function Friends() {
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  const [friends, setFriends] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [friendsLoading, setFriendsLoading] = useState(true);
  const [searchLoading, setSearchLoading] = useState(false);
  const [actionProfileId, setActionProfileId] = useState("");
  const [removingFriendProfileId, setRemovingFriendProfileId] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const authHeaders = useMemo(() => (
    currentUser?.uid ? { "X-Firebase-UID": currentUser.uid } : {}
  ), [currentUser]);

  const loadFriends = useCallback(async () => {
    if (!currentUser?.uid) {
      setFriendsLoading(false);
      return;
    }

    try {
      setFriendsLoading(true);
      const response = await axios.get(`${API}/friends`, { headers: authHeaders });
      setFriends(Array.isArray(response.data) ? response.data : []);
    } catch (loadError) {
      console.error("Error loading friends:", loadError);
      setError("Arkadaş listesi yüklenirken bir hata oluştu.");
    } finally {
      setFriendsLoading(false);
    }
  }, [authHeaders, currentUser]);

  useEffect(() => {
    loadFriends();
  }, [loadFriends]);

  const handleSearch = async (event) => {
    event.preventDefault();
    const trimmedQuery = searchQuery.trim();

    setError("");
    setSuccess("");

    if (!trimmedQuery) {
      setSearchResults([]);
      return;
    }

    try {
      setSearchLoading(true);
      const response = await axios.get(`${API}/users/search`, {
        params: { q: trimmedQuery },
        headers: authHeaders,
      });
      setSearchResults(Array.isArray(response.data) ? response.data : []);
    } catch (searchError) {
      console.error("Error searching users:", searchError);
      setError(searchError.response?.data?.detail || "Kullanıcı araması sırasında bir hata oluştu.");
    } finally {
      setSearchLoading(false);
    }
  };

  const handleSendFriendRequest = async (profileId) => {
    try {
      setActionProfileId(profileId);
      setError("");
      setSuccess("");
      await axios.post(
        `${API}/friends/requests`,
        { to_profile_id: profileId },
        { headers: authHeaders }
      );

      setSearchResults((previousResults) => previousResults.map((profile) => (
        profile.profile_id === profileId
          ? { ...profile, relationship_status: "outgoing_pending" }
          : profile
      )));
      setSuccess("Arkadaşlık isteği gönderildi.");
    } catch (requestError) {
      console.error("Error sending friend request:", requestError);
      setError(requestError.response?.data?.detail || "Arkadaşlık isteği gönderilemedi.");
    } finally {
      setActionProfileId("");
    }
  };

  const handleRemoveFriend = async (friend) => {
    if (!window.confirm("Bu kişiyi arkadaş listesinden çıkarmak istiyor musun?")) {
      return;
    }

    try {
      setRemovingFriendProfileId(friend.profile_id);
      setError("");
      setSuccess("");

      await axios.delete(`${API}/friends/${friend.profile_id}`, { headers: authHeaders });

      setFriends((previousFriends) => previousFriends.filter((item) => item.profile_id !== friend.profile_id));
      setSearchResults((previousResults) => previousResults.map((profile) => (
        profile.profile_id === friend.profile_id
          ? { ...profile, relationship_status: "none" }
          : profile
      )));
      setSuccess("Arkadaşlık kaldırıldı.");
    } catch (removeError) {
      console.error("Error removing friend:", removeError);
      setError(removeError.response?.data?.detail || "Arkadaşlık kaldırılırken bir hata oluştu.");
    } finally {
      setRemovingFriendProfileId("");
    }
  };

  return (
    <div className="min-h-screen bg-background p-6 md:p-10" data-testid="friends-page">
      <div className="mx-auto max-w-6xl space-y-6" data-testid="friends-page-container">
        <Card className="rounded-2xl border border-border/70 bg-card shadow-[0_18px_36px_-28px_rgba(15,23,42,0.16)]" data-testid="friends-header-card">
          <CardContent className="p-6 sm:p-8">
            <div className="flex flex-wrap items-start justify-between gap-4" data-testid="friends-header-content">
              <div>
                <h1 className="text-4xl font-bold tracking-tight text-slate-900 dark:text-slate-100" data-testid="friends-title">Arkadaşlar</h1>
                <p className="mt-2 text-base text-slate-600 dark:text-slate-300" data-testid="friends-subtitle">
                  Kullanıcı ara, arkadaşlık isteği gönder ve mevcut arkadaşlarını görüntüle.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-2" data-testid="friends-header-actions">
                <ThemeToggle className="h-10 w-10 rounded-xl border border-border/70 bg-background text-slate-700 hover:bg-secondary dark:text-slate-100" />
                <Button variant="outline" onClick={() => navigate("/notifications")} className="h-10 rounded-xl border-border/70 bg-background hover:bg-secondary" data-testid="friends-go-notifications-button">
                  <Bell className="mr-2 h-4 w-4" /> Gelen İstekler
                </Button>
                <Button variant="outline" onClick={() => navigate("/dashboard")} className="h-10 rounded-xl border-border/70 bg-background hover:bg-secondary" data-testid="friends-go-dashboard-button">
                  <Home className="mr-2 h-4 w-4" /> Dashboard
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {(error || success) && (
          <div
            className={`rounded-2xl border px-4 py-3 text-sm font-medium ${error ? "border-red-200 bg-red-50 text-red-700 dark:border-red-900/60 dark:bg-red-950/30 dark:text-red-300" : "border-green-200 bg-green-50 text-green-700 dark:border-green-900/60 dark:bg-green-950/30 dark:text-green-300"}`}
            data-testid={error ? "friends-error-message" : "friends-success-message"}
          >
            {error || success}
          </div>
        )}

        <Card className="rounded-2xl border border-border/70 bg-card shadow-[0_18px_36px_-28px_rgba(15,23,42,0.16)]" data-testid="friends-search-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl font-bold text-slate-900 dark:text-slate-100" data-testid="friends-search-title">
              <Search className="h-5 w-5 text-slate-800 dark:text-slate-200" /> Kullanıcı Ara
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            <form className="flex flex-col gap-3 sm:flex-row" onSubmit={handleSearch} data-testid="friends-search-form">
              <Input
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder="Handle veya kullanıcı adı ile ara"
                className="h-11 rounded-xl"
                data-testid="friends-search-input"
              />
              <Button type="submit" className="h-11 rounded-xl bg-primary text-primary-foreground hover:bg-slate-800" data-testid="friends-search-submit-button">
                {searchLoading ? "Aranıyor..." : "Ara"}
              </Button>
            </form>

            <div className="grid gap-4 md:grid-cols-2" data-testid="friends-search-results">
              {searchResults.map((profile) => {
                const handleText = formatPublicHandle(profile.handle_display || profile.handle);
                const relationshipLabel = getRelationshipLabel(profile.relationship_status);
                const isDisabled = profile.relationship_status && profile.relationship_status !== "none";

                return (
                  <div key={profile.profile_id} className="rounded-xl border border-border/70 bg-card p-4 shadow-sm" data-testid={`friends-search-result-${profile.profile_id}`}>
                    <div className="flex items-start gap-3" data-testid={`friends-search-result-content-${profile.profile_id}`}>
                      <div className="flex h-12 w-12 items-center justify-center overflow-hidden rounded-2xl bg-slate-900 text-sm font-bold text-white" data-testid={`friends-search-avatar-${profile.profile_id}`}>
                        {profile.avatar_url ? (
                          <img src={profile.avatar_url} alt={`${getPublicUsername(profile)} avatar`} className="h-full w-full object-cover" data-testid={`friends-search-avatar-image-${profile.profile_id}`} />
                        ) : (
                          getAvatarFallback(profile)
                        )}
                      </div>

                      <div className="min-w-0 flex-1 space-y-2" data-testid={`friends-search-details-${profile.profile_id}`}>
                        <div>
                          <p className="font-semibold text-slate-900 dark:text-slate-100" data-testid={`friends-search-username-${profile.profile_id}`}>{getPublicUsername(profile)}</p>
                          {handleText && (
                            <p className="text-sm text-slate-500 dark:text-slate-400" data-testid={`friends-search-handle-${profile.profile_id}`}>{handleText}</p>
                          )}
                        </div>
                        <p className="text-sm text-slate-600 dark:text-slate-300" data-testid={`friends-search-goal-${profile.profile_id}`}>Hedef: {profile.study_goal || "Belirtilmedi"}</p>
                        <p className="text-sm text-slate-600 dark:text-slate-300" data-testid={`friends-search-hours-${profile.profile_id}`}>Günlük Çalışma: {formatDailyStudyHours(profile.daily_study_hours)}</p>
                        <Button
                          onClick={() => handleSendFriendRequest(profile.profile_id)}
                          disabled={isDisabled || actionProfileId === profile.profile_id}
                          className="h-10 rounded-xl bg-primary text-primary-foreground hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
                          data-testid={`friends-search-action-${profile.profile_id}`}
                        >
                          <UserPlus className="mr-2 h-4 w-4" />
                          {actionProfileId === profile.profile_id ? "Gönderiliyor..." : relationshipLabel}
                        </Button>
                      </div>
                    </div>
                  </div>
                );
              })}

              {!searchLoading && searchQuery.trim() && searchResults.length === 0 && (
                <p className="rounded-2xl border border-dashed border-slate-300 px-4 py-6 text-sm text-slate-500 dark:border-slate-700 dark:text-slate-400" data-testid="friends-search-empty-state">
                  Aramana uygun kullanıcı bulunamadı.
                </p>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="rounded-2xl border border-border/70 bg-card shadow-[0_18px_36px_-28px_rgba(15,23,42,0.16)]" data-testid="friends-list-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl font-bold text-slate-900 dark:text-slate-100" data-testid="friends-list-title">
              <Users className="h-5 w-5 text-slate-800 dark:text-slate-200" /> Arkadaş Listen
            </CardTitle>
          </CardHeader>
          <CardContent>
            {friendsLoading ? (
              <p className="py-10 text-center text-slate-600 dark:text-slate-300" data-testid="friends-list-loading">Yükleniyor...</p>
            ) : friends.length === 0 ? (
              <p className="py-10 text-center text-slate-600 dark:text-slate-300" data-testid="friends-list-empty">Henüz arkadaşın bulunmuyor.</p>
            ) : (
              <div className="grid gap-4 md:grid-cols-2" data-testid="friends-list">
                {friends.map((friend) => {
                  const handleText = formatPublicHandle(friend.handle_display || friend.handle);
                  return (
                    <div key={friend.profile_id} className="rounded-xl border border-border/70 bg-card p-4 shadow-sm" data-testid={`friends-list-item-${friend.profile_id}`}>
                      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between" data-testid={`friends-list-item-content-${friend.profile_id}`}>
                        <div className="flex items-start gap-3">
                          <div className="flex h-12 w-12 items-center justify-center overflow-hidden rounded-2xl bg-slate-900 text-sm font-bold text-white" data-testid={`friends-list-avatar-${friend.profile_id}`}>
                            {friend.avatar_url ? (
                              <img src={friend.avatar_url} alt={`${getPublicUsername(friend)} avatar`} className="h-full w-full object-cover" data-testid={`friends-list-avatar-image-${friend.profile_id}`} />
                            ) : (
                              getAvatarFallback(friend)
                            )}
                          </div>

                          <div className="min-w-0 flex-1 space-y-1" data-testid={`friends-list-details-${friend.profile_id}`}>
                            <p className="font-semibold text-slate-900 dark:text-slate-100" data-testid={`friends-list-username-${friend.profile_id}`}>{getPublicUsername(friend)}</p>
                            {handleText && <p className="text-sm text-slate-500 dark:text-slate-400" data-testid={`friends-list-handle-${friend.profile_id}`}>{handleText}</p>}
                            <p className="text-sm text-slate-600 dark:text-slate-300" data-testid={`friends-list-goal-${friend.profile_id}`}>Hedef: {friend.study_goal || "Belirtilmedi"}</p>
                            <p className="text-sm text-slate-600 dark:text-slate-300" data-testid={`friends-list-hours-${friend.profile_id}`}>Günlük Çalışma: {formatDailyStudyHours(friend.daily_study_hours)}</p>
                          </div>
                        </div>

                        <Button
                          type="button"
                          variant="outline"
                          onClick={() => handleRemoveFriend(friend)}
                          disabled={removingFriendProfileId === friend.profile_id}
                          className="h-10 shrink-0 rounded-xl border-red-200 text-red-600 hover:bg-red-50 hover:text-red-700 dark:border-red-900/60 dark:text-red-300 dark:hover:bg-red-950/30"
                          data-testid={`friends-remove-action-${friend.profile_id}`}
                        >
                          <UserMinus className="mr-2 h-4 w-4" />
                          {removingFriendProfileId === friend.profile_id ? "Çıkarılıyor..." : "Arkadaşlıktan Çıkar"}
                        </Button>
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