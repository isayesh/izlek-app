import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { Button } from "@/components/ui/button";
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
    <div className="min-h-screen bg-gray-50 px-4 py-5 sm:px-6 sm:py-6 lg:px-8" data-testid="friends-page">
      <div className="mx-auto w-full max-w-6xl space-y-4" data-testid="friends-page-container">
        <div className="rounded-2xl border border-gray-200 bg-white px-5 py-4 shadow-sm sm:px-6 sm:py-5" data-testid="friends-header-card">
          <div className="flex flex-wrap items-start justify-between gap-4" data-testid="friends-header-content">
            <div className="min-w-0">
              <h1 className="text-xl font-semibold tracking-tight text-gray-900 sm:text-2xl" data-testid="friends-title">Arkadaşlar</h1>
              <p className="mt-0.5 text-sm text-gray-500" data-testid="friends-subtitle">
                Kullanıcı ara, arkadaşlık isteği gönder ve mevcut arkadaşlarını görüntüle.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2" data-testid="friends-header-actions">
              <Button variant="outline" onClick={() => navigate("/notifications")} className="h-9 rounded-lg border-gray-200 bg-white px-3 text-sm text-gray-700 hover:bg-gray-50 hover:text-gray-900" data-testid="friends-go-notifications-button">
                <Bell className="mr-1.5 h-4 w-4" /> Gelen İstekler
              </Button>
              <Button variant="outline" onClick={() => navigate("/dashboard")} className="h-9 rounded-lg border-gray-200 bg-white px-3 text-sm text-gray-700 hover:bg-gray-50 hover:text-gray-900" data-testid="friends-go-dashboard-button">
                <Home className="mr-1.5 h-4 w-4" /> Dashboard
              </Button>
            </div>
          </div>
        </div>

        {(error || success) && (
          <div
            className={`rounded-xl border px-4 py-2.5 text-sm font-medium ${error ? "border-red-200 bg-red-50 text-red-700" : "border-emerald-200 bg-emerald-50 text-emerald-700"}`}
            data-testid={error ? "friends-error-message" : "friends-success-message"}
          >
            {error || success}
          </div>
        )}

        <div className="rounded-2xl border border-gray-200 bg-white shadow-sm" data-testid="friends-search-card">
          <div className="flex items-center gap-2 border-b border-gray-200 px-5 py-3 sm:px-6">
            <Search className="h-4 w-4 text-indigo-600 sm:h-5 sm:w-5" />
            <h2 className="text-sm font-semibold text-gray-900 sm:text-base" data-testid="friends-search-title">Kullanıcı Ara</h2>
          </div>
          <div className="space-y-4 p-4 sm:p-5">
            <form className="flex flex-col gap-2 sm:flex-row" onSubmit={handleSearch} data-testid="friends-search-form">
              <Input
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder="Handle veya kullanıcı adı ile ara"
                className="h-10 rounded-lg border-gray-200 bg-white text-sm placeholder:text-gray-400 focus-visible:border-indigo-500 focus-visible:ring-2 focus-visible:ring-indigo-100"
                data-testid="friends-search-input"
              />
              <Button type="submit" className="h-10 rounded-lg bg-indigo-600 px-5 text-sm font-medium text-white hover:bg-indigo-700 disabled:bg-indigo-300" data-testid="friends-search-submit-button">
                <Search className="mr-1.5 h-4 w-4 sm:hidden" />
                {searchLoading ? "Aranıyor..." : "Ara"}
              </Button>
            </form>

            {searchResults.length > 0 && (
              <div className="grid gap-3 md:grid-cols-2" data-testid="friends-search-results">
                {searchResults.map((profile) => {
                  const handleText = formatPublicHandle(profile.handle_display || profile.handle);
                  const relationshipLabel = getRelationshipLabel(profile.relationship_status);
                  const isDisabled = profile.relationship_status && profile.relationship_status !== "none";

                  return (
                    <div key={profile.profile_id} className="rounded-xl border border-gray-200 bg-white p-3.5 transition-colors hover:bg-gray-50" data-testid={`friends-search-result-${profile.profile_id}`}>
                      <div className="flex items-start gap-3" data-testid={`friends-search-result-content-${profile.profile_id}`}>
                        <div className="flex h-11 w-11 flex-shrink-0 items-center justify-center overflow-hidden rounded-full bg-indigo-100 text-sm font-bold text-indigo-700" data-testid={`friends-search-avatar-${profile.profile_id}`}>
                          {profile.avatar_url ? (
                            <img src={profile.avatar_url} alt={`${getPublicUsername(profile)} avatar`} className="h-full w-full object-cover" data-testid={`friends-search-avatar-image-${profile.profile_id}`} />
                          ) : (
                            getAvatarFallback(profile)
                          )}
                        </div>

                        <div className="min-w-0 flex-1 space-y-1.5" data-testid={`friends-search-details-${profile.profile_id}`}>
                          <div className="min-w-0">
                            <p className="truncate text-sm font-semibold text-gray-900 sm:text-base" data-testid={`friends-search-username-${profile.profile_id}`}>{getPublicUsername(profile)}</p>
                            {handleText && (
                              <p className="truncate text-xs text-gray-500" data-testid={`friends-search-handle-${profile.profile_id}`}>{handleText}</p>
                            )}
                          </div>
                          <p className="text-xs text-gray-600" data-testid={`friends-search-goal-${profile.profile_id}`}>Hedef: {profile.study_goal || "Belirtilmedi"}</p>
                          <p className="text-xs text-gray-600" data-testid={`friends-search-hours-${profile.profile_id}`}>Günlük: {formatDailyStudyHours(profile.daily_study_hours)}</p>
                          <Button
                            onClick={() => handleSendFriendRequest(profile.profile_id)}
                            disabled={isDisabled || actionProfileId === profile.profile_id}
                            className="mt-1 h-9 rounded-lg bg-indigo-600 text-sm font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-gray-200 disabled:text-gray-500"
                            data-testid={`friends-search-action-${profile.profile_id}`}
                          >
                            <UserPlus className="mr-1.5 h-4 w-4" />
                            {actionProfileId === profile.profile_id ? "Gönderiliyor..." : relationshipLabel}
                          </Button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {!searchLoading && searchQuery.trim() && searchResults.length === 0 && (
              <div className="rounded-xl border border-dashed border-gray-300 bg-gray-50 px-4 py-8 text-center" data-testid="friends-search-empty-state">
                <p className="text-sm text-gray-500">Aramana uygun kullanıcı bulunamadı.</p>
              </div>
            )}
          </div>
        </div>

        <div className="rounded-2xl border border-gray-200 bg-white shadow-sm" data-testid="friends-list-card">
          <div className="flex items-center justify-between border-b border-gray-200 px-5 py-3 sm:px-6">
            <div className="flex items-center gap-2 text-sm font-semibold text-gray-900 sm:text-base" data-testid="friends-list-title">
              <Users className="h-4 w-4 text-indigo-600 sm:h-5 sm:w-5" /> Arkadaş Listen
            </div>
            {!friendsLoading && friends.length > 0 && (
              <span className="text-xs font-medium text-gray-400">{friends.length} arkadaş</span>
            )}
          </div>
          <div className="p-4 sm:p-5">
            {friendsLoading ? (
              <p className="py-10 text-center text-sm text-gray-500" data-testid="friends-list-loading">Yükleniyor...</p>
            ) : friends.length === 0 ? (
              <div className="flex flex-col items-center justify-center gap-3 px-4 py-10 text-center" data-testid="friends-list-empty">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-indigo-50 text-indigo-600">
                  <Users className="h-6 w-6" />
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-semibold text-gray-900 sm:text-base">Henüz arkadaşın yok</p>
                  <p className="text-sm text-gray-500">Arkadaş ekleyerek birlikte çalışmaya başlayabilirsin.</p>
                </div>
                <Button
                  type="button"
                  onClick={() => {
                    const input = document.querySelector('[data-testid="friends-search-input"]');
                    const card = document.querySelector('[data-testid="friends-search-card"]');
                    if (card && card.scrollIntoView) {
                      card.scrollIntoView({ behavior: "smooth", block: "start" });
                    }
                    if (input && input.focus) {
                      setTimeout(() => input.focus(), 200);
                    }
                  }}
                  className="mt-1 h-9 rounded-lg bg-indigo-600 px-4 text-sm font-medium text-white hover:bg-indigo-700"
                  data-testid="friends-empty-cta"
                >
                  <UserPlus className="mr-1.5 h-4 w-4" /> Arkadaş ara
                </Button>
              </div>
            ) : (
              <div className="grid gap-3 md:grid-cols-2" data-testid="friends-list">
                {friends.map((friend) => {
                  const handleText = formatPublicHandle(friend.handle_display || friend.handle);
                  return (
                    <div key={friend.profile_id} className="rounded-xl border border-gray-200 bg-white p-3.5 transition-colors hover:bg-gray-50" data-testid={`friends-list-item-${friend.profile_id}`}>
                      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between" data-testid={`friends-list-item-content-${friend.profile_id}`}>
                        <div className="flex min-w-0 items-start gap-3">
                          <div className="flex h-11 w-11 flex-shrink-0 items-center justify-center overflow-hidden rounded-full bg-indigo-100 text-sm font-bold text-indigo-700" data-testid={`friends-list-avatar-${friend.profile_id}`}>
                            {friend.avatar_url ? (
                              <img src={friend.avatar_url} alt={`${getPublicUsername(friend)} avatar`} className="h-full w-full object-cover" data-testid={`friends-list-avatar-image-${friend.profile_id}`} />
                            ) : (
                              getAvatarFallback(friend)
                            )}
                          </div>

                          <div className="min-w-0 flex-1 space-y-0.5" data-testid={`friends-list-details-${friend.profile_id}`}>
                            <p className="truncate text-sm font-semibold text-gray-900 sm:text-base" data-testid={`friends-list-username-${friend.profile_id}`}>{getPublicUsername(friend)}</p>
                            {handleText && <p className="truncate text-xs text-gray-500" data-testid={`friends-list-handle-${friend.profile_id}`}>{handleText}</p>}
                            <p className="text-xs text-gray-600" data-testid={`friends-list-goal-${friend.profile_id}`}>Hedef: {friend.study_goal || "Belirtilmedi"}</p>
                            <p className="text-xs text-gray-600" data-testid={`friends-list-hours-${friend.profile_id}`}>Günlük: {formatDailyStudyHours(friend.daily_study_hours)}</p>
                          </div>
                        </div>

                        <Button
                          type="button"
                          variant="outline"
                          onClick={() => handleRemoveFriend(friend)}
                          disabled={removingFriendProfileId === friend.profile_id}
                          className="h-9 shrink-0 rounded-lg border-gray-200 bg-white text-xs font-medium text-gray-600 hover:border-red-200 hover:bg-red-50 hover:text-red-600 sm:text-sm"
                          data-testid={`friends-remove-action-${friend.profile_id}`}
                        >
                          <UserMinus className="mr-1.5 h-4 w-4" />
                          {removingFriendProfileId === friend.profile_id ? "Çıkarılıyor..." : "Çıkar"}
                        </Button>
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