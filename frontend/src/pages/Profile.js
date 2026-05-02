import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import axios from "axios";
import {
  BarChart3,
  Home,
  MessageCircle,
  MessageSquare,
  Trophy,
  User,
  Users,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useAuth } from "@/contexts/AuthContext";
import { API } from "@/App";
import {
  getForumCurrentUserComments,
  getForumCurrentUserLikedPosts,
  getForumCurrentUserPosts,
  getForumUserProfile,
  getForumUserStats,
  getForumUsersByUsernames,
  isForumFollowing,
  subscribeForumFeedStore,
  subscribeForumFollowStore,
  toggleForumFollow,
} from "@/pages/forumSocialStore";

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

const getProfileMetaLine = (gradeLevel, studyField) => {
  const parts = [GRADE_LEVEL_LABELS[gradeLevel] || gradeLevel, STUDY_FIELD_LABELS[studyField] || studyField].filter(Boolean);
  return parts.length ? parts.join(" • ") : "";
};

const getInitials = (name = "") =>
  name
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part.charAt(0).toUpperCase())
    .join("") || "İ";

const getRelativeTimeLabel = (isoString) => {
  const diffMs = Date.now() - new Date(isoString).getTime();
  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;

  if (diffMs < hour) {
    const minutes = Math.max(1, Math.round(diffMs / minute));
    return `${minutes} dk`;
  }

  if (diffMs < day) {
    const hours = Math.max(1, Math.round(diffMs / hour));
    return `${hours} sa`;
  }

  const days = Math.max(1, Math.round(diffMs / day));
  return `${days} g`;
};

const getDefaultProfileData = (currentUser) => ({
  username: localStorage.getItem("userName") || currentUser?.displayName || "",
  handle: "",
  avatar_url: "",
  grade_level: "",
  study_field: "",
});

const mapProfileData = (profile, currentUser) => ({
  username: profile?.username || profile?.name || localStorage.getItem("userName") || currentUser?.displayName || "",
  handle: profile?.handle || "",
  avatar_url: profile?.avatar_url || "",
  grade_level: profile?.grade_level || "",
  study_field: profile?.study_field || "",
});

export default function Profile() {
  const navigate = useNavigate();
  const location = useLocation();
  const { currentUser } = useAuth();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [profileData, setProfileData] = useState(getDefaultProfileData(currentUser));
  const [listModalState, setListModalState] = useState({ open: false, type: "followers" });
  const [, setFollowStoreVersion] = useState(0);
  const [feedStoreVersion, setFeedStoreVersion] = useState(0);

  useEffect(() => {
    const unsubscribeFeed = subscribeForumFeedStore(() => {
      setFeedStoreVersion((prev) => prev + 1);
    });

    return unsubscribeFeed;
  }, []);

  useEffect(() => {
    const unsubscribe = subscribeForumFollowStore(() => {
      setFollowStoreVersion((prev) => prev + 1);
    });

    return unsubscribe;
  }, []);

  useEffect(() => {
    const loadProfile = async () => {
      if (!currentUser?.uid) {
        setLoading(false);
        return;
      }

      setLoading(true);
      setError("");

      try {
        const response = await axios.get(`${API}/profile`, {
          params: { firebase_uid: currentUser.uid },
        });

        const profile = response.data;
        if (profile?.id) {
          setProfileData(mapProfileData(profile, currentUser));
        } else {
          setProfileData(getDefaultProfileData(currentUser));
        }
      } catch (loadError) {
        console.error("Error loading profile:", loadError);
        setError("Profil bilgileri yüklenirken bir hata oluştu.");
        setProfileData(getDefaultProfileData(currentUser));
      } finally {
        setLoading(false);
      }
    };

    loadProfile();
  }, [currentUser]);

  const displayName = profileData.username || currentUser?.displayName || "İzlek Kullanıcısı";
  const username = profileData.handle?.trim() || "sen";
  const profileMetaLine = getProfileMetaLine(profileData.grade_level, profileData.study_field);

  const socialProfile = useMemo(() => getForumUserProfile("sen", displayName), [displayName]);
  const profileStats = getForumUserStats("sen");
  const ownPosts = useMemo(() => getForumCurrentUserPosts("sen"), [feedStoreVersion]);
  const ownComments = useMemo(() => getForumCurrentUserComments("sen"), [feedStoreVersion]);
  const likedPosts = useMemo(() => getForumCurrentUserLikedPosts(), [feedStoreVersion]);

  const listUsers = useMemo(() => {
    const sourceList = listModalState.type === "followers" ? socialProfile.followersList : socialProfile.followingList;
    return getForumUsersByUsernames(sourceList);
  }, [listModalState.type, socialProfile.followersList, socialProfile.followingList]);

  const navigationActions = [
    { label: "Ana Sayfa", icon: Home, onClick: () => navigate("/") },
    { label: "Net Takibi", icon: BarChart3, onClick: () => navigate("/net-tracking") },
    { label: "Odalar", icon: Users, onClick: () => navigate("/rooms") },
    { label: "Liderlik", icon: Trophy, onClick: () => navigate("/leaderboard") },
    { label: "Forum", icon: MessageCircle, onClick: () => navigate("/forum") },
    { label: "Profil", icon: User, onClick: () => navigate("/profile") },
    { label: "DM Kutum", icon: MessageSquare, onClick: () => navigate("/messages") },
    { label: "Arkadaşlar", icon: Users, onClick: () => navigate("/friends") },
  ];

  const navActivePaths = {
    "Ana Sayfa": ["/", "/dashboard"],
    "Net Takibi": ["/net-tracking"],
    Odalar: ["/rooms"],
    Liderlik: ["/leaderboard"],
    Forum: ["/forum"],
    Profil: ["/profile", "/profile/edit"],
    "DM Kutum": ["/messages"],
    Arkadaşlar: ["/friends"],
  };

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center text-gray-900" data-testid="profile-loading-state">
        Profil yükleniyor...
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground" data-testid="profile-page">
      <div className="px-4 pb-10 pt-4 sm:px-6 sm:pb-12 lg:px-10 xl:px-12">
        <div className="space-y-8 sm:space-y-10">
          <header className="flex flex-col gap-3 border-b border-indigo-200 bg-indigo-50/35 pb-4 xl:flex-row xl:items-center xl:gap-4">
            <button
              type="button"
              onClick={() => navigate("/dashboard")}
              className="inline-flex h-12 w-fit shrink-0 items-center px-1"
              data-testid="profile-header-brand"
              aria-label="Dashboard"
            >
              <span className="font-display text-4xl font-extrabold leading-none tracking-[-0.03em] text-gray-900">izlek</span>
            </button>

            <div className="w-full max-w-full flex-1 overflow-x-auto pb-1 xl:pb-0">
              <div className="flex items-center gap-2.5 lg:gap-3 xl:justify-center xl:gap-3.5">
                {navigationActions.map((action) => {
                  const Icon = action.icon;
                  const isActive = (navActivePaths[action.label] || []).includes(location.pathname);

                  return (
                    <Button
                      key={action.label}
                      variant="ghost"
                      size="sm"
                      onClick={action.onClick}
                      className={`h-11 shrink-0 rounded-[15px] border px-5 text-sm font-semibold tracking-[0.01em] shadow-none transition-colors duration-200 [&_svg]:size-4 ${
                        isActive
                          ? "border-indigo-200 bg-indigo-100/75 text-indigo-700"
                          : "border-transparent text-slate-600 hover:border-indigo-200/90 hover:bg-indigo-100/70 hover:text-indigo-700 active:bg-indigo-100/80"
                      }`}
                    >
                      <Icon className="h-4 w-4" />
                      <span>{action.label}</span>
                    </Button>
                  );
                })}
              </div>
            </div>
          </header>

          {error && (
            <div className="mx-auto w-full max-w-5xl rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-700" data-testid="profile-error-message">
              {error}
            </div>
          )}

          <main className="mx-auto w-full max-w-5xl space-y-6">
            <Card className="overflow-hidden border-border/70 bg-card/95">
              <div className="h-28 w-full bg-gradient-to-r from-indigo-600/85 via-purple-600/75 to-indigo-500/85" />
              <CardContent className="space-y-6 p-6 pt-0 sm:p-7 sm:pt-0">
                <div className="-mt-12 flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
                  <div className="flex items-end gap-4">
                    {profileData.avatar_url.trim() ? (
                      <img
                        src={profileData.avatar_url}
                        alt="Profil avatarı"
                        className="h-24 w-24 rounded-full border-4 border-white object-cover shadow-sm"
                      />
                    ) : (
                      <div className="flex h-24 w-24 items-center justify-center rounded-full border-4 border-white bg-gradient-to-br from-slate-700 to-slate-900 text-2xl font-semibold text-white shadow-sm">
                        {getInitials(displayName)}
                      </div>
                    )}
                    <div className="pb-1">
                      <p className="text-2xl font-bold tracking-tight text-foreground">{displayName}</p>
                      <p className="text-sm text-muted-foreground">@{username}</p>
                    </div>
                  </div>

                  <Button
                    type="button"
                    onClick={() => navigate("/profile/edit")}
                    className="h-10 rounded-lg bg-indigo-600 px-4 text-sm font-semibold text-white hover:bg-indigo-700"
                    data-testid="profile-edit-button"
                  >
                    Profili düzenle
                  </Button>
                </div>

                <div className="space-y-1">
                  <p className="text-sm leading-6 text-slate-700">{socialProfile.bio}</p>
                  <p className="text-sm font-medium text-indigo-700">Odak: {profileMetaLine || socialProfile.studyFocus}</p>
                </div>

                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 sm:gap-4">
                  <div className="rounded-xl border border-border/70 bg-background/80 px-3 py-2.5 text-center">
                    <p className="text-base font-semibold text-foreground">{profileStats.posts}</p>
                    <p className="text-xs text-muted-foreground">Gönderi</p>
                  </div>

                  <button
                    type="button"
                    onClick={() => setListModalState({ open: true, type: "followers" })}
                    className="rounded-xl border border-border/70 bg-background/80 px-3 py-2.5 text-center transition-colors duration-200 hover:border-indigo-200 hover:bg-indigo-50"
                  >
                    <p className="text-base font-semibold text-foreground">{profileStats.followers}</p>
                    <p className="text-xs text-muted-foreground">Takipçi</p>
                  </button>

                  <button
                    type="button"
                    onClick={() => setListModalState({ open: true, type: "following" })}
                    className="rounded-xl border border-border/70 bg-background/80 px-3 py-2.5 text-center transition-colors duration-200 hover:border-indigo-200 hover:bg-indigo-50"
                  >
                    <p className="text-base font-semibold text-foreground">{profileStats.following}</p>
                    <p className="text-xs text-muted-foreground">Takip edilen</p>
                  </button>

                  <div className="rounded-xl border border-border/70 bg-background/80 px-3 py-2.5 text-center">
                    <p className="text-base font-semibold text-foreground">{profileStats.studyHours}s</p>
                    <p className="text-xs text-muted-foreground">Çalışma saati</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="border-border/70 bg-card/95">
              <CardContent className="p-4 sm:p-5">
                <Tabs defaultValue="posts" className="w-full">
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="posts">Paylaşımlar</TabsTrigger>
                    <TabsTrigger value="comments">Yorumlar</TabsTrigger>
                    <TabsTrigger value="likes">Beğeniler</TabsTrigger>
                  </TabsList>

                  <TabsContent value="posts" className="mt-4 space-y-3">
                    {ownPosts.length > 0 ? (
                      ownPosts.map((post) => (
                        <div key={post.id} className="rounded-xl border border-border/70 bg-background/85 p-4">
                          <p className="text-sm leading-6 text-slate-700">{post.content}</p>
                          <div className="mt-2 flex items-center gap-3 text-xs text-muted-foreground">
                            <span>{getRelativeTimeLabel(post.createdAt)}</span>
                            <span>•</span>
                            <span>{post.likeCount} beğeni</span>
                            <span>•</span>
                            <span>{post.commentCount} yorum</span>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="rounded-xl border border-dashed border-border/80 bg-background/70 px-4 py-8 text-center text-sm text-muted-foreground">
                        Henüz paylaşım yapmadın.
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="comments" className="mt-4 space-y-3">
                    {ownComments.length > 0 ? (
                      ownComments.map((comment) => (
                        <div key={comment.id} className="rounded-xl border border-border/70 bg-background/85 p-4">
                          <p className="text-sm text-slate-700">{comment.text}</p>
                          <p className="mt-1 truncate text-xs text-muted-foreground">Paylaşım: {comment.postContent}</p>
                          <p className="mt-1 text-xs text-muted-foreground">{getRelativeTimeLabel(comment.createdAt)}</p>
                        </div>
                      ))
                    ) : (
                      <div className="rounded-xl border border-dashed border-border/80 bg-background/70 px-4 py-8 text-center text-sm text-muted-foreground">
                        Henüz yorum yapmadın.
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="likes" className="mt-4 space-y-3">
                    {likedPosts.length > 0 ? (
                      likedPosts.map((post) => (
                        <div key={post.id} className="rounded-xl border border-border/70 bg-background/85 p-4">
                          <p className="text-sm leading-6 text-slate-700">{post.content}</p>
                          <p className="mt-1 text-xs text-muted-foreground">@{post.username}</p>
                        </div>
                      ))
                    ) : (
                      <div className="rounded-xl border border-dashed border-border/80 bg-background/70 px-4 py-8 text-center text-sm text-muted-foreground">
                        Henüz beğeni yok.
                      </div>
                    )}
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </main>
        </div>
      </div>

      <Dialog open={listModalState.open} onOpenChange={(open) => setListModalState((prev) => ({ ...prev, open }))}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{listModalState.type === "followers" ? "Takipçiler" : "Takip edilenler"}</DialogTitle>
          </DialogHeader>

          <div className="space-y-2.5">
            {listUsers.length > 0 ? (
              listUsers.map((listUser) => {
                const listUserFollowing = isForumFollowing(listUser.username);
                return (
                  <div key={listUser.username} className="flex items-center justify-between rounded-lg border border-border/70 bg-background/85 p-2.5">
                    <button
                      type="button"
                      onClick={() => {
                        navigate(`/user/${listUser.username}`);
                        setListModalState({ open: false, type: "followers" });
                      }}
                      className="flex min-w-0 items-center gap-2 text-left"
                    >
                      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-slate-700 to-slate-900 text-xs font-semibold text-white">
                        {getInitials(listUser.displayName)}
                      </div>
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium text-foreground">{listUser.displayName}</p>
                        <p className="truncate text-xs text-muted-foreground">@{listUser.username}</p>
                      </div>
                    </button>

                    <Button
                      type="button"
                      size="sm"
                      onClick={() => toggleForumFollow(listUser.username)}
                      disabled={listUser.username === "sen"}
                      className={`h-8 rounded-lg px-3 text-xs ${
                        listUserFollowing
                          ? "border border-indigo-200 bg-indigo-100 text-indigo-700 hover:bg-indigo-200"
                          : "bg-indigo-600 text-white hover:bg-indigo-700"
                      }`}
                    >
                      {listUser.username === "sen" ? "Sen" : listUserFollowing ? "Takipte" : "Takip et"}
                    </Button>
                  </div>
                );
              })
            ) : (
              <div className="rounded-lg border border-dashed border-border/80 px-4 py-6 text-center text-sm text-muted-foreground">
                Liste boş görünüyor.
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
