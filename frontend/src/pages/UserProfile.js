import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import {
  BarChart3,
  Clock3,
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
import {
  getForumUserProfile,
  getForumUserStats,
  getForumUsersByUsernames,
  isForumFollowing,
  PUBLIC_PROFILE_ACTIVITY,
  PUBLIC_PROFILE_POSTS,
  subscribeForumFollowStore,
  toggleForumFollow,
} from "@/pages/forumSocialStore";

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

export default function UserProfile() {
  const navigate = useNavigate();
  const location = useLocation();
  const { username = "" } = useParams();

  const [, setFollowStoreVersion] = useState(0);
  const [listModalState, setListModalState] = useState({ open: false, type: "followers" });

  useEffect(() => {
    const unsubscribe = subscribeForumFollowStore(() => {
      setFollowStoreVersion((prev) => prev + 1);
    });

    return unsubscribe;
  }, []);

  const profile = useMemo(() => getForumUserProfile(username), [username]);
  const profileStats = getForumUserStats(profile.username);
  const isFollowing = isForumFollowing(profile.username);

  const profilePosts = useMemo(
    () => PUBLIC_PROFILE_POSTS.filter((post) => post.username === profile.username),
    [profile.username]
  );

  const activity = PUBLIC_PROFILE_ACTIVITY[profile.username] || { comments: [], likes: [] };

  const listUsers = useMemo(() => {
    const sourceList = listModalState.type === "followers" ? profile.followersList : profile.followingList;
    return getForumUsersByUsernames(sourceList);
  }, [listModalState.type, profile.followersList, profile.followingList]);

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

  return (
    <div className="min-h-screen bg-background text-foreground" data-testid="public-profile-page">
      <div className="px-4 pb-10 pt-4 sm:px-6 sm:pb-12 lg:px-10 xl:px-12">
        <div className="space-y-8 sm:space-y-10">
          <header className="flex flex-col gap-3 border-b border-indigo-200 bg-indigo-50/35 pb-4 xl:flex-row xl:items-center xl:gap-4">
            <button
              type="button"
              onClick={() => navigate("/dashboard")}
              className="inline-flex h-12 w-fit shrink-0 items-center px-1"
              data-testid="user-profile-header-brand"
              aria-label="Dashboard"
            >
              <span className="font-display text-4xl font-extrabold leading-none tracking-[-0.03em] text-gray-900">izlek</span>
            </button>

            <div className="w-full max-w-full flex-1 overflow-x-auto pb-1 xl:pb-0">
              <div className="flex items-center gap-2.5 lg:gap-3 xl:justify-center xl:gap-3.5">
                {navigationActions.map((action) => {
                  const Icon = action.icon;
                  const isActive =
                    action.label === "Forum"
                      ? location.pathname.startsWith("/forum") || location.pathname.startsWith("/user/")
                      : (navActivePaths[action.label] || []).includes(location.pathname);

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

          <main className="mx-auto w-full max-w-5xl space-y-6">
            <Card className="overflow-hidden border-border/70 bg-card/95">
              <div className="h-36 w-full bg-gradient-to-r from-indigo-600/85 via-purple-600/75 to-indigo-500/85" />
              <CardContent className="space-y-7 p-6 pt-6 sm:p-7 sm:pt-7">
                <div className="-mt-12 flex flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
                  <div className="flex items-start gap-4 sm:gap-5">
                    <div className="flex h-24 w-24 items-center justify-center rounded-full border-4 border-white bg-gradient-to-br from-slate-700 to-slate-900 text-2xl font-semibold text-white shadow-sm">
                      {getInitials(profile.displayName)}
                    </div>
                    <div className="pt-10 sm:pt-11">
                      <p className="text-2xl font-bold tracking-tight text-foreground">{profile.displayName}</p>
                      <p className="text-sm text-muted-foreground">@{profile.username}</p>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-2 sm:mt-11">
                    <Button
                      type="button"
                      onClick={() => toggleForumFollow(profile.username)}
                      disabled={profile.username === "sen"}
                      className={`h-10 rounded-lg px-4 text-sm font-semibold ${
                        isFollowing
                          ? "border border-indigo-200 bg-indigo-100 text-indigo-700 hover:bg-indigo-200"
                          : "bg-indigo-600 text-white hover:bg-indigo-700"
                      }`}
                    >
                      {profile.username === "sen" ? "Bu sensin" : isFollowing ? "Takip ediliyor" : "Takip et"}
                    </Button>
                    <Button type="button" variant="outline" className="h-10 rounded-lg px-4 text-sm">
                      Mesaj gönder
                    </Button>
                  </div>
                </div>

                <div className="space-y-1">
                  <p className="text-sm leading-6 text-slate-700">{profile.bio}</p>
                  <p className="text-sm font-medium text-indigo-700">Odak: {profile.studyFocus}</p>
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
                    {profilePosts.length > 0 ? (
                      profilePosts.map((post) => (
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
                        Henüz paylaşım yok.
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="comments" className="mt-4 space-y-3">
                    {activity.comments.length > 0 ? (
                      activity.comments.map((comment) => (
                        <div key={comment.id} className="rounded-xl border border-border/70 bg-background/85 p-4">
                          <p className="text-sm text-slate-700">{comment.text}</p>
                          <p className="mt-1 text-xs text-muted-foreground">{getRelativeTimeLabel(comment.createdAt)}</p>
                        </div>
                      ))
                    ) : (
                      <div className="rounded-xl border border-dashed border-border/80 bg-background/70 px-4 py-8 text-center text-sm text-muted-foreground">
                        Henüz yorum yok.
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="likes" className="mt-4 space-y-3">
                    {activity.likes.length > 0 ? (
                      activity.likes.map((like) => (
                        <div key={like.id} className="rounded-xl border border-border/70 bg-background/85 p-4 text-sm text-slate-700">
                          {like.text}
                        </div>
                      ))
                    ) : (
                      <div className="rounded-xl border border-dashed border-border/80 bg-background/70 px-4 py-8 text-center text-sm text-muted-foreground">
                        Henüz beğeni etkinliği yok.
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
