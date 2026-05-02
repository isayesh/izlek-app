import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import {
  ArrowLeft,
  BarChart3,
  Eye,
  Heart,
  Home,
  ImageOff,
  MessageCircle,
  MessageSquare,
  Repeat2,
  Trophy,
  User,
  Users,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  addForumPostComment,
  createForumSharePost,
  extractForumMentions,
  getForumFeedPosts,
  incrementForumPostView,
  subscribeForumFeedStore,
  toggleForumPostLike,
} from "@/pages/forumSocialStore";

const MENTION_REGEX = /@([a-zA-Z0-9_]+)/g;

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

const renderTextWithMentions = (text = "", onMentionClick, keyPrefix = "mention") => {
  const elements = [];
  let cursor = 0;
  let mentionIndex = 0;
  let match;

  while ((match = MENTION_REGEX.exec(text)) !== null) {
    const [fullMatch, username] = match;
    const matchStart = match.index;

    if (matchStart > cursor) {
      elements.push(text.slice(cursor, matchStart));
    }

    elements.push(
      <button
        key={`${keyPrefix}-${username}-${mentionIndex}`}
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          onMentionClick(username);
        }}
        className="font-semibold text-indigo-600 hover:text-indigo-700"
      >
        {fullMatch}
      </button>
    );

    cursor = matchStart + fullMatch.length;
    mentionIndex += 1;
  }

  if (cursor < text.length) {
    elements.push(text.slice(cursor));
  }

  MENTION_REGEX.lastIndex = 0;
  return elements;
};

export default function PostDetail() {
  const navigate = useNavigate();
  const location = useLocation();
  const { id = "" } = useParams();
  const viewedRef = useRef("");

  const [posts, setPosts] = useState(() => getForumFeedPosts());
  const [commentDraft, setCommentDraft] = useState("");
  const [shareDraft, setShareDraft] = useState("");
  const [shareExpanded, setShareExpanded] = useState(false);
  const [shareMenuOpen, setShareMenuOpen] = useState(false);
  const [imageLoadErrorByPost, setImageLoadErrorByPost] = useState({});

  useEffect(() => {
    const unsubscribeFeed = subscribeForumFeedStore(() => {
      setPosts(getForumFeedPosts());
    });

    return unsubscribeFeed;
  }, []);

  const postMap = useMemo(
    () => Object.fromEntries(posts.map((postItem) => [postItem.id, postItem])),
    [posts]
  );

  const post = postMap[id] || null;
  const sourcePost = post?.sharedFromPostId ? postMap[post.sharedFromPostId] : null;
  const isPureRepost = Boolean(post?.sharedFromPostId) && !(post?.content || "").trim();
  const hasImageError = Boolean(imageLoadErrorByPost[post?.id || ""]);

  useEffect(() => {
    if (!post?.id) return;

    if (viewedRef.current !== post.id) {
      viewedRef.current = post.id;
      incrementForumPostView(post.id);
    }
  }, [post?.id]);

  const mentions = useMemo(() => {
    if (!post) return [];

    const mentionSet = new Set(extractForumMentions(post.content || ""));
    (post.comments || []).forEach((comment) => {
      extractForumMentions(comment.text || "").forEach((username) => mentionSet.add(username));
    });

    return [...mentionSet];
  }, [post]);

  const navigationActions = [
    { label: "Ana Sayfa", icon: Home, onClick: () => navigate("/") },
    { label: "Net Takibi", icon: BarChart3, onClick: () => navigate("/net-tracking") },
    { label: "Odalar", icon: Users, onClick: () => navigate("/rooms") },
    { label: "Liderlik", icon: Trophy, onClick: () => navigate("/leaderboard") },
    { label: "Forum", icon: MessageCircle, onClick: () => navigate("/forum") },
    { label: "Profil", icon: User, onClick: () => navigate("/profile") },
    { label: "DM Kutum", icon: MessageSquare, onClick: () => navigate("/messages") },
  ];

  const navActivePaths = {
    "Ana Sayfa": ["/", "/dashboard"],
    "Net Takibi": ["/net-tracking"],
    Odalar: ["/rooms"],
    Liderlik: ["/leaderboard"],
    Forum: ["/forum"],
    Profil: ["/profile", "/profile/edit"],
    "DM Kutum": ["/messages"],
  };

  const handleCommentSubmit = () => {
    const trimmed = commentDraft.trim();
    if (!post?.id || !trimmed) return;

    addForumPostComment(post.id, trimmed, "sen");
    setCommentDraft("");
  };

  const handleRepost = () => {
    if (!post?.id) return;

    createForumSharePost({
      postId: post.id,
      quoteText: "",
      displayName: "Sen",
      username: "sen",
    });

    setShareMenuOpen(false);
    setShareExpanded(false);
    setShareDraft("");
    navigate("/forum");
  };

  const handleOpenQuote = () => {
    setShareMenuOpen(false);
    setShareExpanded(true);
  };

  const handleShareSubmit = () => {
    if (!post?.id) return;

    createForumSharePost({
      postId: post.id,
      quoteText: shareDraft,
      displayName: "Sen",
      username: "sen",
    });

    setShareDraft("");
    setShareExpanded(false);
    setShareMenuOpen(false);
    navigate("/forum");
  };

  if (!post) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto max-w-3xl px-4 py-10 sm:px-6">
          <Card className="border-border/70 bg-card/95">
            <CardContent className="space-y-4 p-6">
              <p className="text-sm text-muted-foreground">Bu paylaşım bulunamadı veya kaldırıldı.</p>
              <Button type="button" onClick={() => navigate("/forum")} className="w-fit rounded-lg bg-indigo-600 text-white hover:bg-indigo-700">
                Foruma dön
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground" data-testid="post-detail-page">
      <div className="px-4 pb-10 pt-4 sm:px-6 sm:pb-12 lg:px-10 xl:px-12">
        <div className="space-y-8 sm:space-y-10">
          <header className="flex flex-col gap-3 border-b border-indigo-200 bg-indigo-50/35 pb-4 xl:flex-row xl:items-center xl:gap-4">
            <button
              type="button"
              onClick={() => navigate("/dashboard")}
              className="inline-flex h-12 w-fit shrink-0 items-center px-1"
              data-testid="post-detail-header-brand"
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
                      ? location.pathname.startsWith("/forum") || location.pathname.startsWith("/post/")
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

          <main className="mx-auto w-full max-w-3xl space-y-4">
            <Button
              type="button"
              variant="ghost"
              onClick={() => navigate("/forum")}
              className="h-9 w-fit rounded-lg border border-transparent px-2.5 text-slate-600 hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700"
            >
              <ArrowLeft className="h-4 w-4" />
              Foruma dön
            </Button>

            <Card className="border-border/70 bg-card/95">
              <CardHeader className="space-y-3 p-4 pb-2 sm:p-5 sm:pb-2">
                <div className="flex items-start gap-3">
                  <button
                    type="button"
                    onClick={() => navigate(`/user/${post.username}`)}
                    className="group flex min-w-0 flex-1 items-start gap-3 text-left"
                  >
                    <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-slate-700 to-slate-900 text-sm font-semibold text-white transition-transform duration-200 group-hover:scale-[1.03]">
                      {getInitials(post.displayName)}
                    </div>

                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
                        <span className="font-semibold text-foreground transition-colors duration-200 group-hover:text-indigo-700">
                          {post.displayName}
                        </span>
                        <span className="text-muted-foreground">@{post.username}</span>
                        <span className="text-muted-foreground">· {getRelativeTimeLabel(post.createdAt)}</span>
                      </div>
                    </div>
                  </button>
                </div>
              </CardHeader>

              <CardContent className="space-y-4 p-4 pt-1 sm:p-5 sm:pt-1">
                {isPureRepost && (
                  <p className="text-xs font-semibold text-muted-foreground">
                    {post.username === "sen" ? "Sen yeniden paylaştın" : `${post.displayName} yeniden paylaştı`}
                  </p>
                )}

                {post.content && (
                  <p className="whitespace-pre-wrap text-sm leading-6 text-slate-700 sm:text-[15px]">
                    {renderTextWithMentions(post.content, (username) => navigate(`/user/${username}`), `detail-${post.id}`)}
                  </p>
                )}

                {post.imagePreviewUrl && (
                  <div className="overflow-hidden rounded-xl border border-border/70 bg-muted/20">
                    {!hasImageError ? (
                      <img
                        src={post.imagePreviewUrl}
                        alt="Paylaşım görseli"
                        className="max-h-[500px] h-auto w-full object-contain"
                        onError={() =>
                          setImageLoadErrorByPost((prev) => ({
                            ...prev,
                            [post.id]: true,
                          }))
                        }
                      />
                    ) : (
                      <div className="flex h-52 w-full items-center justify-center gap-2 text-sm text-muted-foreground">
                        <ImageOff className="h-4 w-4" />
                        <span>Görsel önizlemesi yüklenemedi</span>
                      </div>
                    )}
                  </div>
                )}

                {post.sharedFromPostId && (
                  <div className="rounded-xl border border-border/70 bg-background/80 p-3">
                    {sourcePost ? (
                      <>
                        <p className="text-xs font-semibold text-muted-foreground">@{sourcePost.username} paylaşımından alıntı</p>
                        <p className="mt-1 whitespace-pre-wrap text-sm text-slate-700">
                          {renderTextWithMentions(sourcePost.content, (username) => navigate(`/user/${username}`), `detail-source-${sourcePost.id}`)}
                        </p>
                      </>
                    ) : (
                      <p className="text-xs text-muted-foreground">Orijinal paylaşım artık mevcut değil.</p>
                    )}
                  </div>
                )}

                <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => toggleForumPostLike(post.id)}
                    className={`h-9 rounded-lg border px-3 text-xs sm:text-sm ${
                      post.liked
                        ? "border-rose-200 bg-rose-50 text-rose-600 hover:bg-rose-100"
                        : "border-transparent text-slate-600 hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700"
                    }`}
                  >
                    <Heart className={`h-4 w-4 ${post.liked ? "fill-current" : ""}`} />
                    <span>{post.likeCount}</span>
                  </Button>

                  <div className="inline-flex h-9 items-center gap-2 rounded-lg border border-transparent px-3 text-xs text-slate-600 sm:text-sm">
                    <MessageCircle className="h-4 w-4" />
                    <span>{post.commentCount}</span>
                  </div>

                  <div className="relative z-30">
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => setShareMenuOpen((prev) => !prev)}
                      className="h-9 rounded-lg border border-transparent px-3 text-xs text-slate-600 hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700 sm:text-sm"
                    >
                      <Repeat2 className="h-4 w-4" />
                      <span>{post.shareCount || 0}</span>
                    </Button>

                    {shareMenuOpen && (
                      <div className="absolute left-0 top-10 z-[80] w-44 overflow-hidden rounded-lg border border-border/80 bg-white shadow-lg">
                        <button
                          type="button"
                          className="flex w-full items-center px-3 py-2 text-left text-sm text-slate-700 hover:bg-indigo-50"
                          onClick={handleRepost}
                        >
                          Yeniden paylaş
                        </button>
                        <button
                          type="button"
                          className="flex w-full items-center border-t border-border/70 px-3 py-2 text-left text-sm text-slate-700 hover:bg-indigo-50"
                          onClick={handleOpenQuote}
                        >
                          Alıntıla
                        </button>
                      </div>
                    )}
                  </div>

                  <div className="inline-flex h-9 items-center gap-2 rounded-lg border border-transparent px-3 text-xs text-slate-500 sm:text-sm">
                    <Eye className="h-4 w-4" />
                    <span>{post.viewCount || 0}</span>
                  </div>
                </div>

                {shareExpanded && (
                  <div className="space-y-2 rounded-xl border border-border/70 bg-muted/20 p-3 sm:p-4">
                    <textarea
                      value={shareDraft}
                      onChange={(event) => setShareDraft(event.target.value)}
                      placeholder="Paylaşımına not ekle (opsiyonel)"
                      rows={2}
                      className="w-full resize-none rounded-lg border border-border/80 bg-background px-3 py-2 text-sm outline-none transition-colors duration-200 focus:border-indigo-300 focus:ring-2 focus:ring-indigo-200/70"
                    />
                    <div className="flex justify-end gap-2">
                      <Button
                        type="button"
                        variant="outline"
                        className="h-9 rounded-lg px-3"
                        onClick={() => setShareExpanded(false)}
                      >
                        Vazgeç
                      </Button>
                      <Button
                        type="button"
                        className="h-9 rounded-lg bg-indigo-600 px-4 text-white hover:bg-indigo-700"
                        onClick={handleShareSubmit}
                      >
                        Paylaşımı alıntıla
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="border-border/70 bg-card/95">
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Yorumlar</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {post.comments.length > 0 ? (
                  post.comments.map((comment) => (
                    <div key={comment.id} className="rounded-lg border border-border/60 bg-background/90 px-3 py-2">
                      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                        <button
                          type="button"
                          className="font-semibold text-foreground hover:text-indigo-700"
                          onClick={() => navigate(`/user/${comment.author}`)}
                        >
                          @{comment.author}
                        </button>
                        <span>· {getRelativeTimeLabel(comment.createdAt)}</span>
                      </div>
                      <p className="mt-1 whitespace-pre-wrap text-sm text-slate-700">
                        {renderTextWithMentions(comment.text, (username) => navigate(`/user/${username}`), `detail-comment-${comment.id}`)}
                      </p>
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-muted-foreground">Henüz yorum yapılmadı.</p>
                )}

                <div className="flex flex-col gap-2 sm:flex-row">
                  <input
                    type="text"
                    value={commentDraft}
                    onChange={(event) => setCommentDraft(event.target.value)}
                    placeholder="Yorumunu yaz..."
                    className="h-10 flex-1 rounded-lg border border-border/80 bg-background px-3 text-sm outline-none transition-colors duration-200 focus:border-indigo-300 focus:ring-2 focus:ring-indigo-200/70"
                  />
                  <Button
                    type="button"
                    size="sm"
                    onClick={handleCommentSubmit}
                    disabled={!commentDraft.trim()}
                    className="h-10 rounded-lg bg-indigo-600 px-4 text-white hover:bg-indigo-700"
                  >
                    Gönder
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card className="border-border/70 bg-card/95">
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Bahsedilen Kullanıcılar</CardTitle>
              </CardHeader>
              <CardContent>
                {mentions.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {mentions.map((username) => (
                      <button
                        key={username}
                        type="button"
                        onClick={() => navigate(`/user/${username}`)}
                        className="rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700 hover:bg-indigo-100"
                      >
                        @{username}
                      </button>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">Bu paylaşımda etiketlenen kullanıcı yok.</p>
                )}
              </CardContent>
            </Card>
          </main>
        </div>
      </div>
    </div>
  );
}
