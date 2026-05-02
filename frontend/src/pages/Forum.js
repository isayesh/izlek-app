import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  BarChart3,
  Eye,
  Hash,
  Heart,
  Home,
  ImagePlus,
  MessageCircle,
  MessageSquare,
  Repeat2,
  ShieldCheck,
  Trophy,
  User,
  Users,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import {
  addForumPostComment,
  createForumPost,
  createForumSharePost,
  getForumFeedPosts,
  getForumUserProfile,
  getForumUserStats,
  incrementForumPostView,
  isForumFollowing,
  subscribeForumFeedStore,
  subscribeForumFollowStore,
  toggleForumFollow,
  toggleForumPostLike,
} from "@/pages/forumSocialStore";

const TOPIC_KEYWORDS = [
  { topic: "TYT", keywords: ["tyt", "problem", "paragraf"] },
  { topic: "AYT", keywords: ["ayt", "türev", "integral", "limit", "edebiyat"] },
  { topic: "Deneme Analizi", keywords: ["deneme", "yanlış", "analiz", "net"] },
  { topic: "Matematik", keywords: ["matematik", "geometri"] },
  { topic: "Pomodoro", keywords: ["pomodoro", "mola", "odak"] },
  { topic: "Çalışma Programı", keywords: ["program", "plan", "rutin"] },
  { topic: "Motivasyon", keywords: ["motivasyon", "yoruldum", "bıktım", "stres"] },
  { topic: "Soru Çözümü", keywords: ["soru", "çözüm", "test"] },
];

const SUGGESTED_DISCUSSIONS = [
  "TYT matematikte problem hızını nasıl artırıyorsunuz?",
  "Denemeden sonra yanlış analizi nasıl yapılmalı?",
  "Günde kaç saat çalışmak sürdürülebilir?",
  "Paragraf netlerini artırmak için ne yapıyorsunuz?",
];

const COMMUNITY_RULES = [
  "Saygılı dil kullan, kişisel saldırıdan kaçın.",
  "Deneme ve kaynak önerilerinde yapıcı ve açıklayıcı ol.",
  "Spam ve aynı içeriği tekrarlamaktan kaçın.",
];

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

export default function Forum() {
  const navigate = useNavigate();
  const location = useLocation();
  const imageInputRef = useRef(null);
  const renderedPostIdsRef = useRef(new Set());

  const [posts, setPosts] = useState(() => getForumFeedPosts());
  const [composerText, setComposerText] = useState("");
  const [composerImage, setComposerImage] = useState(null);
  const [commentDrafts, setCommentDrafts] = useState({});
  const [expandedCommentsByPost, setExpandedCommentsByPost] = useState({});
  const [expandedShareByPost, setExpandedShareByPost] = useState({});
  const [shareMenuByPost, setShareMenuByPost] = useState({});
  const [shareDraftsByPost, setShareDraftsByPost] = useState({});
  const [activeProfileUsername, setActiveProfileUsername] = useState("");
  const [, setFollowStoreVersion] = useState(0);

  useEffect(() => {
    const unsubscribeFeed = subscribeForumFeedStore(() => {
      setPosts(getForumFeedPosts());
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
    posts.forEach((post) => {
      if (!renderedPostIdsRef.current.has(post.id)) {
        renderedPostIdsRef.current.add(post.id);
        incrementForumPostView(post.id);
      }
    });
  }, [posts]);

  const postMap = useMemo(
    () => Object.fromEntries(posts.map((post) => [post.id, post])),
    [posts]
  );

  const activeProfile = activeProfileUsername ? getForumUserProfile(activeProfileUsername) : null;
  const activeProfileStats = activeProfile ? getForumUserStats(activeProfile.username) : null;
  const isActiveProfileFollowing = activeProfile ? isForumFollowing(activeProfile.username) : false;

  const canSubmitPost = composerText.trim().length > 0;

  const inferMainTopicFromText = (text = "") => {
    const normalizedText = text.toLocaleLowerCase("tr-TR");
    const matchedTopic = TOPIC_KEYWORDS.find(({ keywords }) =>
      keywords.some((keyword) => normalizedText.includes(keyword))
    );
    return matchedTopic?.topic || "Genel";
  };

  const trendingTopics = useMemo(() => {
    const topicCounts = posts.reduce((accumulator, post) => {
      const topic = inferMainTopicFromText(post.content || "");
      return {
        ...accumulator,
        [topic]: (accumulator[topic] || 0) + 1,
      };
    }, {});

    return Object.entries(topicCounts)
      .map(([topic, count]) => ({ topic, count }))
      .sort((a, b) => b.count - a.count || a.topic.localeCompare(b.topic, "tr"))
      .slice(0, 6);
  }, [posts]);

  const navigationActions = useMemo(
    () => [
      { label: "Ana Sayfa", icon: Home, onClick: () => navigate("/") },
      { label: "Net Takibi", icon: BarChart3, onClick: () => navigate("/net-tracking") },
      { label: "Odalar", icon: Users, onClick: () => navigate("/rooms") },
      { label: "Liderlik", icon: Trophy, onClick: () => navigate("/leaderboard") },
      { label: "Forum", icon: MessageCircle, onClick: () => navigate("/forum") },
      { label: "Profil", icon: User, onClick: () => navigate("/profile") },
      { label: "DM Kutum", icon: MessageSquare, onClick: () => navigate("/messages") },
    ],
    [navigate]
  );

  const navActivePaths = {
    "Ana Sayfa": ["/", "/dashboard"],
    "Net Takibi": ["/net-tracking"],
    Odalar: ["/rooms"],
    Liderlik: ["/leaderboard"],
    Forum: ["/forum"],
    Profil: ["/profile"],
    "DM Kutum": ["/messages"],
  };

  const resetComposer = () => {
    setComposerText("");
    setComposerImage((prevImage) => {
      if (prevImage?.previewUrl) {
        URL.revokeObjectURL(prevImage.previewUrl);
      }
      return null;
    });

    if (imageInputRef.current) {
      imageInputRef.current.value = "";
    }
  };

  const removeComposerImage = () => {
    setComposerImage((prevImage) => {
      if (prevImage?.previewUrl) {
        URL.revokeObjectURL(prevImage.previewUrl);
      }
      return null;
    });

    if (imageInputRef.current) {
      imageInputRef.current.value = "";
    }
  };

  const handleComposerSubmit = (event) => {
    event.preventDefault();
    if (!canSubmitPost) return;

    createForumPost({
      displayName: "Sen",
      username: "sen",
      content: composerText,
      imageName: composerImage?.name || "",
      imagePreviewUrl: composerImage?.previewUrl || "",
    });

    resetComposer();
  };

  const handleImageSelected = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const previewUrl = URL.createObjectURL(file);
    setComposerImage((prevImage) => {
      if (prevImage?.previewUrl) {
        URL.revokeObjectURL(prevImage.previewUrl);
      }
      return {
        name: file.name,
        previewUrl,
      };
    });
  };

  const toggleCommentArea = (postId) => {
    setExpandedCommentsByPost((prevState) => ({
      ...prevState,
      [postId]: !prevState[postId],
    }));
  };

  const submitComment = (postId) => {
    const draft = (commentDrafts[postId] || "").trim();
    if (!draft) return;

    addForumPostComment(postId, draft, "sen");
    setExpandedCommentsByPost((prevState) => ({
      ...prevState,
      [postId]: true,
    }));

    setCommentDrafts((prev) => ({
      ...prev,
      [postId]: "",
    }));
  };

  const toggleShareMenu = (postId) => {
    setShareMenuByPost((prev) => ({
      ...prev,
      [postId]: !prev[postId],
    }));
  };

  const openQuoteComposer = (postId) => {
    setExpandedShareByPost((prev) => ({
      ...prev,
      [postId]: true,
    }));
    setShareMenuByPost((prev) => ({
      ...prev,
      [postId]: false,
    }));
  };

  const submitRepost = (postId) => {
    createForumSharePost({
      postId,
      quoteText: "",
      displayName: "Sen",
      username: "sen",
    });

    setShareMenuByPost((prev) => ({
      ...prev,
      [postId]: false,
    }));

    setExpandedShareByPost((prev) => ({
      ...prev,
      [postId]: false,
    }));

    setShareDraftsByPost((prev) => ({
      ...prev,
      [postId]: "",
    }));
  };

  const submitShare = (postId) => {
    createForumSharePost({
      postId,
      quoteText: shareDraftsByPost[postId] || "",
      displayName: "Sen",
      username: "sen",
    });

    setExpandedShareByPost((prev) => ({
      ...prev,
      [postId]: false,
    }));

    setShareMenuByPost((prev) => ({
      ...prev,
      [postId]: false,
    }));

    setShareDraftsByPost((prev) => ({
      ...prev,
      [postId]: "",
    }));
  };

  const handleMentionNavigation = (username) => {
    navigate(`/user/${username}`);
  };

  return (
    <div className="min-h-screen bg-background text-foreground" data-testid="forum-page">
      <div className="px-4 pb-10 pt-4 sm:px-6 sm:pb-12 lg:px-10 xl:px-12">
        <div className="space-y-8 sm:space-y-10">
          <header className="flex flex-col gap-3 border-b border-indigo-200 bg-indigo-50/35 pb-4 xl:flex-row xl:items-center xl:gap-4">
            <button
              type="button"
              onClick={() => navigate("/dashboard")}
              className="inline-flex h-12 w-fit shrink-0 items-center px-1"
              data-testid="forum-header-brand"
              aria-label="Dashboard"
            >
              <span className="font-display text-4xl font-extrabold leading-none tracking-[-0.03em] text-gray-900">izlek</span>
            </button>

            <div className="w-full max-w-full flex-1 overflow-x-auto pb-1 xl:pb-0" data-testid="forum-header-nav">
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

          <div className="mx-auto grid max-w-6xl grid-cols-1 gap-6 xl:grid-cols-[minmax(0,2fr)_minmax(280px,1fr)] xl:gap-7">
            <main className="w-full min-w-0 space-y-5 xl:justify-self-center">
              <section className="space-y-2 px-1">
                <h1 className="font-display text-3xl font-bold tracking-tight text-foreground sm:text-4xl">Forum</h1>
                <p className="max-w-2xl text-sm leading-6 text-muted-foreground sm:text-base">
                  İzlek topluluğunun ders çalışma, net takibi ve sınav hazırlığı paylaşımları burada.
                </p>
              </section>

              <Card className="overflow-hidden border-indigo-100/80 bg-card/90" data-testid="forum-composer-card">
                <CardContent className="p-4 sm:p-5">
                  <form onSubmit={handleComposerSubmit} className="space-y-4">
                    <div className="flex items-start gap-3">
                      <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 text-sm font-bold text-white">
                        SN
                      </div>

                      <div className="w-full min-w-0 space-y-3">
                        <textarea
                          value={composerText}
                          onChange={(event) => setComposerText(event.target.value)}
                          placeholder="Ne düşünüyorsun?"
                          rows={3}
                          className="w-full resize-none rounded-xl border border-border/80 bg-background px-3 py-2.5 text-sm text-foreground outline-none transition-colors duration-200 placeholder:text-muted-foreground focus:border-indigo-300 focus:ring-2 focus:ring-indigo-200/70"
                          data-testid="forum-composer-textarea"
                        />

                        {composerImage?.name && (
                          <div className="inline-flex max-w-full items-center gap-2 rounded-lg border border-indigo-100 bg-indigo-50/70 px-3 py-1.5 text-xs text-indigo-700">
                            <ImagePlus className="h-3.5 w-3.5 shrink-0" />
                            <span className="truncate">{composerImage.name}</span>
                            <button
                              type="button"
                              onClick={removeComposerImage}
                              className="rounded px-1 font-semibold text-indigo-600 hover:bg-indigo-100"
                            >
                              Kaldır
                            </button>
                          </div>
                        )}

                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div>
                            <input
                              ref={imageInputRef}
                              type="file"
                              accept="image/*"
                              onChange={handleImageSelected}
                              className="hidden"
                            />
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              onClick={() => imageInputRef.current?.click()}
                              className="h-9 rounded-lg border border-transparent px-3 text-slate-600 hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700"
                              data-testid="forum-composer-image-button"
                            >
                              <ImagePlus className="h-4 w-4" />
                              Görsel ekle
                            </Button>
                          </div>

                          <Button
                            type="submit"
                            disabled={!canSubmitPost}
                            className="h-9 rounded-lg bg-indigo-600 px-5 text-sm font-semibold text-white hover:bg-indigo-700"
                            data-testid="forum-composer-submit-button"
                          >
                            Paylaş
                          </Button>
                        </div>
                      </div>
                    </div>
                  </form>
                </CardContent>
              </Card>

              <div className="space-y-4" data-testid="forum-feed-list">
                {posts.map((post) => {
                  const inferredTopic = inferMainTopicFromText(post.content || "");
                  const authorProfile = getForumUserProfile(post.username, post.displayName);
                  const sourcePost = post.sharedFromPostId ? postMap[post.sharedFromPostId] : null;
                  const hasCommentsExpanded = Boolean(expandedCommentsByPost[post.id]);
                  const hasShareExpanded = Boolean(expandedShareByPost[post.id]);
                  const hasShareMenuOpen = Boolean(shareMenuByPost[post.id]);
                  const isPureRepost = Boolean(post.sharedFromPostId) && !(post.content || "").trim();

                  return (
                    <Card
                      key={post.id}
                      className="border-border/70 bg-card/95 transition-all duration-200 hover:border-indigo-200/90 hover:shadow-[0_18px_38px_-28px_rgba(79,70,229,0.4)]"
                      data-testid={`forum-post-${post.id}`}
                    >
                      <CardHeader className="space-y-3 p-4 pb-2 sm:p-5 sm:pb-2">
                        <div className="flex items-start gap-3">
                          <button
                            type="button"
                            onClick={() => setActiveProfileUsername(authorProfile.username)}
                            className="group flex min-w-0 flex-1 items-start gap-3 text-left"
                          >
                            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-slate-700 to-slate-900 text-sm font-semibold text-white transition-transform duration-200 group-hover:scale-[1.03]">
                              {getInitials(authorProfile.displayName)}
                            </div>

                            <div className="min-w-0 flex-1">
                              <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
                                <span className="font-semibold text-foreground transition-colors duration-200 group-hover:text-indigo-700">
                                  {authorProfile.displayName}
                                </span>
                                <span className="text-muted-foreground">@{authorProfile.username}</span>
                                <span className="text-muted-foreground">· {getRelativeTimeLabel(post.createdAt)}</span>
                              </div>

                              <span className="mt-2 inline-flex w-fit items-center gap-1 rounded-full border border-indigo-200/80 bg-indigo-50 px-2.5 py-1 text-[11px] font-semibold text-indigo-700">
                                <Hash className="h-3 w-3" />
                                #{inferredTopic}
                              </span>
                            </div>
                          </button>
                        </div>
                      </CardHeader>

                      <CardContent className="space-y-4 p-4 pt-1 sm:p-5 sm:pt-1">
                        <button
                          type="button"
                          onClick={() => navigate(`/post/${post.id}`)}
                          className="w-full space-y-3 text-left"
                          data-testid={`forum-post-open-${post.id}`}
                        >
                          {isPureRepost && (
                            <p className="text-xs font-semibold text-muted-foreground">
                              {post.username === "sen" ? "Sen yeniden paylaştın" : `${post.displayName} yeniden paylaştı`}
                            </p>
                          )}

                          {post.content && (
                            <p className="whitespace-pre-wrap text-sm leading-6 text-slate-700 sm:text-[15px]">
                              {renderTextWithMentions(post.content, handleMentionNavigation, `post-${post.id}`)}
                            </p>
                          )}

                          {post.imagePreviewUrl && (
                            <div className="overflow-hidden rounded-xl border border-border/70 bg-muted/30">
                              <img
                                src={post.imagePreviewUrl}
                                alt={post.imageName || "paylaşım görseli"}
                                className="h-auto w-full object-cover"
                              />
                            </div>
                          )}

                          {post.sharedFromPostId && (
                            <div className="rounded-xl border border-border/70 bg-background/80 p-3">
                              {sourcePost ? (
                                <>
                                  <p className="text-xs font-semibold text-muted-foreground">
                                    @{sourcePost.username} paylaşımından alıntı
                                  </p>
                                  <p className="mt-1 whitespace-pre-wrap text-sm text-slate-700">
                                    {renderTextWithMentions(sourcePost.content, handleMentionNavigation, `shared-${post.id}`)}
                                  </p>
                                </>
                              ) : (
                                <p className="text-xs text-muted-foreground">Orijinal paylaşım artık mevcut değil.</p>
                              )}
                            </div>
                          )}
                        </button>

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
                            data-testid={`forum-like-button-${post.id}`}
                          >
                            <Heart className={`h-4 w-4 ${post.liked ? "fill-current" : ""}`} />
                            <span>{post.likeCount}</span>
                          </Button>

                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={() => toggleCommentArea(post.id)}
                            className="h-9 rounded-lg border border-transparent px-3 text-xs text-slate-600 hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700 sm:text-sm"
                            data-testid={`forum-comment-button-${post.id}`}
                          >
                            <MessageCircle className="h-4 w-4" />
                            <span>{post.commentCount}</span>
                          </Button>

                          <div className="relative z-30">
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              onClick={() => toggleShareMenu(post.id)}
                              className="h-9 rounded-lg border border-transparent px-3 text-xs text-slate-600 hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700 sm:text-sm"
                              data-testid={`forum-share-button-${post.id}`}
                            >
                              <Repeat2 className="h-4 w-4" />
                              <span>{post.shareCount || 0}</span>
                            </Button>

                            {hasShareMenuOpen && (
                              <div className="absolute left-0 top-10 z-[80] w-44 overflow-hidden rounded-lg border border-border/80 bg-white shadow-lg">
                                <button
                                  type="button"
                                  className="flex w-full items-center px-3 py-2 text-left text-sm text-slate-700 hover:bg-indigo-50"
                                  onClick={() => submitRepost(post.id)}
                                >
                                  Yeniden paylaş
                                </button>
                                <button
                                  type="button"
                                  className="flex w-full items-center border-t border-border/70 px-3 py-2 text-left text-sm text-slate-700 hover:bg-indigo-50"
                                  onClick={() => openQuoteComposer(post.id)}
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

                        {hasShareExpanded && (
                          <div className="space-y-2 rounded-xl border border-border/70 bg-muted/20 p-3 sm:p-4">
                            <textarea
                              value={shareDraftsByPost[post.id] || ""}
                              onChange={(event) =>
                                setShareDraftsByPost((prev) => ({
                                  ...prev,
                                  [post.id]: event.target.value,
                                }))
                              }
                              placeholder="Paylaşımına not ekle (opsiyonel)"
                              rows={2}
                              className="w-full resize-none rounded-lg border border-border/80 bg-background px-3 py-2 text-sm outline-none transition-colors duration-200 focus:border-indigo-300 focus:ring-2 focus:ring-indigo-200/70"
                            />
                            <div className="flex justify-end gap-2">
                              <Button
                                type="button"
                                variant="outline"
                                className="h-9 rounded-lg px-3"
                                onClick={() =>
                                  setExpandedShareByPost((prev) => ({
                                    ...prev,
                                    [post.id]: false,
                                  }))
                                }
                              >
                                Vazgeç
                              </Button>
                              <Button
                                type="button"
                                className="h-9 rounded-lg bg-indigo-600 px-4 text-white hover:bg-indigo-700"
                                onClick={() => submitShare(post.id)}
                              >
                                Paylaşımı alıntıla
                              </Button>
                            </div>
                          </div>
                        )}

                        {hasCommentsExpanded && (
                          <div className="space-y-3 rounded-xl border border-border/70 bg-muted/20 p-3 sm:p-4">
                            {post.comments.length > 0 ? (
                              <div className="space-y-2.5">
                                {post.comments.map((comment) => (
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
                                      {renderTextWithMentions(comment.text, handleMentionNavigation, `comment-${comment.id}`)}
                                    </p>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <p className="text-xs text-muted-foreground">Henüz yorum yok. İlk yorumu sen yap.</p>
                            )}

                            <div className="flex flex-col gap-2 sm:flex-row">
                              <input
                                type="text"
                                value={commentDrafts[post.id] || ""}
                                onChange={(event) =>
                                  setCommentDrafts((prev) => ({
                                    ...prev,
                                    [post.id]: event.target.value,
                                  }))
                                }
                                placeholder="Yorumunu yaz..."
                                className="h-10 flex-1 rounded-lg border border-border/80 bg-background px-3 text-sm outline-none transition-colors duration-200 focus:border-indigo-300 focus:ring-2 focus:ring-indigo-200/70"
                                data-testid={`forum-comment-input-${post.id}`}
                              />
                              <Button
                                type="button"
                                size="sm"
                                onClick={() => submitComment(post.id)}
                                disabled={!(commentDrafts[post.id] || "").trim()}
                                className="h-10 rounded-lg bg-indigo-600 px-4 text-white hover:bg-indigo-700"
                                data-testid={`forum-comment-submit-${post.id}`}
                              >
                                Gönder
                              </Button>
                            </div>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  );
                })}
              </div>

              <Dialog
                open={Boolean(activeProfileUsername)}
                onOpenChange={(open) => {
                  if (!open) {
                    setActiveProfileUsername("");
                  }
                }}
              >
                <DialogContent className="max-w-md">
                  {activeProfile && activeProfileStats && (
                    <div className="space-y-4">
                      <DialogHeader>
                        <DialogTitle>Profil Önizleme</DialogTitle>
                      </DialogHeader>

                      <div className="flex items-start gap-3">
                        <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-slate-700 to-slate-900 text-lg font-semibold text-white">
                          {getInitials(activeProfile.displayName)}
                        </div>
                        <div className="min-w-0">
                          <p className="text-lg font-semibold text-foreground">{activeProfile.displayName}</p>
                          <p className="text-sm text-muted-foreground">@{activeProfile.username}</p>
                          <p className="mt-1 text-sm leading-6 text-slate-700">{activeProfile.bio}</p>
                          <p className="text-sm font-medium text-indigo-700">Odak: {activeProfile.studyFocus}</p>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                        <div className="rounded-lg border border-border/70 bg-background/80 px-2.5 py-2 text-center">
                          <p className="text-sm font-semibold text-foreground">{activeProfileStats.posts}</p>
                          <p className="text-[11px] text-muted-foreground">Gönderi</p>
                        </div>
                        <div className="rounded-lg border border-border/70 bg-background/80 px-2.5 py-2 text-center">
                          <p className="text-sm font-semibold text-foreground">{activeProfileStats.followers}</p>
                          <p className="text-[11px] text-muted-foreground">Takipçi</p>
                        </div>
                        <div className="rounded-lg border border-border/70 bg-background/80 px-2.5 py-2 text-center">
                          <p className="text-sm font-semibold text-foreground">{activeProfileStats.following}</p>
                          <p className="text-[11px] text-muted-foreground">Takip edilen</p>
                        </div>
                        <div className="rounded-lg border border-border/70 bg-background/80 px-2.5 py-2 text-center">
                          <p className="text-sm font-semibold text-foreground">{activeProfileStats.studyHours}s</p>
                          <p className="text-[11px] text-muted-foreground">Çalışma saati</p>
                        </div>
                      </div>

                      <div className="flex flex-wrap justify-end gap-2">
                        <Button
                          type="button"
                          onClick={() => toggleForumFollow(activeProfile.username)}
                          disabled={activeProfile.username === "sen"}
                          className={`h-9 rounded-lg px-4 text-sm ${
                            isActiveProfileFollowing
                              ? "border border-indigo-200 bg-indigo-100 text-indigo-700 hover:bg-indigo-200"
                              : "bg-indigo-600 text-white hover:bg-indigo-700"
                          }`}
                        >
                          {activeProfile.username === "sen"
                            ? "Bu sensin"
                            : isActiveProfileFollowing
                              ? "Takip ediliyor"
                              : "Takip et"}
                        </Button>
                        <Button
                          type="button"
                          variant="outline"
                          className="h-9 rounded-lg px-4 text-sm"
                          onClick={() => {
                            navigate(`/user/${activeProfile.username}`);
                            setActiveProfileUsername("");
                          }}
                        >
                          Profili görüntüle
                        </Button>
                      </div>
                    </div>
                  )}
                </DialogContent>
              </Dialog>
            </main>

            <aside className="w-full min-w-0 space-y-4 xl:sticky xl:top-4 xl:max-w-[340px] xl:self-start" data-testid="forum-sidebar">
              <Card className="border-indigo-100/70 bg-card/90">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Trend Konular</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2.5 pt-0">
                  {trendingTopics.length > 0 ? (
                    trendingTopics.map(({ topic, count }) => (
                      <button
                        type="button"
                        key={`${topic}-${count}`}
                        className="flex w-full items-center justify-between rounded-lg border border-transparent bg-transparent px-2.5 py-2 text-left text-sm text-slate-700 transition-colors duration-200 hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700"
                      >
                        <span>{`#${topic} · ${count} gönderi`}</span>
                      </button>
                    ))
                  ) : (
                    <p className="px-2.5 py-2 text-sm text-muted-foreground">Henüz trend konusu oluşmadı.</p>
                  )}
                </CardContent>
              </Card>

              <Card className="border-border/70 bg-card/90">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Önerilen Tartışmalar</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 pt-0">
                  {SUGGESTED_DISCUSSIONS.map((item, index) => (
                    <div key={item} className="rounded-lg border border-border/70 bg-background/80 px-3 py-2.5">
                      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Başlık {index + 1}</p>
                      <p className="mt-1 text-sm leading-6 text-slate-700">{item}</p>
                    </div>
                  ))}
                </CardContent>
              </Card>

              <Card className="border-emerald-100 bg-emerald-50/60">
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-base text-emerald-800">
                    <ShieldCheck className="h-4 w-4" />
                    Topluluk Kuralları
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                  <ul className="space-y-2 text-sm text-emerald-900/90">
                    {COMMUNITY_RULES.map((rule) => (
                      <li key={rule} className="flex items-start gap-2">
                        <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-600" />
                        <span>{rule}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </aside>
          </div>
        </div>
      </div>
    </div>
  );
}
