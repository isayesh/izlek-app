import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  BarChart3,
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
  getForumUserProfile,
  getForumUserStats,
  isForumFollowing,
  subscribeForumFollowStore,
  toggleForumFollow,
} from "@/pages/forumSocialStore";

const minutesAgo = (minutes) => new Date(Date.now() - minutes * 60 * 1000).toISOString();

const FORUM_SEED_POSTS = [
  {
    id: "seed-1",
    displayName: "Mert Analiz",
    username: "merttaktik",
    content:
      "Derbide 4-2-3-1 yerine ikinci yarı 4-3-3'e dönüş kritik oldu. Orta sahada bir ekstra oyuncu kazanınca hem ikinci toplar hem de geçiş savunması toparlandı.",
    createdAt: minutesAgo(18),
    likeCount: 42,
    commentCount: 0,
    liked: false,
    comments: [
      {
        id: "seed-1-comment-1",
        author: "oyunkurucu10",
        text: "Kesinlikle, özellikle 60'tan sonra merkezde üstünlük netti.",
        createdAt: minutesAgo(11),
      },
      {
        id: "seed-1-comment-2",
        author: "savunmaci5",
        text: "Beklerin içe kat etmesi de pas açılarını çok artırdı.",
        createdAt: minutesAgo(7),
      },
    ],
    commentsExpanded: false,
  },
  {
    id: "seed-2",
    displayName: "Transfer Radarı",
    username: "transferhatti",
    content:
      "Fenerbahçe'nin sol bek rotasyonu için iki isim daha gündeme girmiş. Biri tempolu çizgi oyuncusu, diğeri ise oyun kurulumunda daha güçlü bir profil.",
    createdAt: minutesAgo(54),
    likeCount: 65,
    commentCount: 0,
    liked: false,
    comments: [
      {
        id: "seed-2-comment-1",
        author: "sari_lacivertli",
        text: "Top ayağında sakin bir bek daha mantıklı olur gibi.",
        createdAt: minutesAgo(42),
      },
    ],
    commentsExpanded: false,
  },
  {
    id: "seed-3",
    displayName: "Kartal Tribune",
    username: "kartaltribune",
    content:
      "Beşiktaş'ın genç oyunculara verdiği süre artarsa ligin ikinci yarısında çok daha atletik bir yapı görebiliriz. Rotasyon doğru yönetilirse tavan çok yüksek.",
    createdAt: minutesAgo(96),
    likeCount: 38,
    commentCount: 0,
    liked: false,
    comments: [],
    commentsExpanded: false,
  },
  {
    id: "seed-4",
    displayName: "Aslan Gündem",
    username: "aslan_gundem",
    content:
      "Galatasaray'da son haftalarda ön alan pres tetikleyicileri daha net. Forvetin gölge markajı ve 8 numaranın sıçraması rakibin çıkışını ciddi şekilde yavaşlattı.",
    createdAt: minutesAgo(140),
    likeCount: 71,
    commentCount: 0,
    liked: false,
    comments: [
      {
        id: "seed-4-comment-1",
        author: "pas_oyunu",
        text: "Özellikle iç sahada rakipler rahat çıkamıyor, çok doğru tespit.",
        createdAt: minutesAgo(112),
      },
    ],
    commentsExpanded: false,
  },
  {
    id: "seed-5",
    displayName: "Anadolu Scout",
    username: "anadolusc",
    content:
      "Bu hafta alt sıralar kadar Avrupa hattı da çok karışacak. İkili averaj hesapları devreye girince her puan altın değerinde.",
    createdAt: minutesAgo(215),
    likeCount: 24,
    commentCount: 0,
    liked: false,
    comments: [],
    commentsExpanded: false,
  },
].map((post) => ({
  ...post,
  commentCount: post.comments.length,
}));

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
  "Bu hafta en etkili orta saha kurgusu hangisiydi?",
  "3'lü savunma mı 4'lü savunma mı: Süper Lig'e hangisi daha uygun?",
  "Yaz transfer döneminde en kritik ihtiyaç hangi bölge?",
];

const COMMUNITY_RULES = [
  "Saygılı dil kullan, kişisel saldırıdan kaçın.",
  "Kaynaklı haber paylaş, manipülatif başlık açma.",
  "Spam ve aynı içeriği tekrarlamaktan kaçın.",
];

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

export default function Forum() {
  const navigate = useNavigate();
  const location = useLocation();
  const imageInputRef = useRef(null);

  const [posts, setPosts] = useState(FORUM_SEED_POSTS);
  const [composerText, setComposerText] = useState("");
  const [composerImage, setComposerImage] = useState(null);
  const [commentDrafts, setCommentDrafts] = useState({});
  const [shareFeedbackByPost, setShareFeedbackByPost] = useState({});
  const [activeProfileUsername, setActiveProfileUsername] = useState("");
  const [, setFollowStoreVersion] = useState(0);

  useEffect(() => {
    const unsubscribe = subscribeForumFollowStore(() => {
      setFollowStoreVersion((prev) => prev + 1);
    });

    return unsubscribe;
  }, []);

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
      { label: "Arkadaşlar", icon: Users, onClick: () => navigate("/friends") },
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
    Arkadaşlar: ["/friends"],
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

    const newPost = {
      id: `post-${Date.now()}`,
      displayName: "Sen",
      username: "sen",
      content: composerText.trim(),
      createdAt: new Date().toISOString(),
      likeCount: 0,
      commentCount: 0,
      liked: false,
      comments: [],
      commentsExpanded: false,
      imageName: composerImage?.name || "",
      imagePreviewUrl: composerImage?.previewUrl || "",
    };

    setPosts((prevPosts) => [newPost, ...prevPosts]);
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

  const toggleLike = (postId) => {
    setPosts((prevPosts) =>
      prevPosts.map((post) => {
        if (post.id !== postId) return post;
        const nextLiked = !post.liked;
        return {
          ...post,
          liked: nextLiked,
          likeCount: Math.max(0, post.likeCount + (nextLiked ? 1 : -1)),
        };
      })
    );
  };

  const toggleCommentArea = (postId) => {
    setPosts((prevPosts) =>
      prevPosts.map((post) =>
        post.id === postId
          ? {
              ...post,
              commentsExpanded: !post.commentsExpanded,
            }
          : post
      )
    );
  };

  const submitComment = (postId) => {
    const draft = (commentDrafts[postId] || "").trim();
    if (!draft) return;

    const newComment = {
      id: `comment-${Date.now()}`,
      author: "sen",
      text: draft,
      createdAt: new Date().toISOString(),
    };

    setPosts((prevPosts) =>
      prevPosts.map((post) => {
        if (post.id !== postId) return post;
        const nextComments = [...post.comments, newComment];
        return {
          ...post,
          comments: nextComments,
          commentCount: nextComments.length,
          commentsExpanded: true,
        };
      })
    );

    setCommentDrafts((prev) => ({
      ...prev,
      [postId]: "",
    }));
  };

  const handleShare = async (postId) => {
    const targetPost = posts.find((post) => post.id === postId);
    if (!targetPost) return;

    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(`@${targetPost.username}: ${targetPost.content}`);
      }
    } catch {
      // no-op: visual feedback below still works even if clipboard is blocked
    }

    setShareFeedbackByPost((prev) => ({ ...prev, [postId]: true }));

    window.setTimeout(() => {
      setShareFeedbackByPost((prev) => ({ ...prev, [postId]: false }));
    }, 1300);
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
                  İzlek topluluğunun futbol sohbetleri, maç analizleri ve gündem başlıkları burada.
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
                  return (
                    <Card
                    key={post.id}
                    className="overflow-hidden border-border/70 bg-card/95 transition-all duration-200 hover:border-indigo-200/90 hover:shadow-[0_18px_38px_-28px_rgba(79,70,229,0.4)]"
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
                              <span className="font-semibold text-foreground transition-colors duration-200 group-hover:text-indigo-700">{authorProfile.displayName}</span>
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
                      <p className="whitespace-pre-wrap text-sm leading-6 text-slate-700 sm:text-[15px]">{post.content}</p>

                      {post.imagePreviewUrl && (
                        <div className="overflow-hidden rounded-xl border border-border/70 bg-muted/30">
                          <img src={post.imagePreviewUrl} alt={post.imageName || "paylaşım görseli"} className="h-auto w-full object-cover" />
                        </div>
                      )}

                      <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={() => toggleLike(post.id)}
                          className={`h-9 rounded-lg border px-3 text-xs sm:text-sm ${
                            post.liked
                              ? "border-rose-200 bg-rose-50 text-rose-600 hover:bg-rose-100"
                              : "border-transparent text-slate-600 hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700"
                          }`}
                          data-testid={`forum-like-button-${post.id}`}
                        >
                          <Heart className={`h-4 w-4 ${post.liked ? "fill-current" : ""}`} />
                          {post.likeCount}
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
                          {post.commentCount}
                        </Button>

                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={() => handleShare(post.id)}
                          className="h-9 rounded-lg border border-transparent px-3 text-xs text-slate-600 hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700 sm:text-sm"
                          data-testid={`forum-share-button-${post.id}`}
                        >
                          <Repeat2 className="h-4 w-4" />
                          {shareFeedbackByPost[post.id] ? "Paylaşıldı" : "Paylaş"}
                        </Button>
                      </div>

                      {post.commentsExpanded && (
                        <div className="space-y-3 rounded-xl border border-border/70 bg-muted/20 p-3 sm:p-4">
                          {post.comments.length > 0 ? (
                            <div className="space-y-2.5">
                              {post.comments.map((comment) => (
                                <div key={comment.id} className="rounded-lg border border-border/60 bg-background/90 px-3 py-2">
                                  <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                                    <span className="font-semibold text-foreground">@{comment.author}</span>
                                    <span>· {getRelativeTimeLabel(comment.createdAt)}</span>
                                  </div>
                                  <p className="mt-1 text-sm text-slate-700">{comment.text}</p>
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
