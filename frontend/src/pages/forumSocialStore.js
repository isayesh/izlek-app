const minutesAgo = (minutes) => new Date(Date.now() - minutes * 60 * 1000).toISOString();

const FORUM_USERS = {
  sen: {
    displayName: "Sen",
    username: "sen",
    bio: "Düzenli çalışıp netlerini istikrarlı artırmaya odaklanıyorsun.",
    studyFocus: "AYT Matematik",
    stats: {
      posts: 0,
      followers: 52,
      following: 37,
      studyHours: 214,
    },
    followersList: ["merttaktik", "aslan_gundem", "sari_lacivertli"],
    followingList: ["merttaktik", "transferhatti", "kartaltribune"],
  },
  merttaktik: {
    displayName: "Mert Analiz",
    username: "merttaktik",
    bio: "TYT-AYT deneme analizleri ve haftalık çalışma notları paylaşıyorum.",
    studyFocus: "TYT Matematik",
    stats: {
      posts: 34,
      followers: 128,
      following: 76,
      studyHours: 412,
    },
    followersList: ["sen", "aslan_gundem", "denemeperisi", "sari_lacivertli"],
    followingList: ["transferhatti", "denemeperisi", "paragrafci"],
  },
  transferhatti: {
    displayName: "Transfer Radarı",
    username: "transferhatti",
    bio: "Günlük rutin paylaşımı ve AYT net artırma stratejileri.",
    studyFocus: "AYT Edebiyat",
    stats: {
      posts: 21,
      followers: 96,
      following: 63,
      studyHours: 355,
    },
    followersList: ["sen", "merttaktik", "paragrafci"],
    followingList: ["merttaktik", "kartaltribune", "anadolusc"],
  },
  kartaltribune: {
    displayName: "Kartal Tribune",
    username: "kartaltribune",
    bio: "Pomodoro disiplini ve odak yönetimi üzerine paylaşımlar.",
    studyFocus: "Pomodoro Planlama",
    stats: {
      posts: 17,
      followers: 72,
      following: 49,
      studyHours: 289,
    },
    followersList: ["merttaktik", "sen"],
    followingList: ["aslan_gundem", "transferhatti"],
  },
  aslan_gundem: {
    displayName: "Aslan Gündem",
    username: "aslan_gundem",
    bio: "Stres yönetimi, motivasyon ve haftalık deneme raporları.",
    studyFocus: "Deneme Analizi",
    stats: {
      posts: 29,
      followers: 143,
      following: 58,
      studyHours: 467,
    },
    followersList: ["sen", "merttaktik", "denemeperisi", "paragrafci"],
    followingList: ["merttaktik", "transferhatti", "kartaltribune"],
  },
  anadolusc: {
    displayName: "Anadolu Scout",
    username: "anadolusc",
    bio: "Çalışma programı, rutin kurma ve günlük takip sistemleri.",
    studyFocus: "Çalışma Programı",
    stats: {
      posts: 14,
      followers: 67,
      following: 44,
      studyHours: 242,
    },
    followersList: ["transferhatti", "sen"],
    followingList: ["merttaktik", "paragrafci"],
  },
  denemeperisi: {
    displayName: "Deneme Perisi",
    username: "denemeperisi",
    bio: "Haftalık net tabloları ve yanlış analizi notları.",
    studyFocus: "TYT Deneme",
    stats: {
      posts: 19,
      followers: 88,
      following: 52,
      studyHours: 301,
    },
    followersList: ["merttaktik", "aslan_gundem"],
    followingList: ["sen", "transferhatti"],
  },
  paragrafci: {
    displayName: "Paragrafçı",
    username: "paragrafci",
    bio: "Paragraf hızlandırma teknikleri ve soru çözüm rutinleri.",
    studyFocus: "TYT Türkçe",
    stats: {
      posts: 25,
      followers: 104,
      following: 61,
      studyHours: 338,
    },
    followersList: ["sen", "transferhatti"],
    followingList: ["merttaktik", "aslan_gundem"],
  },
  sari_lacivertli: {
    displayName: "Sarı Lacivertli",
    username: "sari_lacivertli",
    bio: "Soru çözüm maratonu ve gece çalışma rutinleri.",
    studyFocus: "Soru Çözümü",
    stats: {
      posts: 11,
      followers: 46,
      following: 35,
      studyHours: 198,
    },
    followersList: ["merttaktik"],
    followingList: ["sen", "kartaltribune"],
  },
};

const defaultProfile = (username = "kullanici", displayName = "İzlek Kullanıcısı") => ({
  displayName,
  username,
  bio: "Henüz biyografi eklenmedi.",
  studyFocus: "Genel Çalışma",
  stats: {
    posts: 0,
    followers: 0,
    following: 0,
    studyHours: 0,
  },
  followersList: [],
  followingList: [],
});

const followStateByUsername = {};
const followerDeltaByUsername = {};
const followListeners = new Set();
const initialCurrentUserFollowing = new Set(FORUM_USERS.sen?.followingList || []);

const initialForumFeedPosts = [
  {
    id: "seed-1",
    displayName: "Mert Analiz",
    username: "merttaktik",
    content:
      "Derbide 4-2-3-1 yerine ikinci yarı 4-3-3'e dönüş kritik oldu. Orta sahada bir ekstra oyuncu kazanınca hem ikinci toplar hem de geçiş savunması toparlandı.",
    createdAt: minutesAgo(18),
    likeCount: 42,
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
    imageName: "",
    imagePreviewUrl: "",
  },
  {
    id: "seed-2",
    displayName: "Transfer Radarı",
    username: "transferhatti",
    content:
      "Fenerbahçe'nin sol bek rotasyonu için iki isim daha gündeme girmiş. Biri tempolu çizgi oyuncusu, diğeri ise oyun kurulumunda daha güçlü bir profil.",
    createdAt: minutesAgo(54),
    likeCount: 65,
    liked: false,
    comments: [
      {
        id: "seed-2-comment-1",
        author: "sari_lacivertli",
        text: "Top ayağında sakin bir bek daha mantıklı olur gibi.",
        createdAt: minutesAgo(42),
      },
    ],
    imageName: "",
    imagePreviewUrl: "",
  },
  {
    id: "seed-3",
    displayName: "Kartal Tribune",
    username: "kartaltribune",
    content:
      "Beşiktaş'ın genç oyunculara verdiği süre artarsa ligin ikinci yarısında çok daha atletik bir yapı görebiliriz. Rotasyon doğru yönetilirse tavan çok yüksek.",
    createdAt: minutesAgo(96),
    likeCount: 38,
    liked: false,
    comments: [],
    imageName: "",
    imagePreviewUrl: "",
  },
  {
    id: "seed-4",
    displayName: "Aslan Gündem",
    username: "aslan_gundem",
    content:
      "Galatasaray'da son haftalarda ön alan pres tetikleyicileri daha net. Forvetin gölge markajı ve 8 numaranın sıçraması rakibin çıkışını ciddi şekilde yavaşlattı.",
    createdAt: minutesAgo(140),
    likeCount: 71,
    liked: false,
    comments: [
      {
        id: "seed-4-comment-1",
        author: "pas_oyunu",
        text: "Özellikle iç sahada rakipler rahat çıkamıyor, çok doğru tespit.",
        createdAt: minutesAgo(112),
      },
    ],
    imageName: "",
    imagePreviewUrl: "",
  },
  {
    id: "seed-5",
    displayName: "Anadolu Scout",
    username: "anadolusc",
    content:
      "Bu hafta alt sıralar kadar Avrupa hattı da çok karışacak. İkili averaj hesapları devreye girince her puan altın değerinde.",
    createdAt: minutesAgo(215),
    likeCount: 24,
    liked: false,
    comments: [],
    imageName: "",
    imagePreviewUrl: "",
  },
].map((post) => ({
  ...post,
  commentCount: post.comments.length,
}));

let forumFeedPosts = [...initialForumFeedPosts];
const forumFeedListeners = new Set();

const notifyFollowListeners = () => {
  followListeners.forEach((listener) => listener());
};

export const subscribeForumFollowStore = (listener) => {
  followListeners.add(listener);
  return () => {
    followListeners.delete(listener);
  };
};

export const isForumFollowing = (username = "") => {
  if (!username || username === "sen") {
    return false;
  }

  if (Object.prototype.hasOwnProperty.call(followStateByUsername, username)) {
    return Boolean(followStateByUsername[username]);
  }

  return initialCurrentUserFollowing.has(username);
};

export const toggleForumFollow = (username = "") => {
  if (!username || username === "sen") {
    return false;
  }

  const nextFollowing = !isForumFollowing(username);
  followStateByUsername[username] = nextFollowing;
  followerDeltaByUsername[username] = (followerDeltaByUsername[username] || 0) + (nextFollowing ? 1 : -1);
  notifyFollowListeners();
  return nextFollowing;
};

const notifyForumFeedListeners = () => {
  forumFeedListeners.forEach((listener) => listener());
};

export const subscribeForumFeedStore = (listener) => {
  forumFeedListeners.add(listener);
  return () => {
    forumFeedListeners.delete(listener);
  };
};

export const getForumFeedPosts = () =>
  forumFeedPosts.map((post) => ({
    ...post,
    comments: [...(post.comments || [])],
  }));

export const createForumPost = ({
  displayName = "Sen",
  username = "sen",
  content = "",
  imageName = "",
  imagePreviewUrl = "",
}) => {
  const trimmedContent = content.trim();
  if (!trimmedContent) return null;

  const nextPost = {
    id: `post-${Date.now()}`,
    displayName,
    username,
    content: trimmedContent,
    createdAt: new Date().toISOString(),
    likeCount: 0,
    liked: false,
    comments: [],
    commentCount: 0,
    imageName,
    imagePreviewUrl,
  };

  forumFeedPosts = [nextPost, ...forumFeedPosts];
  notifyForumFeedListeners();
  return nextPost;
};

export const toggleForumPostLike = (postId = "") => {
  if (!postId) return;

  forumFeedPosts = forumFeedPosts.map((post) => {
    if (post.id !== postId) return post;
    const nextLiked = !post.liked;
    return {
      ...post,
      liked: nextLiked,
      likeCount: Math.max(0, post.likeCount + (nextLiked ? 1 : -1)),
    };
  });

  notifyForumFeedListeners();
};

export const addForumPostComment = (postId = "", text = "", author = "sen") => {
  const trimmedText = text.trim();
  if (!postId || !trimmedText) return null;

  const nextComment = {
    id: `comment-${Date.now()}`,
    author,
    text: trimmedText,
    createdAt: new Date().toISOString(),
  };

  forumFeedPosts = forumFeedPosts.map((post) => {
    if (post.id !== postId) return post;
    const nextComments = [...(post.comments || []), nextComment];
    return {
      ...post,
      comments: nextComments,
      commentCount: nextComments.length,
    };
  });

  notifyForumFeedListeners();
  return nextComment;
};

export const getForumCurrentUserComments = (username = "sen") =>
  forumFeedPosts.flatMap((post) =>
    (post.comments || [])
      .filter((comment) => comment.author === username)
      .map((comment) => ({
        ...comment,
        postId: post.id,
        postContent: post.content,
      }))
  );

export const getForumCurrentUserLikedPosts = () => forumFeedPosts.filter((post) => post.liked);

export const getForumCurrentUserPosts = (username = "sen") =>
  forumFeedPosts.filter((post) => post.username === username);

export const getForumUserProfile = (username = "", fallbackDisplayName = "") => {
  const knownProfile = FORUM_USERS[username];
  if (knownProfile) return knownProfile;
  return defaultProfile(username || "kullanici", fallbackDisplayName || "İzlek Kullanıcısı");
};

export const getForumUserStats = (username = "") => {
  const profile = getForumUserProfile(username);
  const dynamicPostCount = forumFeedPosts.filter((post) => post.username === username).length;
  const resolvedPostCount = username === "sen" ? dynamicPostCount : Math.max(profile.stats.posts, dynamicPostCount);

  return {
    posts: resolvedPostCount,
    followers: Math.max(0, profile.stats.followers + (followerDeltaByUsername[username] || 0)),
    following: profile.stats.following,
    studyHours: profile.stats.studyHours,
  };
};

export const doesForumUserFollowCurrentUser = (username = "") => {
  if (!username || username === "sen") {
    return false;
  }

  const profile = getForumUserProfile(username);
  return (profile.followingList || []).includes("sen");
};

export const isForumMutualFollow = (username = "") =>
  isForumFollowing(username) && doesForumUserFollowCurrentUser(username);

export const getForumUsersByUsernames = (usernames = []) =>
  usernames.map((username) => getForumUserProfile(username));

export const PUBLIC_PROFILE_POSTS = [
  {
    id: "profile-post-1",
    username: "merttaktik",
    content: "Bugün 2 deneme çözdüm. Yanlışların %70'i zaman yönetiminden geldi, yarın ilk hedef süreyi iyileştirmek.",
    createdAt: minutesAgo(180),
    likeCount: 16,
    commentCount: 4,
  },
  {
    id: "profile-post-2",
    username: "merttaktik",
    content: "TYT problem setinde 40 soruda 37 doğru. Hız için önce kolay-orta-zor sıralaması bende iyi çalıştı.",
    createdAt: minutesAgo(900),
    likeCount: 23,
    commentCount: 6,
  },
  {
    id: "profile-post-3",
    username: "aslan_gundem",
    content: "Son 1 haftadır pomodoro 40/10 modeline geçtim, odak sürem belirgin arttı.",
    createdAt: minutesAgo(320),
    likeCount: 19,
    commentCount: 3,
  },
  {
    id: "profile-post-4",
    username: "transferhatti",
    content: "AYT edebiyat tekrar planını 3 güne böldüm. Konu + soru + kısa analiz üçlüsü daha verimli geldi.",
    createdAt: minutesAgo(1440),
    likeCount: 12,
    commentCount: 2,
  },
];

export const PUBLIC_PROFILE_ACTIVITY = {
  merttaktik: {
    comments: [
      { id: "c1", text: "Bu deneme grafiği çok iyi, ben de benzer tablo kullanıyorum.", createdAt: minutesAgo(220) },
      { id: "c2", text: "Paragraf hız tekniği önerin varsa paylaşır mısın?", createdAt: minutesAgo(730) },
    ],
    likes: [
      { id: "l1", text: "‘Pomodoro 50/10 mu 40/10 mu?’ başlığını beğendi." },
      { id: "l2", text: "‘AYT net artış rutini’ paylaşımını beğendi." },
    ],
  },
};
?’ başlığını beğendi." },
      { id: "l2", text: "‘AYT net artış rutini’ paylaşımını beğendi." },
    ],
  },
};
