const minutesAgo = (minutes) => new Date(Date.now() - minutes * 60 * 1000).toISOString();

const FORUM_USERS = {
  sen: {
    displayName: "Sen",
    username: "sen",
    bio: "Düzenli çalışıp netlerini istikrarlı artırmaya odaklanıyorsun.",
    studyFocus: "AYT Matematik",
    stats: {
      posts: 8,
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

const notifyFollowListeners = () => {
  followListeners.forEach((listener) => listener());
};

export const subscribeForumFollowStore = (listener) => {
  followListeners.add(listener);
  return () => {
    followListeners.delete(listener);
  };
};

export const isForumFollowing = (username = "") => Boolean(followStateByUsername[username]);

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

export const getForumUserProfile = (username = "", fallbackDisplayName = "") => {
  const knownProfile = FORUM_USERS[username];
  if (knownProfile) return knownProfile;
  return defaultProfile(username || "kullanici", fallbackDisplayName || "İzlek Kullanıcısı");
};

export const getForumUserStats = (username = "") => {
  const profile = getForumUserProfile(username);
  return {
    posts: profile.stats.posts,
    followers: Math.max(0, profile.stats.followers + (followerDeltaByUsername[username] || 0)),
    following: profile.stats.following,
    studyHours: profile.stats.studyHours,
  };
};

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
