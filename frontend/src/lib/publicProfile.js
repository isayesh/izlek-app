export const getPublicUsername = (profile) => {
  const username = profile?.username?.trim();
  if (username) return username;

  const displayName = profile?.display_name?.trim();
  if (displayName) return displayName;

  return "Bilinmeyen Kullanıcı";
};

export const formatPublicHandle = (handle) => {
  if (!handle) return "";
  return handle.startsWith("@") ? handle : `@${handle}`;
};

export const formatDailyStudyHours = (value) => {
  if (value === null || value === undefined || value === "") {
    return "Belirtilmedi";
  }

  return `${value} saat`;
};

export const getAvatarFallback = (profile) => {
  return (getPublicUsername(profile) || "İ").trim().charAt(0).toUpperCase();
};