/**
 * localStorage Persistence Layer
 * Simple cache to prevent data loss on refresh
 * Binds all user data to userId
 */

const STORAGE_PREFIX = 'userData_';

/**
 * Get current userId from localStorage
 */
export const getUserId = () => {
  return localStorage.getItem('userId');
};

/**
 * Get user data key based on userId
 */
const getUserDataKey = () => {
  const userId = getUserId();
  if (!userId) return null;
  return `${STORAGE_PREFIX}${userId}`;
};

/**
 * Get all user data from localStorage
 */
export const getUserData = () => {
  const key = getUserDataKey();
  if (!key) return null;
  
  try {
    const data = localStorage.getItem(key);
    return data ? JSON.parse(data) : null;
  } catch (error) {
    console.error('Error reading user data from localStorage:', error);
    return null;
  }
};

/**
 * Save user data to localStorage
 */
export const saveUserData = (data) => {
  const key = getUserDataKey();
  if (!key) return false;
  
  try {
    const existing = getUserData() || {};
    const updated = { ...existing, ...data };
    localStorage.setItem(key, JSON.stringify(updated));
    return true;
  } catch (error) {
    console.error('Error saving user data to localStorage:', error);
    return false;
  }
};

/**
 * Get programs from localStorage
 */
export const getStoredPrograms = () => {
  const userData = getUserData();
  return userData?.programs || [];
};

/**
 * Save programs to localStorage
 */
export const savePrograms = (programs) => {
  return saveUserData({ programs });
};

/**
 * Get profile data from localStorage
 */
export const getStoredProfile = () => {
  const userData = getUserData();
  return {
    profileId: userData?.profileId || localStorage.getItem('profileId'),
    userName: userData?.userName || localStorage.getItem('userName')
  };
};

/**
 * Save profile data to localStorage
 */
export const saveProfile = (profileId, userName) => {
  // Keep backward compatibility with existing localStorage keys
  localStorage.setItem('profileId', profileId);
  localStorage.setItem('userName', userName);
  
  // Also save in userData
  return saveUserData({ profileId, userName });
};

/**
 * Get room data from localStorage
 */
export const getStoredRoom = () => {
  const userData = getUserData();
  return {
    currentRoomId: userData?.currentRoomId || localStorage.getItem('currentRoomId'),
    currentUserId: userData?.currentUserId || localStorage.getItem('currentUserId')
  };
};

/**
 * Save room data to localStorage
 */
export const saveRoom = (roomId, userId) => {
  // Keep backward compatibility with existing localStorage keys
  localStorage.setItem('currentRoomId', roomId);
  localStorage.setItem('currentUserId', userId);
  
  // Also save in userData
  return saveUserData({ currentRoomId: roomId, currentUserId: userId });
};

/**
 * Clear all user data (useful for logout/reset)
 */
export const clearUserData = () => {
  const key = getUserDataKey();
  if (key) {
    localStorage.removeItem(key);
  }
};

/**
 * Get program creation draft from localStorage
 */
export const getProgramDraft = () => {
  const userData = getUserData();
  return userData?.programDraft || null;
};

/**
 * Save program creation draft to localStorage
 */
export const saveProgramDraft = (draftData) => {
  return saveUserData({ programDraft: draftData });
};

/**
 * Clear program creation draft from localStorage
 */
export const clearProgramDraft = () => {
  const userData = getUserData();
  if (userData && userData.programDraft) {
    const { programDraft, ...rest } = userData;
    const key = getUserDataKey();
    if (key) {
      try {
        localStorage.setItem(key, JSON.stringify(rest));
        return true;
      } catch (error) {
        console.error('Error clearing program draft:', error);
        return false;
      }
    }
  }
  return false;
};

/**
 * Check if onboarding has been completed
 * This prevents re-showing onboarding on auth/network errors
 */
export const isOnboardingCompleted = () => {
  const userId = getUserId();
  if (!userId) return false;
  
  // Check if profile exists in localStorage
  const profileId = localStorage.getItem('profileId');
  if (profileId) return true;
  
  // Also check userData
  const userData = getUserData();
  return !!(userData?.profileId || userData?.onboardingCompleted);
};

/**
 * Mark onboarding as completed
 */
export const setOnboardingCompleted = () => {
  localStorage.setItem('onboardingCompleted', 'true');
  return saveUserData({ onboardingCompleted: true });
};
