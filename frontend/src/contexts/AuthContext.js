import { createContext, useContext, useState, useEffect } from 'react';
import { 
  createUserWithEmailAndPassword, 
  signInWithEmailAndPassword, 
  signOut, 
  onAuthStateChanged,
  updateProfile
} from 'firebase/auth';
import {
  auth,
  hasFirebaseInitializationError,
  isFirebaseConfigured,
  isFirebasePlaceholderConfig,
} from '@/firebase';
import { clearUserData } from '@/lib/storage';

const AuthContext = createContext({});
const DEV_BYPASS_USER = {
  uid: 'dev-user',
  email: 'test@izlek.dev',
  username: 'isa',
  displayName: 'isa',
};

const persistAuthIdentity = (user) => {
  if (typeof window === 'undefined' || !user?.uid) {
    return;
  }

  const resolvedName = user.username || user.displayName || user.email;
  localStorage.setItem('userId', user.uid);
  localStorage.setItem('currentUserId', user.uid);
  localStorage.setItem('userName', resolvedName);
};

const clearAuthIdentity = () => {
  if (typeof window === 'undefined') {
    return;
  }

  localStorage.removeItem('userId');
  localStorage.removeItem('profileId');
  localStorage.removeItem('userName');
  localStorage.removeItem('currentRoomId');
  localStorage.removeItem('currentUserId');
};

const isDevelopmentOrPreviewEnvironment = () => {
  if (typeof window === 'undefined') {
    return process.env.NODE_ENV === 'development';
  }

  const hostname = window.location.hostname;
  const isLocalHost = hostname === 'localhost' || hostname === '127.0.0.1';
  const isPreviewHost = hostname.includes('preview.emergentagent.com');

  return process.env.NODE_ENV === 'development' || isLocalHost || isPreviewHost;
};

export const useAuth = () => useContext(AuthContext);

export function AuthProvider({ children }) {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [firebaseConfigured, setFirebaseConfigured] = useState(false);
  const [authBypassEnabled, setAuthBypassEnabled] = useState(false);

  // Check Firebase configuration on mount
  useEffect(() => {
    const configured = isFirebaseConfigured();
    const canUseDevelopmentBypass = isDevelopmentOrPreviewEnvironment();
    const hasPlaceholderConfig = isFirebasePlaceholderConfig();
    const firebaseInitFailed = hasFirebaseInitializationError() || (configured && !auth);
    const shouldEnableDevelopmentBypass = canUseDevelopmentBypass;

    setFirebaseConfigured(configured && !!auth);
    setAuthBypassEnabled(shouldEnableDevelopmentBypass);

    if (shouldEnableDevelopmentBypass) {
      const bypassReason = hasPlaceholderConfig
        ? 'placeholder-config'
        : !configured
          ? 'missing-config'
          : firebaseInitFailed
            ? 'firebase-init-failed'
            : 'preview-development-mode';

      console.warn(`🔧 DEVELOPMENT/PREVIEW MODE: Mock authentication aktif (${bypassReason})`);
      setCurrentUser(DEV_BYPASS_USER);
      persistAuthIdentity(DEV_BYPASS_USER);
      setLoading(false);
      return;
    }

    if (!configured || !auth || firebaseInitFailed) {
      console.error('Firebase yapılandırması eksik veya başlatılamadı. Lütfen .env dosyasını kontrol edin.');
      setCurrentUser(null);
      clearAuthIdentity();
      setLoading(false);
      return;
    }

    // Subscribe to auth state changes
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setCurrentUser(user);
      
      // Sync userId with Firebase UID when user is logged in
      if (user) {
        persistAuthIdentity(user);
      } else {
        clearAuthIdentity();
      }
      
      setLoading(false);
    });

    return unsubscribe;
  }, []);

  const signup = async (email, password, displayName) => {
    if (authBypassEnabled) {
      setCurrentUser(DEV_BYPASS_USER);
      persistAuthIdentity(DEV_BYPASS_USER);
      return { user: DEV_BYPASS_USER };
    }

    if (!firebaseConfigured || !auth) {
      throw new Error('Firebase yapılandırması eksik');
    }
    
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      
      // Update profile with display name
      if (displayName && userCredential.user) {
        await updateProfile(userCredential.user, { displayName });
      }
      
      return userCredential;
    } catch (error) {
      console.error('Signup error:', error);
      throw error; // Re-throw to let the caller handle it
    }
  };

  const login = async (email, password) => {
    if (authBypassEnabled) {
      setCurrentUser(DEV_BYPASS_USER);
      persistAuthIdentity(DEV_BYPASS_USER);
      return { user: DEV_BYPASS_USER };
    }

    if (!firebaseConfigured || !auth) {
      throw new Error('Firebase yapılandırması eksik');
    }
    
    try {
      return await signInWithEmailAndPassword(auth, email, password);
    } catch (error) {
      // Log the error but don't prevent profile loading from localStorage
      console.error('Login error:', error);
      throw error; // Re-throw to let the caller handle it
    }
  };

  const logout = async () => {
    clearUserData();
    clearAuthIdentity();

    if (authBypassEnabled) {
      setCurrentUser(DEV_BYPASS_USER);
      persistAuthIdentity(DEV_BYPASS_USER);
      return;
    }

    if (!firebaseConfigured || !auth) {
      return;
    }

    // Clear all localStorage data before signing out
    const userId = typeof window === 'undefined' ? null : localStorage.getItem('userId');
    if (userId) {
      // Clear user-specific data
      clearUserData();
    }

    return signOut(auth);
  };

  const value = {
    authBypassEnabled,
    currentUser,
    loading,
    signup,
    login,
    logout,
    firebaseConfigured
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
}

export default AuthContext;