import { createContext, useContext, useState, useEffect } from 'react';
import { 
  createUserWithEmailAndPassword, 
  signInWithEmailAndPassword, 
  signOut, 
  onAuthStateChanged,
  updateProfile
} from 'firebase/auth';
import { auth, isFirebaseConfigured } from '@/firebase';
import { clearUserData } from '@/lib/storage';

const AuthContext = createContext({});

export const useAuth = () => useContext(AuthContext);

export function AuthProvider({ children }) {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [firebaseConfigured, setFirebaseConfigured] = useState(false);

  // Check Firebase configuration on mount
  useEffect(() => {
    const configured = isFirebaseConfigured();
    setFirebaseConfigured(configured);
    
    if (!configured) {
      console.error('Firebase yapılandırması eksik. Lütfen .env dosyasını kontrol edin.');
      setLoading(false);
      return;
    }

    // Subscribe to auth state changes
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setCurrentUser(user);
      
      // Sync userId with Firebase UID when user is logged in
      if (user) {
        localStorage.setItem('userId', user.uid);
      }
      
      setLoading(false);
    });

    return unsubscribe;
  }, []);

  const signup = async (email, password, displayName) => {
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
    if (!firebaseConfigured || !auth) {
      return;
    }
    
    // Clear all localStorage data before signing out
    const userId = localStorage.getItem('userId');
    if (userId) {
      // Clear user-specific data
      clearUserData();
    }
    
    // Clear all auth-related localStorage keys
    localStorage.removeItem('userId');
    localStorage.removeItem('profileId');
    localStorage.removeItem('userName');
    localStorage.removeItem('currentRoomId');
    localStorage.removeItem('currentUserId');
    
    return signOut(auth);
  };

  const value = {
    currentUser,
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