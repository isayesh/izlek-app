// Firebase Configuration
import { getApp, getApps, initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';

const firebaseConfig = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
  authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID,
  storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.REACT_APP_FIREBASE_APP_ID
};

export const isFirebasePlaceholderConfig = () => firebaseConfig.apiKey === 'placeholder';

// Check if Firebase is configured
export const isFirebaseConfigured = () => {
  return !!firebaseConfig.apiKey && !isFirebasePlaceholderConfig() && !!firebaseConfig.projectId;
};

// Initialize Firebase only if configured
let app = null;
let auth = null;
let firebaseInitializationError = null;

if (isFirebaseConfigured()) {
  try {
    app = getApps().length ? getApp() : initializeApp(firebaseConfig);
    auth = getAuth(app);
  } catch (error) {
    firebaseInitializationError = error;
    console.error('Firebase initialization error:', error);
  }
}

export const hasFirebaseInitializationError = () => !!firebaseInitializationError;
export { auth };
export default app;